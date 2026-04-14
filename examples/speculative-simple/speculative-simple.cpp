#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <cctype>
#include <clocale>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

static std::optional<std::string> env_str_nonempty(const char * name) {
    const char * value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return std::nullopt;
    }
    return std::string(value);
}

static bool env_flag(const char * name) {
    const auto value = env_str_nonempty(name);
    if (!value.has_value()) {
        return false;
    }
    std::string lower = value.value();
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char ch) {
        return (char) std::tolower(ch);
    });
    return lower == "1" || lower == "true" || lower == "yes" || lower == "on";
}

static int env_int(const char * name, int default_value) {
    const auto value = env_str_nonempty(name);
    if (!value.has_value()) {
        return default_value;
    }
    return std::stoi(*value);
}

static bool sampling_is_greedy(const common_params_sampling & sampling) {
    return sampling.temp <= 0.0f &&
           sampling.top_k <= 1 &&
           sampling.top_p >= 1.0f &&
           sampling.mirostat == 0;
}

static llama_token sample_and_accept_current(struct common_sampler * smpl, struct llama_context * ctx) {
    const llama_token token = common_sampler_sample(smpl, ctx, -1);
    common_sampler_accept(smpl, token, true);
    return token;
}

static llama_token greedy_from_logits(const llama_vocab * vocab, const float * logits) {
    const int n_vocab = llama_vocab_n_tokens(vocab);
    llama_token best = 0;
    float best_logit = -INFINITY;
    for (int i = 0; i < n_vocab; ++i) {
        if (llama_vocab_is_eog(vocab, i)) {
            continue;
        }
        if (logits[i] > best_logit) {
            best_logit = logits[i];
            best = i;
        }
    }
    return best;
}

struct tree_leaf_path {
    int seq_id = 0;
    uint32_t leaf_node_id = 0;
    std::vector<const common_speculative_tree_node *> nodes;
};

static std::vector<tree_leaf_path> build_tree_leaf_paths(const common_speculative_tree & tree) {
    std::unordered_map<uint32_t, std::vector<const common_speculative_tree_node *>> children;
    children.reserve(tree.nodes.size());
    for (const auto & node : tree.nodes) {
        children[node.parent_id].push_back(&node);
    }

    std::vector<tree_leaf_path> result;
    std::vector<const common_speculative_tree_node *> stack;

    std::function<void(uint32_t)> dfs = [&](uint32_t parent_id) {
        auto it = children.find(parent_id);
        if (it == children.end() || it->second.empty()) {
            if (stack.empty()) {
                result.push_back(tree_leaf_path{});
            } else {
                tree_leaf_path path;
                path.leaf_node_id = stack.back()->node_id;
                path.nodes = stack;
                result.push_back(std::move(path));
            }
            return;
        }

        for (const auto * child : it->second) {
            stack.push_back(child);
            dfs(child->node_id);
            stack.pop_back();
        }
    };

    dfs(0);

    for (size_t i = 0; i < result.size(); ++i) {
        result[i].seq_id = (int) i;
    }

    return result;
}

struct tree_verify_result {
    llama_tokens ids;
    int proposed = 0;
    int accepted_from_draft = 0;
    uint32_t committed_node_id = 0;
    int committed_seq_id = 0;
    llama_token next_token = 0;
};

static tree_verify_result verify_tree_batched(
        llama_context * ctx_tgt,
        llama_batch & batch_tgt,
        llama_memory_t mem_tgt,
        common_sampler * smpl,
        const llama_vocab * vocab,
        int & n_past,
        const common_speculative_tree & tree) {
    tree_verify_result result;
    result.proposed = 1 + (int) tree.nodes.size();

    const auto leaf_paths = build_tree_leaf_paths(tree);
    const int n_leaf = std::max<int>(1, leaf_paths.size());
    const int n_past_base = n_past;

    for (int seq = 1; seq < n_leaf; ++seq) {
        llama_memory_seq_rm(mem_tgt, seq, -1, -1);
        llama_memory_seq_cp(mem_tgt, 0, seq, -1, -1);
    }

    common_batch_clear(batch_tgt);

    int idx_after_seed = -1;
    std::unordered_map<uint32_t, int> node_batch_idx;
    std::unordered_map<uint32_t, int> node_seq;
    node_batch_idx.reserve(tree.nodes.size());
    node_seq.reserve(tree.nodes.size());

    for (const auto & path : leaf_paths) {
        common_batch_add(batch_tgt, tree.seed_token, n_past_base, { path.seq_id }, true);
        if (idx_after_seed < 0) {
            idx_after_seed = batch_tgt.n_tokens - 1;
        }
    }

    size_t max_nodes = 0;
    for (const auto & path : leaf_paths) {
        max_nodes = std::max(max_nodes, path.nodes.size());
    }

    for (size_t depth = 0; depth < max_nodes; ++depth) {
        for (const auto & path : leaf_paths) {
            if (depth >= path.nodes.size()) {
                continue;
            }
            const auto * node = path.nodes[depth];
            common_batch_add(batch_tgt, node->token, n_past_base + 1 + (int) depth, { path.seq_id }, true);
            node_batch_idx.emplace(node->node_id, batch_tgt.n_tokens - 1);
            node_seq.emplace(node->node_id, path.seq_id);
        }
    }

    if (llama_decode(ctx_tgt, batch_tgt) != 0) {
        throw std::runtime_error("llama_decode failed in batched tree verification");
    }

    uint32_t parent_id = 0;
    int committed_seq = 0;
    result.ids.push_back(tree.seed_token);
    result.accepted_from_draft = 1;
    int repr_idx = idx_after_seed;
    int committed_tokens = 1;
    common_sampler_accept(smpl, tree.seed_token, true);
    llama_token predicted = common_sampler_sample(smpl, ctx_tgt, repr_idx);
    result.next_token = predicted;

    while (true) {
        const common_speculative_tree_node * matched = nullptr;
        for (const auto & node : tree.nodes) {
            if (node.parent_id == parent_id && node.token == predicted) {
                matched = &node;
                break;
            }
        }

        if (matched == nullptr) {
            break;
        }

        parent_id = matched->node_id;
        result.committed_node_id = matched->node_id;
        committed_seq = node_seq.at(matched->node_id);
        repr_idx = node_batch_idx.at(matched->node_id);
        result.ids.push_back(predicted);
        result.accepted_from_draft += 1;
        committed_tokens += 1;
        common_sampler_accept(smpl, predicted, true);

        predicted = common_sampler_sample(smpl, ctx_tgt, repr_idx);
        result.next_token = predicted;

        bool has_children = false;
        for (const auto & node : tree.nodes) {
            if (node.parent_id == parent_id) {
                has_children = true;
                break;
            }
        }
        if (!has_children) {
            break;
        }
    }

    result.ids.push_back(result.next_token);
    result.committed_seq_id = committed_seq;

    if (result.accepted_from_draft > 0) {
        const int n_keep = n_past_base + committed_tokens;
        const int keep_seq = result.committed_node_id == 0 ? 0 : committed_seq;
        llama_memory_seq_rm(mem_tgt, keep_seq, n_keep, -1);
        llama_memory_seq_keep(mem_tgt, keep_seq);
        if (keep_seq != 0) {
            llama_memory_seq_cp(mem_tgt, keep_seq, 0, -1, -1);
        }
        llama_memory_seq_keep(mem_tgt, 0);
        n_past = n_keep;
    }

    return result;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }

    if (params.n_predict < -1) {
        LOG_ERR("%s: --n-predict must be >= -1\n", __func__);
        return 1;
    }

    if (params.speculative.mparams_dft.path.empty()) {
        LOG_ERR("%s: --model-draft is required\n", __func__);
        return 1;
    }

    const bool use_tree = env_flag("LLAMA_GEMMA4_DRAFT_TREE");
    const int tree_width = std::max(1, env_int("LLAMA_GEMMA4_TREE_WIDTH", 2));
    const int tree_depth = std::max(0, env_int("LLAMA_GEMMA4_TREE_DEPTH", 2));
    const int tree_plain_tail = std::max(0, env_int("LLAMA_GEMMA4_TREE_PLAIN_TAIL", 8));

    if (use_tree) {
        const int tree_seq_budget = std::max(1, tree_width * std::max(1, tree_depth));
        params.n_parallel = std::max(params.n_parallel, tree_seq_budget);
        params.n_batch = std::max(params.n_batch, tree_width * (tree_depth + 2) + tree_plain_tail + 8);
    }

    // init llama.cpp
    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model * model_tgt = NULL;

    llama_context * ctx_tgt = NULL;

    // load the target model
    auto llama_init_tgt = common_init_from_params(params);

    model_tgt = llama_init_tgt->model();
    ctx_tgt   = llama_init_tgt->context();

    const llama_vocab * vocab = llama_model_get_vocab(model_tgt);

    // load the draft model
    llama_model_ptr model_dft;

    // TODO: simplify this logic
    {
        const auto & params_spec = params.speculative;

        auto params_dft = params;

        params_dft.n_parallel   = 1;
        params_dft.n_ctx        = params_spec.n_ctx;
        params_dft.n_batch      = llama_n_ctx_seq(ctx_tgt);
        params_dft.devices      = params_spec.devices;
        params_dft.model        = params_spec.mparams_dft;
        params_dft.n_gpu_layers = params_spec.n_gpu_layers;

        if (params_spec.cpuparams.n_threads > 0) {
            params_dft.cpuparams.n_threads       = params.speculative.cpuparams.n_threads;
            params_dft.cpuparams_batch.n_threads = params.speculative.cpuparams_batch.n_threads;
        }

        params_dft.tensor_buft_overrides = params.speculative.tensor_buft_overrides;

        auto mparams_dft = common_model_params_to_llama(params_dft);

        model_dft.reset(llama_model_load_from_file(params_dft.model.path.c_str(), mparams_dft));
        if (model_dft == nullptr) {
            LOG_ERR("failed to load draft model, '%s'\n", params_dft.model.path.c_str());
            return 1;
        }

        params.speculative.model_dft = model_dft.get();
        params.speculative.cparams_dft = common_context_params_to_llama(params_dft);
    }

    // Tokenize the prompt
    std::vector<llama_token> inp;
    inp = common_tokenize(ctx_tgt, params.prompt, true, true);

    if (llama_n_ctx(ctx_tgt) < (uint32_t) inp.size()) {
        LOG_ERR("%s: the prompt exceeds the context size (%d tokens, ctx %d)\n", __func__, (int) inp.size(), llama_n_ctx(ctx_tgt));

        return 1;
    }

    if (llama_n_batch(ctx_tgt) < (uint32_t) inp.size()) {
        LOG_ERR("%s: the prompt exceeds the batch size (%d tokens, batch %d)\n", __func__, (int) inp.size(), llama_n_batch(ctx_tgt));

        return 1;
    }

    LOG("\n\n");

    for (auto id : inp) {
        LOG("%s", common_token_to_piece(ctx_tgt, id).c_str());
    }

    int n_predict = 0;
    int n_drafted = 0;
    int n_accept  = 0;

    // used to determine end of generation
    bool has_eos = false;

    // ================================================
    // everything until here is standard initialization
    // the relevant stuff for speculative decoding starts here

    const auto t_enc_start = ggml_time_us();

    // target model sampling context
    struct common_sampler * smpl = common_sampler_init(model_tgt, params.sampling);

    // eval the prompt
    llama_decode(ctx_tgt, llama_batch_get_one(inp.data(), inp.size() - 1));

    // note: keep the last token separate!
    llama_token id_last = inp.back();

    // all tokens currently in the target context
    llama_tokens prompt_tgt(inp.begin(), inp.end() - 1);
    prompt_tgt.reserve(llama_n_ctx(ctx_tgt));

    int n_past = inp.size() - 1;

    // init the speculator
    const auto & params_spec = params.speculative;

    struct common_speculative * spec = common_speculative_init(params.speculative, ctx_tgt);

    common_speculative_begin(spec, prompt_tgt);

    const int batch_cap = std::max<int>(llama_n_batch(ctx_tgt), tree_width * (tree_depth + 2) + tree_plain_tail + 8);
    llama_batch batch_tgt = llama_batch_init(batch_cap, 0, std::max(1, params.n_parallel));
    llama_memory_t mem_tgt = llama_get_memory(ctx_tgt);

    if (use_tree) {
        if (!sampling_is_greedy(params.sampling)) {
            LOG_ERR("%s: LLAMA_GEMMA4_DRAFT_TREE currently requires greedy decoding (--temp 0 --top-k 1 --top-p 1 --mirostat 0)\n", __func__);
            common_speculative_free(spec);
            common_sampler_free(smpl);
            llama_batch_free(batch_tgt);
            llama_backend_free();
            return 1;
        }
        if (!common_speculative_supports_tree(spec)) {
            LOG_ERR("%s: LLAMA_GEMMA4_DRAFT_TREE is enabled but the speculative backend does not support tree preview\n", __func__);
            common_speculative_free(spec);
            common_sampler_free(smpl);
            llama_batch_free(batch_tgt);
            llama_backend_free();
            return 1;
        }
    }

    auto decode_target = [&](const llama_tokens & tokens, bool output_all) {
        if (tokens.empty()) {
            return;
        }
        common_batch_clear(batch_tgt);
        for (size_t i = 0; i < tokens.size(); ++i) {
            const bool want_logits = output_all || (i + 1 == tokens.size());
            common_batch_add(batch_tgt, tokens[i], n_past + (int) i, { 0 }, want_logits);
        }
        if (llama_decode(ctx_tgt, batch_tgt) != 0) {
            throw std::runtime_error("llama_decode failed");
        }
        n_past += (int) tokens.size();
    };

    const auto t_enc_end = ggml_time_us();

    const auto t_dec_start = ggml_time_us();

    while (true) {
        llama_tokens ids;
        int draft_count = 0;

        if (use_tree) {
            const auto seed_level = common_speculative_preview_tree_level(spec, tree_width);
            if (!seed_level.expects_seed || seed_level.candidates.size() != 1) {
                throw std::runtime_error("tree speculative backend returned an invalid seed level");
            }

            decode_target({ id_last }, false);

            if (seed_level.candidates.front().token != id_last) {
                common_speculative_discard(spec);
                common_speculative_accept_committed(spec, { id_last });
                ids.push_back(sample_and_accept_current(smpl, ctx_tgt));
            } else {
                common_speculative_commit_tree_token(spec, id_last);

                const auto tree = common_speculative_preview_tree(spec, tree_width, tree_depth);
                std::unique_ptr<common_sampler, decltype(&common_sampler_free)> tree_smpl(
                        common_sampler_clone(smpl),
                        common_sampler_free);
                const llama_token predicted_seed = common_sampler_sample(tree_smpl.get(), ctx_tgt, -1);

                tree_verify_result tree_result;
                if (predicted_seed != tree.seed_token) {
                    common_speculative_discard(spec);
                    tree_result.proposed = 1 + (int) tree.nodes.size();
                    tree_result.ids = { predicted_seed };
                    tree_result.next_token = predicted_seed;
                } else {
                    tree_result = verify_tree_batched(ctx_tgt, batch_tgt, mem_tgt, tree_smpl.get(), vocab, n_past, tree);
                }

                draft_count += tree_result.proposed;
                n_drafted += tree_result.proposed;
                n_accept += tree_result.accepted_from_draft;

                if (tree_result.accepted_from_draft == 0) {
                    common_speculative_discard(spec);
                    ids = tree_result.ids;
                } else {
                    common_speculative_commit_tree_node(spec, tree_result.committed_node_id);
                    ids.assign(tree_result.ids.begin(), tree_result.ids.end() - 1);

                    bool used_tail = false;
                    if (tree_plain_tail > 0) {
                        const int remaining = std::max(0, params_spec.n_max - tree_result.accepted_from_draft);
                        const int tail_cap = std::min(tree_plain_tail, remaining);
                        if (tail_cap > 0) {
                            llama_tokens draft_tail = common_speculative_draft_plain(spec, tail_cap);
                            if (draft_tail.size() < (size_t) params_spec.n_min) {
                                common_speculative_discard(spec);
                            } else if (!draft_tail.empty()) {
                                draft_count += (int) draft_tail.size();
                                n_drafted += (int) draft_tail.size();

                                if (tree_result.next_token != draft_tail.front()) {
                                    common_speculative_discard(spec);
                                    ids.push_back(tree_result.next_token);
                                } else {
                                    const int n_past_tail = n_past;
                                    common_batch_clear(batch_tgt);
                                    for (size_t i = 0; i < draft_tail.size(); ++i) {
                                        common_batch_add(batch_tgt, draft_tail[i], n_past + (int) i, { 0 }, true);
                                    }
                                    if (llama_decode(ctx_tgt, batch_tgt) != 0) {
                                        throw std::runtime_error("llama_decode failed");
                                    }

                                    size_t accepted_tail = 1;
                                    common_sampler_accept(tree_smpl.get(), draft_tail[0], true);
                                    llama_token next_tail = tree_result.next_token;
                                    for (size_t i = 1; i < draft_tail.size(); ++i) {
                                        next_tail = common_sampler_sample(tree_smpl.get(), ctx_tgt, (int32_t) (i - 1));
                                        if (next_tail != draft_tail[i]) {
                                            break;
                                        }
                                        accepted_tail += 1;
                                        common_sampler_accept(tree_smpl.get(), draft_tail[i], true);
                                    }
                                    if (accepted_tail == draft_tail.size()) {
                                        next_tail = common_sampler_sample(tree_smpl.get(), ctx_tgt, (int32_t) (draft_tail.size() - 1));
                                    }

                                    n_accept += (int) accepted_tail;
                                    n_past = n_past_tail + (int) accepted_tail;
                                    llama_tokens committed_tail(draft_tail.begin(), draft_tail.begin() + (ptrdiff_t) accepted_tail);
                                    common_speculative_accept_committed(spec, committed_tail);
                                    ids.insert(ids.end(), committed_tail.begin(), committed_tail.end());
                                    ids.push_back(next_tail);
                                    llama_memory_seq_rm(mem_tgt, 0, n_past, -1);
                                }
                                used_tail = true;
                            }
                        }
                    }

                    if (!used_tail) {
                        ids.push_back(tree_result.next_token);
                    }
                }

                for (llama_token token : ids) {
                    common_sampler_accept(smpl, token, true);
                }
            }
        } else {
            // optionally, generate draft tokens that can be appended to the target batch
            //
            // this is the most important part of the speculation. the more probable tokens that are provided here
            // the better the performance will be. in theory, this computation can be performed asynchronously and even
            // offloaded to a remote device. it doesn't even have to be based on an LLM. instead, it can provide tokens
            // from a cache or lookup tables.
            //
            llama_tokens draft = common_speculative_draft(spec, params_spec, prompt_tgt, id_last);

            // always have a token to evaluate from before - id_last
            common_batch_clear(batch_tgt);
            common_batch_add  (batch_tgt, id_last, n_past++, { 0 }, true);

            // evaluate the target model on [id_last, draft0, draft1, ..., draftN-1]
            {
                // do not waste time on small drafts
                if (draft.size() < (size_t) params_spec.n_min) {
                    common_speculative_discard(spec);
                    draft.clear();
                }

                for (size_t i = 0; i < draft.size(); ++i) {
                    common_batch_add(batch_tgt, draft[i], n_past + i, { 0 }, true);
                }

                llama_decode(ctx_tgt, batch_tgt);
            }

            // sample from the full target batch and return the accepted tokens based on the target sampler
            const auto drafted_ids = common_sampler_sample_and_accept_n(smpl, ctx_tgt, draft);

            GGML_ASSERT(drafted_ids.size() > 0); // there will always be at least one accepted token

            n_past    += drafted_ids.size() - 1;
            draft_count = (int) draft.size();
            n_drafted += draft.size(); // note: we ignore the discarded small drafts
            n_accept  += drafted_ids.size() - 1;
            common_speculative_accept_tokens(spec, drafted_ids, drafted_ids.size() - 1);
            ids = drafted_ids;
        }

        GGML_ASSERT(ids.size() > 0);

        if (params.n_predict >= 0) {
            const int remaining = params.n_predict - n_predict;
            if (remaining <= 0) {
                break;
            }
            if ((int) ids.size() > remaining) {
                ids.resize((size_t) remaining);
            }
        }

        n_predict += ids.size();

        // process the accepted tokens and update contexts
        //
        // this is the standard token post-processing that we normally do
        // in this case, we do it for a group of accepted tokens at once
        //
        for (size_t i = 0; i < ids.size(); ++i) {
            prompt_tgt.push_back(id_last);

            id_last = ids[i];

            if (llama_vocab_is_eog(vocab, id_last)) {
                has_eos = true;
                break;
            }

            const std::string token_str = common_token_to_piece(ctx_tgt, id_last);

            if (params.use_color && i + 1 < ids.size()) {
                LOG("\u001b[%dm%s\u001b[37m", (36 - 0 % 6), token_str.c_str());
            } else {
                LOG("%s", token_str.c_str());
            }
        }

        LOG_DBG("accepted %d/%d draft tokens, the last target token is: (%d)\n", (int) ids.size() - 1, draft_count, id_last);

        {
            LOG_DBG("clear kv cache from any extra tokens, n_past = %d\n", n_past);

            llama_memory_seq_rm(llama_get_memory(ctx_tgt), 0, n_past, -1);
        }

        if ((params.n_predict >= 0 && n_predict > params.n_predict) || has_eos) {
            break;
        }
    }

    auto t_dec_end = ggml_time_us();

    const int n_input = inp.size();

    LOG("\n\n");

    LOG_INF("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_input,   (t_enc_end - t_enc_start) / 1e6f, inp.size() / ((t_enc_end - t_enc_start) / 1e6f));
    LOG_INF("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_predict, (t_dec_end - t_dec_start) / 1e6f, n_predict  / ((t_dec_end - t_dec_start) / 1e6f));

    LOG_INF("\n");
    LOG_INF("n_draft   = %d\n", params_spec.n_max);
    LOG_INF("n_predict = %d\n", n_predict);
    LOG_INF("n_drafted = %d\n", n_drafted);
    LOG_INF("n_accept  = %d\n", n_accept);
    LOG_INF("accept    = %.3f%%\n", 100.0f * n_accept / n_drafted);

    LOG_INF("\n");
    LOG_INF("draft:\n\n");

    LOG_INF("\n");
    LOG_INF("target:\n\n");
    common_perf_print(ctx_tgt, smpl);

    llama_batch_free(batch_tgt);

    common_sampler_free(smpl);
    common_speculative_free(spec);

    llama_backend_free();

    LOG("\n\n");

    return 0;
}
