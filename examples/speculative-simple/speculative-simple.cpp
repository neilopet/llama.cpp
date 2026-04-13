#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <cctype>
#include <clocale>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <optional>
#include <string>
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

    llama_batch batch_tgt = llama_batch_init(llama_n_batch(ctx_tgt), 0, 1);

    const bool use_tree = env_flag("LLAMA_GEMMA4_DRAFT_TREE");
    const int tree_width = std::max(1, env_int("LLAMA_GEMMA4_TREE_WIDTH", 2));
    const int tree_depth = std::max(0, env_int("LLAMA_GEMMA4_TREE_DEPTH", 2));
    const int tree_plain_tail = std::max(0, env_int("LLAMA_GEMMA4_TREE_PLAIN_TAIL", 8));

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

                bool stop_tree = false;
                for (int depth = 0; depth < tree_depth; ++depth) {
                    const auto level = common_speculative_preview_tree_level(spec, tree_width);
                    if (level.expects_seed) {
                        throw std::runtime_error("tree speculative backend returned a seed level after seed commit");
                    }
                    if (level.candidates.empty()) {
                        break;
                    }

                    draft_count += (int) level.candidates.size();
                    n_drafted += (int) level.candidates.size();

                    const llama_token predicted = sample_and_accept_current(smpl, ctx_tgt);
                    const auto match = std::find_if(
                            level.candidates.begin(),
                            level.candidates.end(),
                            [predicted](const common_speculative_tree_candidate & candidate) {
                                return candidate.token == predicted;
                            });

                    if (match == level.candidates.end()) {
                        common_speculative_discard(spec);
                        common_speculative_accept_committed(spec, { predicted });
                        ids.push_back(predicted);
                        decode_target({ predicted }, false);
                        stop_tree = true;
                        break;
                    }

                    common_speculative_commit_tree_token(spec, predicted);
                    ids.push_back(predicted);
                    n_accept += 1;
                    decode_target({ predicted }, false);
                }

                bool used_tail = false;
                if (!stop_tree && tree_plain_tail > 0) {
                    const int remaining = std::max(0, params_spec.n_max - (int) ids.size());
                    const int tail_cap = std::min(tree_plain_tail, remaining);
                    if (tail_cap > 0) {
                        llama_tokens draft_tail = common_speculative_draft_plain(spec, tail_cap);
                        if (draft_tail.size() < (size_t) params_spec.n_min) {
                            common_speculative_discard(spec);
                        } else if (!draft_tail.empty()) {
                            draft_count += (int) draft_tail.size();
                            n_drafted += (int) draft_tail.size();
                            const llama_token first_tail = sample_and_accept_current(smpl, ctx_tgt);
                            if (first_tail != draft_tail.front()) {
                                common_speculative_discard(spec);
                                ids.push_back(first_tail);
                            } else {
                                const int n_past_tail = n_past;
                                common_batch_clear(batch_tgt);
                                for (size_t i = 0; i < draft_tail.size(); ++i) {
                                    common_batch_add(batch_tgt, draft_tail[i], n_past + (int) i, { 0 }, true);
                                }
                                if (llama_decode(ctx_tgt, batch_tgt) != 0) {
                                    throw std::runtime_error("llama_decode failed");
                                }
                                n_past += (int) draft_tail.size();

                                ids.push_back(first_tail);

                                if (draft_tail.size() == 1) {
                                    n_accept += 1;
                                    common_speculative_accept_committed(spec, { first_tail });
                                    n_past = n_past_tail + 1;
                                    ids.push_back(sample_and_accept_current(smpl, ctx_tgt));
                                } else {
                                    llama_tokens draft_suffix(draft_tail.begin() + 1, draft_tail.end());
                                    std::vector<int> idxs(draft_tail.size());
                                    for (size_t i = 0; i < idxs.size(); ++i) {
                                        idxs[i] = (int) i;
                                    }

                                    const auto ids_suffix = common_sampler_sample_and_accept_n(smpl, ctx_tgt, idxs, draft_suffix);
                                    size_t accepted_suffix = 0;
                                    while (accepted_suffix < draft_suffix.size() &&
                                           accepted_suffix < ids_suffix.size() &&
                                           ids_suffix[accepted_suffix] == draft_suffix[accepted_suffix]) {
                                        accepted_suffix++;
                                    }

                                    const size_t accepted_tail = 1 + accepted_suffix;
                                    n_accept += (int) accepted_tail;
                                    n_past = n_past_tail + (int) accepted_tail;
                                    llama_tokens committed_tail(draft_tail.begin(), draft_tail.begin() + (ptrdiff_t) accepted_tail);
                                    common_speculative_accept_committed(spec, committed_tail);
                                    ids.insert(ids.end(), ids_suffix.begin(), ids_suffix.end());
                                }
                            }
                            used_tail = true;
                        }
                    }
                }

                if (!used_tail) {
                    const llama_token next = sample_and_accept_current(smpl, ctx_tgt);
                    ids.push_back(next);
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
