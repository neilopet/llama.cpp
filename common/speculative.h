#pragma once

#include "llama.h"
#include "common.h"

struct common_speculative;
struct common_speculative_tree_candidate {
    llama_token token = 0;
    float score = 0.0f;
    float mtp_logprob = 0.0f;
    float base_logprob = 0.0f;
};

struct common_speculative_tree_level {
    bool expects_seed = false;
    uint32_t depth = 0;
    std::vector<common_speculative_tree_candidate> candidates;
};

// comma separated list of all types
std::string common_speculative_type_name_str();

// convert string to type
enum common_speculative_type common_speculative_type_from_name(const std::string & name);

// convert type to string
std::string common_speculative_type_to_str(enum common_speculative_type type);

// check if the llama_context is compatible for speculative decoding
// note: clears the memory of the context
bool common_speculative_is_compat(llama_context * ctx_tgt);

common_speculative * common_speculative_init(
        common_params_speculative & params,
        llama_context             * ctx_tgt);

void common_speculative_free(common_speculative * spec);

// optionally call once at the beginning of a new generation
void common_speculative_begin(common_speculative * spec, const llama_tokens & prompt);

// sample up to n_draft tokens and add them to the batch using the draft model
llama_tokens common_speculative_draft(
                     common_speculative * spec,
        const common_params_speculative & params,
                     const llama_tokens & prompt,
                            llama_token   id_last);

// informs the speculative decoder that n_accepted tokens were accepted by the target model
void common_speculative_accept(common_speculative * spec, uint16_t n_accepted);
void common_speculative_accept_tokens(common_speculative * spec, const llama_tokens & accepted_tokens, uint16_t n_accepted);
void common_speculative_accept_committed(common_speculative * spec, const llama_tokens & committed_tokens);
void common_speculative_discard(common_speculative * spec);
bool common_speculative_supports_tree(common_speculative * spec);
common_speculative_tree_level common_speculative_preview_tree_level(common_speculative * spec, int max_width);
void common_speculative_commit_tree_token(common_speculative * spec, llama_token token);
llama_tokens common_speculative_draft_plain(common_speculative * spec, int max_tokens);

// print statistics about the speculative decoding
void common_speculative_print_stats(const common_speculative * spec);
