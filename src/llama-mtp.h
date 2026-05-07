#pragma once

#include "llama.h"

#include <vector>

struct llama_mtp {
    llama_context * ctx_mtp    = nullptr; // non-owning
    llama_batch     hook_batch = {};      // sized to n_ubatch; embeddings are packed [e_input ; h_input]
    llama_token   * hook_token = nullptr; // owned token storage; hook_batch.token is nulled for embedding-only batches

    // Cross-ubatch shift state: pair (h_p, x_{p+1}) at MTP pos p+1. The last
    // h-row of one ubatch needs the first token/embedding of the NEXT ubatch
    // to pair with, so it's stashed here until that next ubatch fires. Text
    // batches validate continuity with the scalar position. Media/M-RoPE
    // batches can keep a constant temporal position across many rows, so they
    // rely on explicit sequence removal/task reset to clear stale pending
    // state.
    std::vector<float> pending_h;
    llama_pos          pending_pos = -1;
};
