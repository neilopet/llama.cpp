#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include "llama-context.h"
#include "llama-model.h"
#include "llama-memory.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <optional>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;
namespace fs = std::filesystem;

static constexpr float MTP_EPS = 1e-6f;
static constexpr int   MTP_DIM_IN = 3072;
static constexpr int   MTP_DIM_HID = 256;
static constexpr int   MTP_DIM_OUT_HIDDEN = 1536;
static constexpr int   MTP_VOCAB = 262144;

struct blob_entry {
    int index = -1;
    std::optional<std::string> file;
    std::optional<std::string> scales_file;
    std::vector<int64_t> shape;
    std::string tensor_type;
};

static std::vector<uint8_t> read_file_bytes(const fs::path & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open " + path.string());
    }
    in.seekg(0, std::ios::end);
    std::vector<uint8_t> out((size_t) in.tellg());
    in.seekg(0, std::ios::beg);
    if (!out.empty()) {
        in.read(reinterpret_cast<char *>(out.data()), out.size());
    }
    return out;
}

static std::vector<float> read_f32_file(const fs::path & path) {
    auto bytes = read_file_bytes(path);
    if (bytes.size() % sizeof(float) != 0) {
        throw std::runtime_error("invalid f32 blob size for " + path.string());
    }
    std::vector<float> out(bytes.size() / sizeof(float));
    std::memcpy(out.data(), bytes.data(), bytes.size());
    return out;
}

static float read_scalar_f32(const fs::path & path) {
    auto v = read_f32_file(path);
    if (v.size() != 1) {
        throw std::runtime_error("expected scalar f32 in " + path.string());
    }
    return v[0];
}

static int32_t read_scalar_i32(const fs::path & path) {
    auto bytes = read_file_bytes(path);
    if (bytes.size() != sizeof(int32_t)) {
        throw std::runtime_error("expected scalar i32 in " + path.string());
    }
    int32_t out;
    std::memcpy(&out, bytes.data(), sizeof(out));
    return out;
}

static int8_t decode_signed_int2(uint8_t v) {
    switch (v & 0x3u) {
        case 0: return 0;
        case 1: return 1;
        case 2: return -2;
        default: return -1;
    }
}

static int8_t decode_signed_int4(uint8_t v) {
    v &= 0x0fu;
    return v >= 8 ? (int8_t) v - 16 : (int8_t) v;
}

static int8_t quantize_value_to_i8(float value, float inv_scale) {
    const float q = std::round(value * inv_scale);
    return (int8_t) std::max(-128.0f, std::min(127.0f, q));
}

static float gelu_scalar(float x) {
    constexpr float SQRT_2_OVER_PI = 0.7978846f;
    const float x3 = x*x*x;
    return 0.5f*x*(1.0f + std::tanh(SQRT_2_OVER_PI*(x + 0.044715f*x3)));
}

static int argmax_logits(const float * logits, int n_vocab) {
    int best = 0;
    float best_v = logits[0];
    for (int i = 1; i < n_vocab; ++i) {
        if (logits[i] > best_v) {
            best_v = logits[i];
            best = i;
        }
    }
    return best;
}

static llama_token argmax_logits_skip_eog(const llama_vocab * vocab, const float * logits, int n_vocab) {
    int best = -1;
    float best_v = -INFINITY;
    for (int i = 0; i < n_vocab; ++i) {
        if (llama_vocab_is_eog(vocab, i)) {
            continue;
        }
        if (best < 0 || logits[i] > best_v) {
            best = i;
            best_v = logits[i];
        }
    }
    if (best < 0) {
        return (llama_token) argmax_logits(logits, n_vocab);
    }
    return (llama_token) best;
}

struct packed_embedding_table {
    int rows = 0;
    int cols = 0;
    int bits = 0;
    std::vector<float> scales;
    std::vector<uint8_t> packed;

    std::vector<float> dequantize_row(int row) const {
        if (row < 0 || row >= rows) {
            throw std::runtime_error("embedding row out of range");
        }
        std::vector<float> out(cols);
        const float scale = scales.at(row);
        if (bits == 2) {
            const int row_bytes = cols / 4;
            const uint8_t * src = packed.data() + row*row_bytes;
            for (int i = 0; i < row_bytes; ++i) {
                const uint8_t b = src[i];
                const int base = 4*i;
                out[base + 0] = decode_signed_int2((b >> 0) & 0x3u) * scale;
                out[base + 1] = decode_signed_int2((b >> 2) & 0x3u) * scale;
                out[base + 2] = decode_signed_int2((b >> 4) & 0x3u) * scale;
                out[base + 3] = decode_signed_int2((b >> 6) & 0x3u) * scale;
            }
        } else if (bits == 4) {
            const int row_bytes = cols / 2;
            const uint8_t * src = packed.data() + row*row_bytes;
            for (int i = 0; i < row_bytes; ++i) {
                const uint8_t b = src[i];
                const int base = 2*i;
                out[base + 0] = decode_signed_int4((b >> 0) & 0x0fu) * scale;
                out[base + 1] = decode_signed_int4((b >> 4) & 0x0fu) * scale;
            }
        } else {
            throw std::runtime_error("unsupported packed embedding bits");
        }
        return out;
    }
};

struct quantized_matrix_i8 {
    int rows = 0;
    int cols = 0;
    std::vector<float> scales;
    std::vector<int8_t> data;
};

struct quantized_matrix_i4 {
    int rows = 0;
    int cols = 0;
    std::vector<float> scales;
    std::vector<uint8_t> packed;

    inline int8_t qvalue(int row, int col) const {
        const int row_bytes = cols / 2;
        const uint8_t byte = packed[row*row_bytes + col/2];
        return decode_signed_int4((col % 2 == 0) ? (byte & 0x0f) : ((byte >> 4) & 0x0f));
    }
};

struct quantized_linear_i8 {
    float input_scale = 0.0f;
    float output_scale = 0.0f;
    quantized_matrix_i8 weight;

    std::vector<float> run(const std::vector<float> & input) const {
        if ((int) input.size() != weight.cols) {
            throw std::runtime_error("QuantizedLinearI8 input mismatch");
        }
        const float inv_in = 1.0f / input_scale;
        std::vector<int8_t> q_input(weight.cols);
        for (int i = 0; i < weight.cols; ++i) {
            q_input[i] = quantize_value_to_i8(input[i], inv_in);
        }
        std::vector<float> out(weight.rows);
        for (int r = 0; r < weight.rows; ++r) {
            const int8_t * row = weight.data.data() + r*weight.cols;
            int32_t acc = 0;
            for (int c = 0; c < weight.cols; ++c) {
                acc += (int32_t) q_input[c] * (int32_t) row[c];
            }
            const float float_acc = (float) acc * input_scale * weight.scales[r];
            const int8_t q_out = quantize_value_to_i8(float_acc, 1.0f / output_scale);
            out[r] = (float) q_out * output_scale;
        }
        return out;
    }
};

struct quantized_linear_i4 {
    float input_scale = 0.0f;
    float output_scale = 0.0f;
    quantized_matrix_i4 weight;

    std::vector<float> run(const std::vector<float> & input) const {
        if ((int) input.size() != weight.cols) {
            throw std::runtime_error("QuantizedLinearI4 input mismatch");
        }
        const float inv_in = 1.0f / input_scale;
        std::vector<int8_t> q_input(weight.cols);
        for (int i = 0; i < weight.cols; ++i) {
            q_input[i] = quantize_value_to_i8(input[i], inv_in);
        }
        std::vector<float> out(weight.rows);
        for (int r = 0; r < weight.rows; ++r) {
            int32_t acc = 0;
            for (int c = 0; c < weight.cols; ++c) {
                acc += (int32_t) q_input[c] * (int32_t) weight.qvalue(r, c);
            }
            const float float_acc = (float) acc * input_scale * weight.scales[r];
            const int8_t q_out = quantize_value_to_i8(float_acc, 1.0f / output_scale);
            out[r] = (float) q_out * output_scale;
        }
        return out;
    }
};

struct float_linear_i4 {
    quantized_matrix_i4 weight;

    std::vector<float> run(const std::vector<float> & input) const {
        if ((int) input.size() != weight.cols) {
            throw std::runtime_error("FloatLinearI4 input mismatch");
        }
        std::vector<float> out(weight.rows);
        for (int r = 0; r < weight.rows; ++r) {
            float acc = 0.0f;
            const float scale = weight.scales[r];
            for (int c = 0; c < weight.cols; ++c) {
                acc += input[c] * (float) weight.qvalue(r, c) * scale;
            }
            out[r] = acc;
        }
        return out;
    }

    int argmax_softcapped(const std::vector<float> & input, float mul_in, float mul_out) const {
        if ((int) input.size() != weight.cols) {
            throw std::runtime_error("FloatLinearI4 input mismatch");
        }
        int best = 0;
        float best_v = -INFINITY;
        for (int r = 0; r < weight.rows; ++r) {
            float acc = 0.0f;
            const float scale = weight.scales[r];
            for (int c = 0; c < weight.cols; ++c) {
                acc += input[c] * (float) weight.qvalue(r, c) * scale;
            }
            acc = std::tanh(acc * mul_in) * mul_out;
            if (acc > best_v) {
                best_v = acc;
                best = r;
            }
        }
        return best;
    }
};

static std::vector<float> rms_norm_last_dim(const std::vector<float> & input, const std::vector<float> & weight, float eps) {
    const int dim = (int) weight.size();
    if (dim <= 0 || input.size() % weight.size() != 0) {
        throw std::runtime_error("rms_norm_last_dim shape mismatch");
    }
    std::vector<float> out(input.size());
    for (size_t off = 0; off < input.size(); off += dim) {
        float mean_sq = 0.0f;
        for (int i = 0; i < dim; ++i) {
            mean_sq += input[off + i] * input[off + i];
        }
        mean_sq /= dim;
        const float inv_rms = 1.0f / std::sqrt(mean_sq + eps);
        for (int i = 0; i < dim; ++i) {
            out[off + i] = input[off + i] * inv_rms * weight[i];
        }
    }
    return out;
}

static std::vector<float> add_vec(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("add_vec size mismatch");
    }
    std::vector<float> out(a.size());
    for (size_t i = 0; i < a.size(); ++i) out[i] = a[i] + b[i];
    return out;
}

static std::vector<float> mul_vec(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("mul_vec size mismatch");
    }
    std::vector<float> out(a.size());
    for (size_t i = 0; i < a.size(); ++i) out[i] = a[i] * b[i];
    return out;
}

static std::vector<float> gelu_vec(const std::vector<float> & a) {
    std::vector<float> out(a.size());
    for (size_t i = 0; i < a.size(); ++i) out[i] = gelu_scalar(a[i]);
    return out;
}

static std::vector<float> apply_rope(const std::vector<float> & input, int heads, int head_dim, const std::vector<float> & rope_div, int input_pos) {
    const int half = head_dim / 2;
    if ((int) input.size() != heads*head_dim || (int) rope_div.size() != half) {
        throw std::runtime_error("apply_rope shape mismatch");
    }
    std::vector<float> out(input.size());
    for (int h = 0; h < heads; ++h) {
        const float * src = input.data() + h*head_dim;
        float * dst = out.data() + h*head_dim;
        for (int i = 0; i < half; ++i) {
            const float angle = rope_div[i] * (float) input_pos;
            const float c = std::cos(angle);
            const float s = std::sin(angle);
            const float a = src[i];
            const float b = src[half + i];
            dst[i] = a*c - b*s;
            dst[half + i] = b*c + a*s;
        }
    }
    return out;
}

static std::vector<float> runtime_bmm_qk_seq_major(const std::vector<float> & query, int query_rows, int dim, const std::vector<int8_t> & cache_seq_major, int active_len, float scale) {
    if ((int) query.size() != query_rows*dim || (int) cache_seq_major.size() < active_len*dim || ((int) cache_seq_major.size() % dim) != 0) {
        throw std::runtime_error(
            "runtime_bmm_qk_seq_major shape mismatch: query=" + std::to_string(query.size()) +
            " expected=" + std::to_string(query_rows*dim) +
            " cache=" + std::to_string(cache_seq_major.size()) +
            " min=" + std::to_string(active_len*dim));
    }
    std::vector<float> out(query_rows*active_len);
    for (int r = 0; r < query_rows; ++r) {
        const float * q = query.data() + r*dim;
        float * dst = out.data() + r*active_len;
        for (int s = 0; s < active_len; ++s) {
            const int8_t * k = cache_seq_major.data() + s*dim;
            float acc = 0.0f;
            for (int d = 0; d < dim; ++d) {
                acc += q[d] * ((float) k[d] * scale);
            }
            dst[s] = acc;
        }
    }
    return out;
}

static std::vector<float> masked_softmax_rows_prefix(const std::vector<float> & scores, int rows, int cols) {
    if ((int) scores.size() != rows*cols) {
        throw std::runtime_error("softmax shape mismatch");
    }
    std::vector<float> out(scores.size());
    for (int r = 0; r < rows; ++r) {
        const float * src = scores.data() + r*cols;
        float * dst = out.data() + r*cols;
        float max_v = -INFINITY;
        for (int c = 0; c < cols; ++c) max_v = std::max(max_v, src[c]);
        float sum = 0.0f;
        for (int c = 0; c < cols; ++c) {
            dst[c] = std::exp(src[c] - max_v);
            sum += dst[c];
        }
        if (sum > 0.0f) {
            for (int c = 0; c < cols; ++c) dst[c] /= sum;
        }
    }
    return out;
}

static std::vector<float> runtime_bmm_v_dim_major(const std::vector<float> & probs, int query_rows, int active_len, const std::vector<int8_t> & cache_dim_major, int dim, float scale) {
    if ((int) probs.size() != query_rows*active_len || (int) cache_dim_major.size() < dim*active_len || ((int) cache_dim_major.size() % dim) != 0) {
        throw std::runtime_error(
            "runtime_bmm_v_dim_major shape mismatch: probs=" + std::to_string(probs.size()) +
            " expected=" + std::to_string(query_rows*active_len) +
            " cache=" + std::to_string(cache_dim_major.size()) +
            " min=" + std::to_string(dim*active_len));
    }
    std::vector<float> out(query_rows*dim);
    for (int r = 0; r < query_rows; ++r) {
        const float * p = probs.data() + r*active_len;
        float * dst = out.data() + r*dim;
        for (int d = 0; d < dim; ++d) {
            const int8_t * vcol = cache_dim_major.data() + d*active_len;
            float acc = 0.0f;
            for (int s = 0; s < active_len; ++s) {
                acc += p[s] * ((float) vcol[s] * scale);
            }
            dst[d] = acc;
        }
    }
    return out;
}

struct mtp_layer {
    std::vector<float> pre_attn_norm_weight;
    std::vector<float> query_norm_weight;
    std::vector<float> post_attn_norm_weight;
    std::vector<float> pre_ffw_norm_weight;
    std::vector<float> post_ffw_norm_weight;
    quantized_linear_i8 q_proj;
    quantized_linear_i8 o_proj;
    quantized_linear_i4 gate_proj;
    quantized_linear_i4 up_proj;
    quantized_linear_i4 down_proj;
    std::vector<float> rope_div;
    int kv_group = 13;
    int heads = 4;
    int head_dim = 256;
};

struct mtp_sidecar {
    int input_min_inclusive = 0;
    int input_max_exclusive = 0;
    int fallback_token = 0;
    float embedder_multiplier = 1.0f;
    packed_embedding_table embedder;

    quantized_linear_i8 pre_project;
    std::vector<float> pre_project_norm_weight;
    std::vector<mtp_layer> layers;
    std::vector<float> final_norm_weight;
    float_linear_i4 logits_head;
    float logits_soft_cap_mul_in = 1.0f;
    float logits_soft_cap_mul_out = 1.0f;
    quantized_linear_i8 hidden_head;
    float k13_scale = 0.0f;
    float v13_scale = 0.0f;
    float k14_scale = 0.0f;
    float v14_scale = 0.0f;

    int normalize_token(int token) const {
        if (token >= input_min_inclusive && token < input_max_exclusive) {
            return token;
        }
        return fallback_token;
    }

    std::vector<float> lookup_embedder(int token) const {
        auto out = embedder.dequantize_row(normalize_token(token));
        for (float & v : out) {
            v *= embedder_multiplier;
        }
        return out;
    }
};

static blob_entry find_entry(const json & arr, int index) {
    for (const auto & item : arr) {
        if (item.at("index").get<int>() == index) {
            blob_entry e;
            e.index = index;
            if (!item.at("file").is_null()) {
                e.file = item.at("file").get<std::string>();
            }
            if (!item.at("scales_file").is_null()) {
                e.scales_file = item.at("scales_file").get<std::string>();
            }
            e.shape = item.at("shape").get<std::vector<int64_t>>();
            e.tensor_type = item.at("tensor_type").get<std::string>();
            return e;
        }
    }
    throw std::runtime_error("missing tensor index " + std::to_string(index));
}

static float load_scale_scalar(const fs::path & root, const json & arr, int index) {
    auto e = find_entry(arr, index);
    if (!e.scales_file) {
        throw std::runtime_error("missing scales for tensor " + std::to_string(index));
    }
    auto scales = read_f32_file(root / *e.scales_file);
    if (scales.size() != 1) {
        throw std::runtime_error("expected exactly one scale for tensor " + std::to_string(index));
    }
    return scales[0];
}

static quantized_matrix_i8 load_qmat_i8(const fs::path & root, const json & arr, int index) {
    auto e = find_entry(arr, index);
    if (e.tensor_type != "int8" || e.shape.size() != 2 || !e.scales_file || !e.file) {
        throw std::runtime_error("invalid int8 matrix tensor " + std::to_string(index));
    }
    quantized_matrix_i8 out;
    out.rows = (int) e.shape[0];
    out.cols = (int) e.shape[1];
    out.scales = read_f32_file(root / *e.scales_file);
    auto bytes = read_file_bytes(root / *e.file);
    out.data.assign(reinterpret_cast<const int8_t *>(bytes.data()), reinterpret_cast<const int8_t *>(bytes.data() + bytes.size()));
    return out;
}

static quantized_matrix_i4 load_qmat_i4(const fs::path & root, const json & arr, int index) {
    auto e = find_entry(arr, index);
    if (e.tensor_type != "int4" || e.shape.size() != 2 || !e.scales_file || !e.file) {
        throw std::runtime_error("invalid int4 matrix tensor " + std::to_string(index));
    }
    quantized_matrix_i4 out;
    out.rows = (int) e.shape[0];
    out.cols = (int) e.shape[1];
    out.scales = read_f32_file(root / *e.scales_file);
    out.packed = read_file_bytes(root / *e.file);
    return out;
}

static packed_embedding_table load_embedder_table(const fs::path & root, const json & arr, int index, int bits) {
    auto e = find_entry(arr, index);
    if (e.shape.size() != 2 || !e.scales_file || !e.file) {
        throw std::runtime_error("invalid embedder tensor " + std::to_string(index));
    }
    packed_embedding_table out;
    out.rows = (int) e.shape[0];
    out.cols = (int) e.shape[1];
    out.bits = bits;
    out.scales = read_f32_file(root / *e.scales_file);
    out.packed = read_file_bytes(root / *e.file);
    return out;
}

static mtp_sidecar load_mtp_sidecar(const fs::path & dir) {
    const json cfg = json::parse(read_file_bytes(dir / "config.json"));
    if (cfg.at("format").get<std::string>() != "gemma4_e2b_mtp_proto_v1") {
        throw std::runtime_error("unexpected sidecar format");
    }
    const auto & mtp_arr = cfg.at("mtp").at("tensors");
    const auto & emb_arr = cfg.at("embedder").at("tensors");
    mtp_sidecar sc;
    sc.input_min_inclusive = cfg.at("embedder_stats").at("input_min_inclusive").get<int>();
    sc.input_max_exclusive = cfg.at("embedder_stats").at("input_max_exclusive").get<int>();
    sc.fallback_token = cfg.at("embedder_stats").at("fallback_token").get<int>();
    sc.embedder_multiplier = cfg.at("embedder_stats").at("embedder_multiplier").get<float>();
    sc.embedder = load_embedder_table(dir, emb_arr, 11, 2);

    sc.pre_project = { load_scale_scalar(dir, mtp_arr, 50), load_scale_scalar(dir, mtp_arr, 52), load_qmat_i8(dir, mtp_arr, 51) };
    sc.pre_project_norm_weight = read_f32_file(dir / *find_entry(mtp_arr, 49).file);

    auto load_layer = [&](int pre_attn_norm, int query_norm, int post_attn_norm, int pre_ffw_norm, int post_ffw_norm,
                          int q_in, int q_w, int q_out,
                          int o_in, int o_w, int o_out,
                          int gate_in, int gate_w, int gate_out,
                          int up_in, int up_w, int up_out,
                          int down_in, int down_w, int down_out,
                          int rope_div, int kv_group, int heads, int head_dim) {
        mtp_layer l;
        l.pre_attn_norm_weight = read_f32_file(dir / *find_entry(mtp_arr, pre_attn_norm).file);
        l.query_norm_weight = read_f32_file(dir / *find_entry(mtp_arr, query_norm).file);
        l.post_attn_norm_weight = read_f32_file(dir / *find_entry(mtp_arr, post_attn_norm).file);
        l.pre_ffw_norm_weight = read_f32_file(dir / *find_entry(mtp_arr, pre_ffw_norm).file);
        l.post_ffw_norm_weight = read_f32_file(dir / *find_entry(mtp_arr, post_ffw_norm).file);
        l.q_proj = { load_scale_scalar(dir, mtp_arr, q_in), load_scale_scalar(dir, mtp_arr, q_out), load_qmat_i8(dir, mtp_arr, q_w) };
        l.o_proj = { load_scale_scalar(dir, mtp_arr, o_in), load_scale_scalar(dir, mtp_arr, o_out), load_qmat_i8(dir, mtp_arr, o_w) };
        l.gate_proj = { load_scale_scalar(dir, mtp_arr, gate_in), load_scale_scalar(dir, mtp_arr, gate_out), load_qmat_i4(dir, mtp_arr, gate_w) };
        l.up_proj = { load_scale_scalar(dir, mtp_arr, up_in), load_scale_scalar(dir, mtp_arr, up_out), load_qmat_i4(dir, mtp_arr, up_w) };
        l.down_proj = { load_scale_scalar(dir, mtp_arr, down_in), load_scale_scalar(dir, mtp_arr, down_out), load_qmat_i4(dir, mtp_arr, down_w) };
        l.rope_div = read_f32_file(dir / *find_entry(mtp_arr, rope_div).file);
        l.kv_group = kv_group;
        l.heads = heads;
        l.head_dim = head_dim;
        return l;
    };

    sc.layers.push_back(load_layer(49, 47, 31, 30, 29, 55, 56, 57, 93, 94, 95, 100, 101, 102, 100, 104, 105, 109, 110, 111, 45, 13, 4, 256));
    sc.layers.push_back(load_layer(28, 47, 27, 26, 25, 116, 117, 118, 137, 138, 139, 144, 145, 146, 144, 148, 149, 153, 154, 155, 45, 13, 4, 256));
    sc.layers.push_back(load_layer(24, 47, 23, 22, 21, 160, 161, 162, 181, 182, 183, 188, 189, 190, 188, 192, 193, 197, 198, 199, 45, 13, 4, 256));
    sc.layers.push_back(load_layer(20, 18, 13, 12, 11, 204, 205, 206, 235, 236, 237, 242, 243, 244, 242, 246, 247, 251, 252, 253, 17, 14, 4, 512));

    sc.final_norm_weight = read_f32_file(dir / *find_entry(mtp_arr, 10).file);
    sc.logits_head = { load_qmat_i4(dir, mtp_arr, 258) };
    sc.logits_soft_cap_mul_in = read_scalar_f32(dir / *find_entry(mtp_arr, 9).file);
    sc.logits_soft_cap_mul_out = read_scalar_f32(dir / *find_entry(mtp_arr, 8).file);
    sc.hidden_head = { load_scale_scalar(dir, mtp_arr, 263), load_scale_scalar(dir, mtp_arr, 265), load_qmat_i8(dir, mtp_arr, 264) };
    sc.k13_scale = load_scale_scalar(dir, mtp_arr, 4);
    sc.v13_scale = load_scale_scalar(dir, mtp_arr, 6);
    sc.k14_scale = load_scale_scalar(dir, mtp_arr, 3);
    sc.v14_scale = load_scale_scalar(dir, mtp_arr, 7);
    return sc;
}

struct exported_kv {
    std::vector<int8_t> k13;
    std::vector<int8_t> v13;
    std::vector<int8_t> k14;
    std::vector<int8_t> v14;
};

struct kv_export_binding {
    llama_memory_context_ptr holder;
    const llama_kv_cache_context * base = nullptr;
    const llama_kv_cache_context * swa  = nullptr;
};

static kv_export_binding bind_kv_context(llama_context * ctx) {
    auto * mem = reinterpret_cast<llama_memory_i *>(llama_get_memory(ctx));
    if (!mem) {
        throw std::runtime_error("context has no memory");
    }
    kv_export_binding out;
    out.holder = mem->init_full();
    if (auto * base = dynamic_cast<llama_kv_cache_context *>(out.holder.get())) {
        out.base = base;
        return out;
    }
    if (auto * iswa = dynamic_cast<llama_kv_cache_iswa_context *>(out.holder.get())) {
        out.base = iswa->get_base();
        out.swa  = iswa->get_swa();
        return out;
    }
    if (auto * hy = dynamic_cast<llama_memory_hybrid_context *>(out.holder.get())) {
        out.base = hy->get_attn();
        return out;
    }
    if (auto * hyi = dynamic_cast<llama_memory_hybrid_iswa_context *>(out.holder.get())) {
        const auto * attn = hyi->get_attn();
        if (attn) {
            out.base = attn->get_base();
            out.swa  = attn->get_swa();
            if (out.base || out.swa) {
                return out;
            }
        }
    }
    throw std::runtime_error("unsupported memory context type for KV export");
}

static float tensor_value_f32(const ggml_tensor * t, const std::vector<uint8_t> & buf, int64_t i0, int64_t i1 = 0, int64_t i2 = 0, int64_t i3 = 0) {
    const size_t off = i0*t->nb[0] + i1*t->nb[1] + i2*t->nb[2] + i3*t->nb[3];
    if (t->type == GGML_TYPE_F16) {
        ggml_fp16_t v;
        std::memcpy(&v, buf.data() + off, sizeof(v));
        return ggml_fp16_to_fp32(v);
    }
    if (t->type == GGML_TYPE_F32) {
        float v;
        std::memcpy(&v, buf.data() + off, sizeof(v));
        return v;
    }
    throw std::runtime_error("unsupported tensor type for KV export");
}

static void export_layer_kv(const llama_kv_cache_context * attn, int layer_index, int filled_len, int total_len, float k_scale, float v_scale, std::vector<int8_t> & k_out, std::vector<int8_t> & v_out) {
    ggml_init_params params = {
        /* mem_size   = */ size_t(16u * ggml_tensor_overhead()),
        /* mem_buffer = */ nullptr,
        /* no_alloc   = */ true,
    };
    ggml_context_ptr gctx { ggml_init(params) };
    if (!gctx) {
        throw std::runtime_error("failed to init ggml context for KV export");
    }

    ggml_tensor * k = attn->get_k(gctx.get(), layer_index);
    ggml_tensor * v = attn->get_v(gctx.get(), layer_index);
    if (!k || !v) {
        throw std::runtime_error("failed to build KV views");
    }

    const int head_dim = (int) k->ne[0];
    const int n_head = (int) k->ne[1];
    const int dim = head_dim * n_head;

    std::vector<uint8_t> k_buf(ggml_nbytes(k));
    std::vector<uint8_t> v_buf(ggml_nbytes(v));
    ggml_backend_tensor_get(k, k_buf.data(), 0, k_buf.size());
    ggml_backend_tensor_get(v, v_buf.data(), 0, v_buf.size());

    k_out.assign(total_len * dim, 0);
    v_out.assign(dim * total_len, 0);
    const float inv_k = 1.0f / k_scale;
    const float inv_v = 1.0f / v_scale;

    for (int seq = 0; seq < filled_len; ++seq) {
        for (int h = 0; h < n_head; ++h) {
            for (int d = 0; d < head_dim; ++d) {
                const int flat = h*head_dim + d;
                const float kf = tensor_value_f32(k, k_buf, d, h, seq, 0);
                k_out[seq*dim + flat] = quantize_value_to_i8(kf, inv_k);
            }
        }
    }

    const bool v_trans = v->ne[0] >= filled_len && v->ne[2] == head_dim;
    if (v_trans) {
        for (int h = 0; h < n_head; ++h) {
            for (int d = 0; d < head_dim; ++d) {
                const int flat = h*head_dim + d;
                for (int seq = 0; seq < filled_len; ++seq) {
                    const float vf = tensor_value_f32(v, v_buf, seq, h, d, 0);
                    v_out[flat*total_len + seq] = quantize_value_to_i8(vf, inv_v);
                }
            }
        }
    } else {
        for (int h = 0; h < n_head; ++h) {
            for (int d = 0; d < head_dim; ++d) {
                const int flat = h*head_dim + d;
                for (int seq = 0; seq < filled_len; ++seq) {
                    const float vf = tensor_value_f32(v, v_buf, d, h, seq, 0);
                    v_out[flat*total_len + seq] = quantize_value_to_i8(vf, inv_v);
                }
            }
        }
    }
}

static exported_kv export_current_kv(llama_context * ctx_dft, const mtp_sidecar & sc, int filled_len, int total_len) {
    auto binding = bind_kv_context(ctx_dft);
    const auto * model = llama_get_model(ctx_dft);
    const auto & hparams = model->hparams;
    constexpr int layer_k13 = 13;
    constexpr int layer_k14 = 14;
    if ((int) hparams.n_layer <= layer_k14) {
        throw std::runtime_error("draft model has too few layers for Gemma4 MTP taps");
    }

    const bool k13_is_swa = hparams.is_swa(layer_k13);
    const bool k14_is_swa = hparams.is_swa(layer_k14);
    const llama_kv_cache_context * ctx13 = k13_is_swa ? binding.swa : binding.base;
    const llama_kv_cache_context * ctx14 = k14_is_swa ? binding.swa : binding.base;
    if (!ctx13 || !ctx14) {
        throw std::runtime_error("required base/swa KV context missing for Gemma4 MTP export");
    }

    exported_kv out;
    export_layer_kv(ctx13, layer_k13, filled_len, total_len, sc.k13_scale, sc.v13_scale, out.k13, out.v13);
    export_layer_kv(ctx14, layer_k14, filled_len, total_len, sc.k14_scale, sc.v14_scale, out.k14, out.v14);
    return out;
}

struct mtp_step_output {
    int token = -1;
    std::vector<float> hidden;
};

struct mtp_timing {
    double kv_export_s = 0.0;
    double mtp_step_s = 0.0;
    int mtp_steps = 0;
};

struct verify_result {
    std::vector<llama_token> accepted;
    int proposed = 0;
    int accepted_from_draft = 0;
    bool full_match = false;
    bool first_token_mismatch = false;
};

struct proto_metrics {
    bool draft_only = false;
    bool use_mtp = false;
    int chunks = 0;
    int proposed_tokens = 0;
    int accepted_tokens = 0;
    int accepted_from_draft = 0;
    int verifier_substitutions = 0;
    int full_match_chunks = 0;
    int first_token_mismatches = 0;
    int emitted_tokens = 0;
    double draft_build_s = 0.0;
    double verify_s = 0.0;
    double draft_sync_s = 0.0;
    double kv_export_s = 0.0;
    double mtp_step_s = 0.0;
    int mtp_steps = 0;
};

struct mtp_layer_trace {
    std::vector<float> hidden_in;
    std::vector<float> q_rope;
    std::vector<float> attn;
    std::vector<float> post_norm;
    std::vector<float> hidden_after_attn;
    std::vector<float> ff_out_norm;
    std::vector<float> hidden_after_ff;
};

struct mtp_trace_output {
    int token = -1;
    std::vector<float> hidden;
    std::vector<float> logits;
    std::vector<float> final_hidden;
    std::vector<mtp_layer_trace> layers;
};

struct oracle_fixture {
    struct chain_step {
        int step_index = 0;
        int input_pos = 0;
        int active_len = 0;
        int input_token = -1;
        int expected_top_token = -1;
        std::vector<float> activations;
        std::vector<float> expected_logits;
        std::vector<float> expected_hidden;
        std::vector<float> expected_final_hidden;
    };

    int draft_input_pos = 0;
    int active_len = 0;
    int max_seq_len = 0;
    int good_token = -1;
    std::vector<float> activations;
    std::vector<uint8_t> mask;
    std::vector<int8_t> k13;
    std::vector<int8_t> k14;
    std::vector<int8_t> v13;
    std::vector<int8_t> v14;
    mtp_trace_output expected;
    std::vector<chain_step> chain;
};

static std::vector<uint32_t> read_u32_file(const fs::path & path) {
    auto bytes = read_file_bytes(path);
    if (bytes.size() % sizeof(uint32_t) != 0) {
        throw std::runtime_error("invalid u32 blob size for " + path.string());
    }
    std::vector<uint32_t> out(bytes.size() / sizeof(uint32_t));
    std::memcpy(out.data(), bytes.data(), bytes.size());
    return out;
}

static std::vector<int8_t> read_i8_file(const fs::path & path) {
    auto bytes = read_file_bytes(path);
    return std::vector<int8_t>(reinterpret_cast<const int8_t *>(bytes.data()),
            reinterpret_cast<const int8_t *>(bytes.data() + bytes.size()));
}

static std::vector<uint8_t> read_u8_file(const fs::path & path) {
    return read_file_bytes(path);
}

static oracle_fixture load_mtp_oracle_fixture(const fs::path & dir) {
    const json cfg = json::parse(read_file_bytes(dir / "config.json"));
    if (cfg.at("format").get<std::string>() != "gemma4_e2b_mtp_oracle_fixture_v1") {
        throw std::runtime_error("unexpected oracle fixture format");
    }
    oracle_fixture fx;
    fx.draft_input_pos = cfg.at("draft_input_pos").get<int>();
    fx.active_len = cfg.at("active_len").get<int>();
    fx.max_seq_len = cfg.at("max_seq_len").get<int>();
    fx.good_token = cfg.at("good_token").get<int>();
    fx.activations = read_f32_file(dir / cfg.at("inputs").at("activations").at("file").get<std::string>());
    fx.mask = read_u8_file(dir / cfg.at("inputs").at("mask").at("file").get<std::string>());
    fx.k13 = read_i8_file(dir / cfg.at("inputs").at("k13").at("file").get<std::string>());
    fx.k14 = read_i8_file(dir / cfg.at("inputs").at("k14").at("file").get<std::string>());
    fx.v13 = read_i8_file(dir / cfg.at("inputs").at("v13").at("file").get<std::string>());
    fx.v14 = read_i8_file(dir / cfg.at("inputs").at("v14").at("file").get<std::string>());
    fx.expected.logits = read_f32_file(dir / cfg.at("expected").at("logits").at("file").get<std::string>());
    fx.expected.hidden = read_f32_file(dir / cfg.at("expected").at("hidden").at("file").get<std::string>());
    fx.expected.final_hidden = read_f32_file(dir / cfg.at("expected").at("final_hidden").at("file").get<std::string>());
    fx.expected.token = cfg.at("expected").at("top_token").get<int>();
    for (const auto & layer : cfg.at("trace").at("layers")) {
        mtp_layer_trace out;
        out.hidden_in = read_f32_file(dir / layer.at("hidden_in").at("file").get<std::string>());
        out.q_rope = read_f32_file(dir / layer.at("q_rope").at("file").get<std::string>());
        out.attn = read_f32_file(dir / layer.at("attn").at("file").get<std::string>());
        out.post_norm = read_f32_file(dir / layer.at("post_norm").at("file").get<std::string>());
        out.hidden_after_attn = read_f32_file(dir / layer.at("hidden_after_attn").at("file").get<std::string>());
        out.ff_out_norm = read_f32_file(dir / layer.at("ff_out_norm").at("file").get<std::string>());
        out.hidden_after_ff = read_f32_file(dir / layer.at("hidden_after_ff").at("file").get<std::string>());
        fx.expected.layers.push_back(std::move(out));
    }
    if (cfg.contains("chain") && cfg.at("chain").contains("teacher_forced_steps")) {
        for (const auto & step : cfg.at("chain").at("teacher_forced_steps")) {
            oracle_fixture::chain_step out;
            out.step_index = step.at("step_index").get<int>();
            out.input_pos = step.at("input_pos").get<int>();
            out.active_len = step.at("active_len").get<int>();
            out.input_token = step.at("input_token").get<int>();
            out.expected_top_token = step.at("expected_top_token").get<int>();
            out.activations = read_f32_file(dir / step.at("activations").at("file").get<std::string>());
            out.expected_logits = read_f32_file(dir / step.at("expected_logits").at("file").get<std::string>());
            out.expected_hidden = read_f32_file(dir / step.at("expected_hidden").at("file").get<std::string>());
            out.expected_final_hidden = read_f32_file(dir / step.at("expected_final_hidden").at("file").get<std::string>());
            fx.chain.push_back(std::move(out));
        }
    }
    return fx;
}

static double cosine_similarity(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size() || a.empty()) {
        return 0.0;
    }
    double dot = 0.0;
    double an = 0.0;
    double bn = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += (double) a[i] * (double) b[i];
        an += (double) a[i] * (double) a[i];
        bn += (double) b[i] * (double) b[i];
    }
    if (an <= 0.0 || bn <= 0.0) {
        return 0.0;
    }
    return dot / std::sqrt(an * bn);
}

static json compare_vec(const std::vector<float> & expected, const std::vector<float> & actual, int topk = 8) {
    if (expected.size() != actual.size()) {
        return {
            {"size_mismatch", true},
            {"expected_elements", expected.size()},
            {"actual_elements", actual.size()},
        };
    }
    double sum_abs = 0.0;
    float max_abs = 0.0f;
    for (size_t i = 0; i < expected.size(); ++i) {
        const float diff = std::fabs(expected[i] - actual[i]);
        sum_abs += diff;
        max_abs = std::max(max_abs, diff);
    }
    json out = {
        {"elements", expected.size()},
        {"max_abs_diff", max_abs},
        {"mean_abs_diff", expected.empty() ? 0.0 : sum_abs / (double) expected.size()},
        {"cosine_similarity", cosine_similarity(expected, actual)},
    };
    if ((int) expected.size() == MTP_VOCAB) {
        const int tok_expected = argmax_logits(expected.data(), (int) expected.size());
        const int tok_actual = argmax_logits(actual.data(), (int) actual.size());
        out["expected_top_token"] = tok_expected;
        out["actual_top_token"] = tok_actual;
        out["top1_match"] = tok_expected == tok_actual;
    }
    GGML_UNUSED(topk);
    return out;
}

static int rank_of_token(const float * logits, int n_vocab, int token) {
    if (token < 0 || token >= n_vocab) {
        return -1;
    }
    const float target = logits[token];
    int rank = 1;
    for (int i = 0; i < n_vocab; ++i) {
        if (logits[i] > target) {
            rank += 1;
        }
    }
    return rank;
}

static std::vector<float> runtime_bmm_qk_seq_major_padded(const std::vector<float> & query, int query_rows, int dim, const std::vector<int8_t> & cache_seq_major, int active_len, int max_seq_len, float scale) {
    if ((int) query.size() != query_rows * dim) {
        throw std::runtime_error("runtime_bmm_qk_seq_major_padded query shape mismatch");
    }
    if ((int) cache_seq_major.size() != max_seq_len * dim) {
        throw std::runtime_error("runtime_bmm_qk_seq_major_padded cache shape mismatch");
    }
    std::vector<float> out(query_rows * max_seq_len, 0.0f);
    for (int r = 0; r < query_rows; ++r) {
        const float * q = query.data() + r*dim;
        float * dst = out.data() + r*max_seq_len;
        for (int seq = 0; seq < active_len; ++seq) {
            const int8_t * k = cache_seq_major.data() + seq*dim;
            float acc = 0.0f;
            for (int c = 0; c < dim; ++c) {
                acc += q[c] * (float) k[c];
            }
            dst[seq] = acc * scale;
        }
    }
    return out;
}

static std::vector<float> masked_softmax_rows_mask(const std::vector<float> & scores, int rows, int cols, const std::vector<uint8_t> & mask) {
    if ((int) scores.size() != rows * cols) {
        throw std::runtime_error("masked_softmax_rows_mask scores shape mismatch");
    }
    if ((int) mask.size() != cols) {
        throw std::runtime_error("masked_softmax_rows_mask mask shape mismatch");
    }
    std::vector<float> out(rows * cols, 0.0f);
    for (int row = 0; row < rows; ++row) {
        const float * src = scores.data() + row*cols;
        float * dst = out.data() + row*cols;
        float max_score = -INFINITY;
        for (int col = 0; col < cols; ++col) {
            const float v = mask[col] ? src[col] : -INFINITY;
            dst[col] = v;
            if (v > max_score) {
                max_score = v;
            }
        }
        if (!std::isfinite(max_score)) {
            continue;
        }
        float sum = 0.0f;
        for (int col = 0; col < cols; ++col) {
            if (std::isfinite(dst[col])) {
                dst[col] = std::exp(dst[col] - max_score);
                sum += dst[col];
            } else {
                dst[col] = 0.0f;
            }
        }
        if (sum > 0.0f) {
            for (int col = 0; col < cols; ++col) {
                dst[col] /= sum;
            }
        }
    }
    return out;
}

static std::vector<float> runtime_bmm_v_padded(const std::vector<float> & probs, int query_rows, int max_seq_len, const std::vector<int8_t> & cache_dim_major, int dim, int active_len, float scale) {
    if ((int) probs.size() != query_rows * max_seq_len) {
        throw std::runtime_error("runtime_bmm_v_padded probs shape mismatch");
    }
    if ((int) cache_dim_major.size() != dim * max_seq_len) {
        throw std::runtime_error("runtime_bmm_v_padded cache shape mismatch");
    }
    std::vector<float> out(query_rows * dim, 0.0f);
    for (int r = 0; r < query_rows; ++r) {
        const float * p = probs.data() + r*max_seq_len;
        float * dst = out.data() + r*dim;
        for (int d = 0; d < dim; ++d) {
            const int8_t * src = cache_dim_major.data() + d*max_seq_len;
            float acc = 0.0f;
            for (int seq = 0; seq < active_len; ++seq) {
                acc += p[seq] * ((float) src[seq] * scale);
            }
            dst[d] = acc;
        }
    }
    return out;
}

static mtp_trace_output run_mtp_step_unpadded_traced(const mtp_sidecar & sc, int input_pos, const std::vector<float> & activations, const exported_kv & kv, int active_len) {
    if ((int) activations.size() != MTP_DIM_IN) {
        throw std::runtime_error("mtp activations size mismatch");
    }

    std::vector<float> hidden = sc.pre_project.run(activations);
    hidden = rms_norm_last_dim(hidden, sc.pre_project_norm_weight, MTP_EPS);
    mtp_trace_output trace;

    for (const auto & layer : sc.layers) {
        mtp_layer_trace layer_trace;
        layer_trace.hidden_in = hidden;
        auto pre = rms_norm_last_dim(hidden, layer.pre_attn_norm_weight, MTP_EPS);
        auto q = layer.q_proj.run(pre);
        q = rms_norm_last_dim(q, layer.query_norm_weight, MTP_EPS);
        q = apply_rope(q, layer.heads, layer.head_dim, layer.rope_div, input_pos);
        layer_trace.q_rope = q;

        const std::vector<int8_t> & k_cache = (layer.kv_group == 13) ? kv.k13 : kv.k14;
        const std::vector<int8_t> & v_cache = (layer.kv_group == 13) ? kv.v13 : kv.v14;
        const float k_scale = (layer.kv_group == 13) ? sc.k13_scale : sc.k14_scale;
        const float v_scale = (layer.kv_group == 13) ? sc.v13_scale : sc.v14_scale;
        auto scores = runtime_bmm_qk_seq_major(q, layer.heads, layer.head_dim, k_cache, active_len, k_scale);
        auto probs = masked_softmax_rows_prefix(scores, layer.heads, active_len);
        auto attn = runtime_bmm_v_dim_major(probs, layer.heads, active_len, v_cache, layer.head_dim, v_scale);
        layer_trace.attn = attn;
        auto post = layer.o_proj.run(attn);
        post = rms_norm_last_dim(post, layer.post_attn_norm_weight, MTP_EPS);
        layer_trace.post_norm = post;
        hidden = add_vec(hidden, post);
        layer_trace.hidden_after_attn = hidden;

        auto ff_in = rms_norm_last_dim(hidden, layer.pre_ffw_norm_weight, MTP_EPS);
        auto gate = gelu_vec(layer.gate_proj.run(ff_in));
        auto up = layer.up_proj.run(ff_in);
        auto gated = mul_vec(gate, up);
        auto ff_out = layer.down_proj.run(gated);
        ff_out = rms_norm_last_dim(ff_out, layer.post_ffw_norm_weight, MTP_EPS);
        layer_trace.ff_out_norm = ff_out;
        hidden = add_vec(hidden, ff_out);
        layer_trace.hidden_after_ff = hidden;
        trace.layers.push_back(std::move(layer_trace));
    }

    trace.final_hidden = rms_norm_last_dim(hidden, sc.final_norm_weight, MTP_EPS);
    trace.logits = sc.logits_head.run(trace.final_hidden);
    for (float & value : trace.logits) {
        value = std::tanh(value * sc.logits_soft_cap_mul_in) * sc.logits_soft_cap_mul_out;
    }
    trace.token = argmax_logits(trace.logits.data(), (int) trace.logits.size());
    trace.hidden = sc.hidden_head.run(trace.final_hidden);
    return trace;
}

static mtp_trace_output run_mtp_step_padded_traced(const mtp_sidecar & sc, const oracle_fixture & fx) {
    if ((int) fx.activations.size() != MTP_DIM_IN) {
        throw std::runtime_error("mtp activations size mismatch");
    }

    std::vector<float> hidden = sc.pre_project.run(fx.activations);
    hidden = rms_norm_last_dim(hidden, sc.pre_project_norm_weight, MTP_EPS);
    mtp_trace_output trace;

    for (const auto & layer : sc.layers) {
        mtp_layer_trace layer_trace;
        layer_trace.hidden_in = hidden;
        auto pre = rms_norm_last_dim(hidden, layer.pre_attn_norm_weight, MTP_EPS);
        auto q = layer.q_proj.run(pre);
        q = rms_norm_last_dim(q, layer.query_norm_weight, MTP_EPS);
        q = apply_rope(q, layer.heads, layer.head_dim, layer.rope_div, fx.draft_input_pos);
        layer_trace.q_rope = q;

        const std::vector<int8_t> & k_cache = (layer.kv_group == 13) ? fx.k13 : fx.k14;
        const std::vector<int8_t> & v_cache = (layer.kv_group == 13) ? fx.v13 : fx.v14;
        const float k_scale = (layer.kv_group == 13) ? sc.k13_scale : sc.k14_scale;
        const float v_scale = (layer.kv_group == 13) ? sc.v13_scale : sc.v14_scale;
        auto scores = runtime_bmm_qk_seq_major_padded(q, layer.heads, layer.head_dim, k_cache, fx.active_len, fx.max_seq_len, k_scale);
        auto probs = masked_softmax_rows_mask(scores, layer.heads, fx.max_seq_len, fx.mask);
        auto attn = runtime_bmm_v_padded(probs, layer.heads, fx.max_seq_len, v_cache, layer.head_dim, fx.active_len, v_scale);
        layer_trace.attn = attn;
        auto post = layer.o_proj.run(attn);
        post = rms_norm_last_dim(post, layer.post_attn_norm_weight, MTP_EPS);
        layer_trace.post_norm = post;
        hidden = add_vec(hidden, post);
        layer_trace.hidden_after_attn = hidden;

        auto ff_in = rms_norm_last_dim(hidden, layer.pre_ffw_norm_weight, MTP_EPS);
        auto gate = gelu_vec(layer.gate_proj.run(ff_in));
        auto up = layer.up_proj.run(ff_in);
        auto gated = mul_vec(gate, up);
        auto ff_out = layer.down_proj.run(gated);
        ff_out = rms_norm_last_dim(ff_out, layer.post_ffw_norm_weight, MTP_EPS);
        layer_trace.ff_out_norm = ff_out;
        hidden = add_vec(hidden, ff_out);
        layer_trace.hidden_after_ff = hidden;
        trace.layers.push_back(std::move(layer_trace));
    }

    trace.final_hidden = rms_norm_last_dim(hidden, sc.final_norm_weight, MTP_EPS);
    trace.logits = sc.logits_head.run(trace.final_hidden);
    for (float & value : trace.logits) {
        value = std::tanh(value * sc.logits_soft_cap_mul_in) * sc.logits_soft_cap_mul_out;
    }
    trace.token = argmax_logits(trace.logits.data(), (int) trace.logits.size());
    trace.hidden = sc.hidden_head.run(trace.final_hidden);
    return trace;
}

static mtp_step_output run_mtp_step_padded(
        const mtp_sidecar & sc,
        int input_pos,
        const std::vector<float> & activations,
        const exported_kv & kv,
        int active_len,
        int max_seq_len) {
    if ((int) activations.size() != MTP_DIM_IN) {
        throw std::runtime_error("mtp activations size mismatch");
    }
    std::vector<uint8_t> mask(max_seq_len, 0);
    for (int i = 0; i < active_len && i < max_seq_len; ++i) {
        mask[i] = 1;
    }

    std::vector<float> hidden = sc.pre_project.run(activations);
    hidden = rms_norm_last_dim(hidden, sc.pre_project_norm_weight, MTP_EPS);
    for (const auto & layer : sc.layers) {
        auto pre = rms_norm_last_dim(hidden, layer.pre_attn_norm_weight, MTP_EPS);
        auto q = layer.q_proj.run(pre);
        q = rms_norm_last_dim(q, layer.query_norm_weight, MTP_EPS);
        q = apply_rope(q, layer.heads, layer.head_dim, layer.rope_div, input_pos);

        const std::vector<int8_t> & k_cache = (layer.kv_group == 13) ? kv.k13 : kv.k14;
        const std::vector<int8_t> & v_cache = (layer.kv_group == 13) ? kv.v13 : kv.v14;
        const float k_scale = (layer.kv_group == 13) ? sc.k13_scale : sc.k14_scale;
        const float v_scale = (layer.kv_group == 13) ? sc.v13_scale : sc.v14_scale;
        auto scores = runtime_bmm_qk_seq_major_padded(q, layer.heads, layer.head_dim, k_cache, active_len, max_seq_len, k_scale);
        auto probs = masked_softmax_rows_mask(scores, layer.heads, max_seq_len, mask);
        auto attn = runtime_bmm_v_padded(probs, layer.heads, max_seq_len, v_cache, layer.head_dim, active_len, v_scale);
        auto post = layer.o_proj.run(attn);
        post = rms_norm_last_dim(post, layer.post_attn_norm_weight, MTP_EPS);
        hidden = add_vec(hidden, post);

        auto ff_in = rms_norm_last_dim(hidden, layer.pre_ffw_norm_weight, MTP_EPS);
        auto gate = gelu_vec(layer.gate_proj.run(ff_in));
        auto up = layer.up_proj.run(ff_in);
        auto gated = mul_vec(gate, up);
        auto ff_out = layer.down_proj.run(gated);
        ff_out = rms_norm_last_dim(ff_out, layer.post_ffw_norm_weight, MTP_EPS);
        hidden = add_vec(hidden, ff_out);
    }

    auto final_hidden = rms_norm_last_dim(hidden, sc.final_norm_weight, MTP_EPS);
    auto logits = sc.logits_head.run(final_hidden);
    for (float & value : logits) {
        value = std::tanh(value * sc.logits_soft_cap_mul_in) * sc.logits_soft_cap_mul_out;
    }
    const int token = sc.logits_head.argmax_softcapped(final_hidden, sc.logits_soft_cap_mul_in, sc.logits_soft_cap_mul_out);
    auto next_hidden = sc.hidden_head.run(final_hidden);
    GGML_UNUSED(logits);
    return { token, std::move(next_hidden) };
}

static mtp_step_output run_mtp_step(const mtp_sidecar & sc, int input_pos, const std::vector<float> & activations, const exported_kv & kv, int active_len) {
    auto traced = run_mtp_step_unpadded_traced(sc, input_pos, activations, kv, active_len);
    return { traced.token, std::move(traced.hidden) };
}

struct snapshot {
    std::vector<uint8_t> data;
};

static snapshot take_snapshot(llama_context * ctx) {
    const size_t n = llama_state_get_size(ctx);
    snapshot s;
    s.data.resize(n);
    const size_t got = llama_state_get_data(ctx, s.data.data(), s.data.size());
    s.data.resize(got);
    return s;
}

static void restore_snapshot(llama_context * ctx, const snapshot & s) {
    const size_t got = llama_state_set_data(ctx, s.data.data(), s.data.size());
    if (got != s.data.size()) {
        throw std::runtime_error("failed to restore context state");
    }
}

static void decode_tokens(llama_context * ctx, const std::vector<llama_token> & tokens, int & n_past, bool output_all) {
    if (tokens.empty()) {
        return;
    }
    llama_batch batch = llama_batch_init((int32_t) tokens.size(), 0, 1);
    common_batch_clear(batch);
    for (size_t i = 0; i < tokens.size(); ++i) {
        const bool want_logits = output_all || (i + 1 == tokens.size());
        common_batch_add(batch, tokens[i], n_past + (int) i, { 0 }, want_logits);
    }
    if (llama_decode(ctx, batch) != 0) {
        llama_batch_free(batch);
        throw std::runtime_error("llama_decode failed");
    }
    n_past += (int) tokens.size();
    llama_batch_free(batch);
}

static std::vector<llama_token> build_plain_draft_chunk(
        llama_context * ctx_dft,
        int n_past_dft,
        int draft_max,
        const llama_vocab * vocab,
        const float * current_logits) {
    std::vector<llama_token> draft;
    draft.reserve(draft_max);
    snapshot snap = take_snapshot(ctx_dft);
    int tmp_n_past = n_past_dft;

    llama_token next = argmax_logits_skip_eog(vocab, current_logits, llama_vocab_n_tokens(vocab));
    draft.push_back(next);
    for (int i = 1; i < draft_max; ++i) {
        decode_tokens(ctx_dft, { next }, tmp_n_past, true);
        next = argmax_logits_skip_eog(vocab, llama_get_logits_ith(ctx_dft, -1), llama_vocab_n_tokens(vocab));
        draft.push_back(next);
    }

    restore_snapshot(ctx_dft, snap);
    return draft;
}

static std::vector<llama_token> build_nested_mtp_draft_chunk(
        llama_context * ctx_dft,
        const mtp_sidecar & sc,
        int n_past_dft,
        int draft_max,
        const llama_vocab * vocab,
        const float * current_logits,
        const float * current_hidden,
        mtp_timing * timing = nullptr) {
    std::vector<llama_token> draft;
    draft.reserve(draft_max);

    const llama_token good = argmax_logits_skip_eog(vocab, current_logits, llama_vocab_n_tokens(vocab));
    draft.push_back(good);
    if (draft_max == 1) {
        return draft;
    }

    const int padded_len = n_past_dft + draft_max;
    const auto t_kv0 = std::chrono::steady_clock::now();
    exported_kv kv = export_current_kv(ctx_dft, sc, n_past_dft, padded_len);
    if (timing) {
        timing->kv_export_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t_kv0).count();
    }
    std::vector<float> hidden(current_hidden, current_hidden + MTP_DIM_OUT_HIDDEN);
    llama_token last_token = good;

    for (int step = 1; step < draft_max; ++step) {
        auto embed = sc.lookup_embedder(last_token);
        std::vector<float> activations;
        activations.reserve(embed.size() + hidden.size());
        activations.insert(activations.end(), embed.begin(), embed.end());
        activations.insert(activations.end(), hidden.begin(), hidden.end());

        const auto t_step0 = std::chrono::steady_clock::now();
        auto out = run_mtp_step_padded(sc, n_past_dft + (step - 1), activations, kv, n_past_dft + step, padded_len);
        if (timing) {
            timing->mtp_step_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t_step0).count();
            timing->mtp_steps += 1;
        }
        last_token = (llama_token) out.token;
        draft.push_back(last_token);
        hidden = std::move(out.hidden);
    }

    return draft;
}

static verify_result verify_draft_chunk(
        llama_context * ctx_tgt,
        int & n_past_tgt,
        const std::vector<llama_token> & draft,
        const llama_vocab * vocab) {
    verify_result result;
    result.proposed = (int) draft.size();
    result.accepted.reserve(draft.size() + 1);

    const int n_vocab = llama_vocab_n_tokens(vocab);
    const llama_token predicted0 = argmax_logits_skip_eog(vocab, llama_get_logits_ith(ctx_tgt, -1), n_vocab);
    if (draft.empty()) {
        result.accepted.push_back(predicted0);
        decode_tokens(ctx_tgt, result.accepted, n_past_tgt, false);
        return result;
    }
    if (predicted0 != draft[0]) {
        result.first_token_mismatch = true;
        result.accepted.push_back(predicted0);
        decode_tokens(ctx_tgt, result.accepted, n_past_tgt, false);
        return result;
    }
    if (draft.size() == 1) {
        result.accepted.push_back(draft[0]);
        result.accepted_from_draft = 1;
        result.full_match = true;
        decode_tokens(ctx_tgt, result.accepted, n_past_tgt, false);
        return result;
    }

    snapshot snap = take_snapshot(ctx_tgt);
    int tmp_n_past = n_past_tgt;
    decode_tokens(ctx_tgt, draft, tmp_n_past, true);

    size_t accepted_prefix = 1;
    bool full_match = true;
    for (size_t i = 1; i < draft.size(); ++i) {
        const llama_token predicted = argmax_logits_skip_eog(vocab, llama_get_logits_ith(ctx_tgt, (int32_t) (i - 1)), n_vocab);
        if (predicted != draft[i]) {
            full_match = false;
            for (size_t j = 0; j < i; ++j) {
                result.accepted.push_back(draft[j]);
            }
            result.accepted.push_back(predicted);
            accepted_prefix = i;
            break;
        }
        accepted_prefix = i + 1;
    }

    if (full_match) {
        n_past_tgt = tmp_n_past;
        result.accepted.assign(draft.begin(), draft.end());
        result.accepted_from_draft = (int) draft.size();
        result.full_match = true;
        return result;
    }

    restore_snapshot(ctx_tgt, snap);
    result.accepted_from_draft = (int) accepted_prefix;
    decode_tokens(ctx_tgt, result.accepted, n_past_tgt, false);
    GGML_UNUSED(accepted_prefix);
    return result;
}

static json run_step_compare(
        llama_context * ctx_dft,
        const mtp_sidecar & sc,
        int n_past_dft,
        int n_steps,
        const llama_vocab * vocab) {
    if (n_steps <= 0) {
        throw std::runtime_error("step compare requires n_steps > 0");
    }
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const float * start_logits = llama_get_logits_ith(ctx_dft, -1);
    const float * start_hidden = llama_get_embeddings_ith(ctx_dft, -1);
    if (!start_logits || !start_hidden) {
        throw std::runtime_error("draft logits/hidden unavailable for step compare");
    }

    snapshot snap = take_snapshot(ctx_dft);
    int tmp_n_past = n_past_dft;
    const int fixed_total_len = n_past_dft + n_steps + 1;
    exported_kv fixed_kv = export_current_kv(ctx_dft, sc, n_past_dft, fixed_total_len);

    std::vector<float> prev_base_hidden(start_hidden, start_hidden + MTP_DIM_OUT_HIDDEN);
    std::vector<float> current_chain_hidden = prev_base_hidden;
    llama_token current_input = argmax_logits_skip_eog(vocab, start_logits, n_vocab);
    llama_token current_chain_last = current_input;

    json steps = json::array();
    for (int step = 1; step <= n_steps; ++step) {
        decode_tokens(ctx_dft, { current_input }, tmp_n_past, true);
        const float * base_logits = llama_get_logits_ith(ctx_dft, -1);
        const float * base_hidden_ptr = llama_get_embeddings_ith(ctx_dft, -1);
        if (!base_logits || !base_hidden_ptr) {
            throw std::runtime_error("base logits/hidden unavailable during step compare");
        }
        std::vector<float> base_hidden(base_hidden_ptr, base_hidden_ptr + MTP_DIM_OUT_HIDDEN);
        const llama_token base_next = argmax_logits_skip_eog(vocab, base_logits, n_vocab);

        auto build_activations = [&](llama_token token, const std::vector<float> & hidden) {
            auto embed = sc.lookup_embedder(token);
            std::vector<float> activations;
            activations.reserve(embed.size() + hidden.size());
            activations.insert(activations.end(), embed.begin(), embed.end());
            activations.insert(activations.end(), hidden.begin(), hidden.end());
            return activations;
        };

        const auto fixed_reset_activations = build_activations(current_input, prev_base_hidden);
        auto fixed_reset = run_mtp_step_padded(
                sc,
                tmp_n_past - 1,
                fixed_reset_activations,
                fixed_kv,
                tmp_n_past,
                fixed_total_len);

        exported_kv fresh_kv = export_current_kv(ctx_dft, sc, tmp_n_past, tmp_n_past + 1);
        const auto fresh_reset_activations = build_activations(current_input, prev_base_hidden);
        auto fresh_reset = run_mtp_step_padded(
                sc,
                tmp_n_past - 1,
                fresh_reset_activations,
                fresh_kv,
                tmp_n_past,
                tmp_n_past + 1);

        const auto current_chain_activations = build_activations(current_chain_last, current_chain_hidden);
        auto current_chain = run_mtp_step_padded(
                sc,
                tmp_n_past - 1,
                current_chain_activations,
                fixed_kv,
                tmp_n_past,
                fixed_total_len);

        auto fresh_chain = run_mtp_step_padded(
                sc,
                tmp_n_past - 1,
                current_chain_activations,
                fresh_kv,
                tmp_n_past,
                tmp_n_past + 1);

        // For ranks, recompute logits using the traced path.
        oracle_fixture fresh_fx;
        fresh_fx.draft_input_pos = tmp_n_past - 1;
        fresh_fx.active_len = tmp_n_past;
        fresh_fx.max_seq_len = tmp_n_past + 1;
        fresh_fx.activations = fresh_reset_activations;
        fresh_fx.mask.assign(fresh_fx.max_seq_len, 0);
        for (int i = 0; i < fresh_fx.active_len && i < fresh_fx.max_seq_len; ++i) {
            fresh_fx.mask[i] = 1;
        }
        fresh_fx.k13 = fresh_kv.k13;
        fresh_fx.k14 = fresh_kv.k14;
        fresh_fx.v13 = fresh_kv.v13;
        fresh_fx.v14 = fresh_kv.v14;
        auto fresh_reset_traced = run_mtp_step_padded_traced(sc, fresh_fx);

        oracle_fixture fixed_fx = fresh_fx;
        fixed_fx.max_seq_len = fixed_total_len;
        fixed_fx.mask.assign(fixed_total_len, 0);
        for (int i = 0; i < tmp_n_past && i < fixed_total_len; ++i) {
            fixed_fx.mask[i] = 1;
        }
        fixed_fx.k13 = fixed_kv.k13;
        fixed_fx.k14 = fixed_kv.k14;
        fixed_fx.v13 = fixed_kv.v13;
        fixed_fx.v14 = fixed_kv.v14;
        auto fixed_reset_traced = run_mtp_step_padded_traced(sc, fixed_fx);

        oracle_fixture current_chain_fx = fixed_fx;
        current_chain_fx.activations = current_chain_activations;
        auto current_chain_traced = run_mtp_step_padded_traced(sc, current_chain_fx);

        oracle_fixture fresh_chain_fx = fresh_fx;
        fresh_chain_fx.activations = current_chain_activations;
        auto fresh_chain_traced = run_mtp_step_padded_traced(sc, fresh_chain_fx);

        steps.push_back({
            {"step_index", step},
            {"input_token", current_input},
            {"base_next_token", base_next},
            {"base_token_rank_fixed_reset", rank_of_token(fixed_reset_traced.logits.data(), n_vocab, base_next)},
            {"base_token_rank_fresh_reset", rank_of_token(fresh_reset_traced.logits.data(), n_vocab, base_next)},
            {"base_token_rank_current_chain", rank_of_token(current_chain_traced.logits.data(), n_vocab, base_next)},
            {"base_token_rank_fresh_chain", rank_of_token(fresh_chain_traced.logits.data(), n_vocab, base_next)},
            {"fixed_reset", {
                {"token", fixed_reset.token},
                {"top1_match", fixed_reset.token == base_next},
                {"hidden_cosine_to_base", cosine_similarity(fixed_reset.hidden, base_hidden)},
            }},
            {"fresh_reset", {
                {"token", fresh_reset.token},
                {"top1_match", fresh_reset.token == base_next},
                {"hidden_cosine_to_base", cosine_similarity(fresh_reset.hidden, base_hidden)},
            }},
            {"current_chain", {
                {"token", current_chain.token},
                {"top1_match", current_chain.token == base_next},
                {"hidden_cosine_to_base", cosine_similarity(current_chain.hidden, base_hidden)},
            }},
            {"fresh_chain", {
                {"token", fresh_chain.token},
                {"top1_match", fresh_chain.token == base_next},
                {"hidden_cosine_to_base", cosine_similarity(fresh_chain.hidden, base_hidden)},
            }},
        });

        prev_base_hidden = std::move(base_hidden);
        current_input = base_next;
        current_chain_last = (llama_token) current_chain.token;
        current_chain_hidden = std::move(current_chain.hidden);
    }

    restore_snapshot(ctx_dft, snap);
    return {
        {"mode", "step_compare"},
        {"n_steps", n_steps},
        {"steps", steps},
    };
}

static std::vector<std::string> strip_custom_args(
        int argc,
        char ** argv,
        std::optional<std::string> & sidecar_dir,
        std::optional<std::string> & fixture_dir,
        std::optional<int> & step_compare,
        std::optional<std::string> & report_path,
        bool & draft_only) {
    std::vector<std::string> out;
    out.reserve(argc);
    out.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--gemma4-mtp-sidecar") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--gemma4-mtp-sidecar requires a directory");
            }
            sidecar_dir = argv[++i];
            continue;
        }
        if (arg == "--gemma4-mtp-fixture") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--gemma4-mtp-fixture requires a directory");
            }
            fixture_dir = argv[++i];
            continue;
        }
        if (arg == "--gemma4-mtp-step-compare") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--gemma4-mtp-step-compare requires an integer");
            }
            step_compare = std::stoi(argv[++i]);
            continue;
        }
        if (arg == "--gemma4-mtp-draft-only") {
            draft_only = true;
            continue;
        }
        if (arg == "--gemma4-mtp-report") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--gemma4-mtp-report requires a path");
            }
            report_path = argv[++i];
            continue;
        }
        out.push_back(std::move(arg));
    }
    return out;
}

int main(int argc, char ** argv) {
    try {
        std::setlocale(LC_NUMERIC, "C");

        std::optional<std::string> sidecar_dir;
        std::optional<std::string> fixture_dir;
        std::optional<int> step_compare;
        std::optional<std::string> report_path;
        bool draft_only = false;
        std::vector<std::string> stripped = strip_custom_args(argc, argv, sidecar_dir, fixture_dir, step_compare, report_path, draft_only);
        std::vector<char *> argv2;
        argv2.reserve(stripped.size());
        for (auto & s : stripped) {
            argv2.push_back(s.data());
        }
        argc = (int) argv2.size();
        argv = argv2.data();

        const bool use_mtp = sidecar_dir.has_value();
        const bool offline_fixture = fixture_dir.has_value();
        proto_metrics metrics;
        metrics.draft_only = draft_only;
        metrics.use_mtp = use_mtp;
        std::optional<mtp_sidecar> mtp;
        if (use_mtp) {
            mtp.emplace(load_mtp_sidecar(*sidecar_dir));
        }

        if (offline_fixture) {
            if (!mtp) {
                throw std::runtime_error("--gemma4-mtp-fixture requires --gemma4-mtp-sidecar");
            }
            oracle_fixture fx = load_mtp_oracle_fixture(*fixture_dir);
            exported_kv kv {
                fx.k13,
                fx.v13,
                fx.k14,
                fx.v14,
            };
            auto unpadded = run_mtp_step_unpadded_traced(*mtp, fx.draft_input_pos, fx.activations, kv, fx.active_len);
            auto padded = run_mtp_step_padded_traced(*mtp, fx);
            json layer_reports = json::array();
            for (size_t i = 0; i < fx.expected.layers.size(); ++i) {
                const auto & exp = fx.expected.layers.at(i);
                const auto & unp = unpadded.layers.at(i);
                const auto & pad = padded.layers.at(i);
                layer_reports.push_back({
                    {"layer_index", i},
                    {"hidden_in_unpadded", compare_vec(exp.hidden_in, unp.hidden_in)},
                    {"hidden_in_padded", compare_vec(exp.hidden_in, pad.hidden_in)},
                    {"q_rope_unpadded", compare_vec(exp.q_rope, unp.q_rope)},
                    {"q_rope_padded", compare_vec(exp.q_rope, pad.q_rope)},
                    {"attn_unpadded", compare_vec(exp.attn, unp.attn)},
                    {"attn_padded", compare_vec(exp.attn, pad.attn)},
                    {"post_norm_unpadded", compare_vec(exp.post_norm, unp.post_norm)},
                    {"post_norm_padded", compare_vec(exp.post_norm, pad.post_norm)},
                    {"hidden_after_attn_unpadded", compare_vec(exp.hidden_after_attn, unp.hidden_after_attn)},
                    {"hidden_after_attn_padded", compare_vec(exp.hidden_after_attn, pad.hidden_after_attn)},
                    {"ff_out_norm_unpadded", compare_vec(exp.ff_out_norm, unp.ff_out_norm)},
                    {"ff_out_norm_padded", compare_vec(exp.ff_out_norm, pad.ff_out_norm)},
                    {"hidden_after_ff_unpadded", compare_vec(exp.hidden_after_ff, unp.hidden_after_ff)},
                    {"hidden_after_ff_padded", compare_vec(exp.hidden_after_ff, pad.hidden_after_ff)},
                });
            }
            json chain_reports = json::array();
            for (const auto & step : fx.chain) {
                oracle_fixture step_fx = fx;
                step_fx.draft_input_pos = step.input_pos;
                step_fx.active_len = step.active_len;
                step_fx.good_token = step.input_token;
                step_fx.activations = step.activations;
                auto got = run_mtp_step_padded_traced(*mtp, step_fx);
                chain_reports.push_back({
                    {"step_index", step.step_index},
                    {"input_pos", step.input_pos},
                    {"active_len", step.active_len},
                    {"input_token", step.input_token},
                    {"expected_top_token", step.expected_top_token},
                    {"actual_top_token", got.token},
                    {"top1_match", got.token == step.expected_top_token},
                    {"logits", compare_vec(step.expected_logits, got.logits)},
                    {"hidden", compare_vec(step.expected_hidden, got.hidden)},
                    {"final_hidden", compare_vec(step.expected_final_hidden, got.final_hidden)},
                });
            }
            json report = {
                {"mode", "offline_fixture_compare"},
                {"fixture_dir", *fixture_dir},
                {"sidecar_dir", *sidecar_dir},
                {"oracle", {
                    {"expected_token", fx.expected.token},
                }},
                {"unpadded", {
                    {"logits", compare_vec(fx.expected.logits, unpadded.logits)},
                    {"hidden", compare_vec(fx.expected.hidden, unpadded.hidden)},
                    {"final_hidden", compare_vec(fx.expected.final_hidden, unpadded.final_hidden)},
                }},
                {"padded", {
                    {"logits", compare_vec(fx.expected.logits, padded.logits)},
                    {"hidden", compare_vec(fx.expected.hidden, padded.hidden)},
                    {"final_hidden", compare_vec(fx.expected.final_hidden, padded.final_hidden)},
                }},
                {"layers", layer_reports},
                {"chain_teacher_forced", chain_reports},
            };
            if (report_path) {
                std::ofstream out(*report_path);
                if (!out) {
                    throw std::runtime_error("failed to open report path");
                }
                out << report.dump(2);
            } else {
                LOG("%s\n", report.dump(2).c_str());
            }
            return 0;
        }

        common_params params;

        common_init();
        if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
            return 1;
        }
        if (params.n_predict < -1) {
            LOG_ERR("%s: --n-predict must be >= -1\n", __func__);
            return 1;
        }
        if (params.n_parallel != 1) {
            LOG_ERR("%s: prototype requires --parallel 1\n", __func__);
            return 1;
        }
        if (!draft_only && params.speculative.mparams_dft.path.empty()) {
            LOG_ERR("%s: --model-draft is required unless --gemma4-mtp-draft-only is set\n", __func__);
            return 1;
        }

        llama_backend_init();
        llama_numa_init(params.numa);

        common_params params_dft = params;
        params_dft.n_parallel = 1;
        params_dft.n_ctx = draft_only ? params.n_ctx : params.speculative.n_ctx;
        params_dft.n_batch = std::max(params_dft.n_batch, params_dft.n_ctx);
        params_dft.devices.clear();
        if (!draft_only) {
            params_dft.model = params.speculative.mparams_dft;
        }
        params_dft.n_gpu_layers = 0;
        params_dft.embedding = true;
        params_dft.cache_type_k = GGML_TYPE_F16;
        params_dft.cache_type_v = GGML_TYPE_F16;

        std::unique_ptr<common_init_result> llama_init_tgt;
        llama_model * model_tgt = nullptr;
        llama_context * ctx_tgt = nullptr;
        if (!draft_only) {
            llama_init_tgt = common_init_from_params(params);
            model_tgt = llama_init_tgt->model();
            ctx_tgt = llama_init_tgt->context();
        }

        auto llama_init_dft = common_init_from_params(params_dft);
        llama_model * model_dft = llama_init_dft->model();
        llama_context * ctx_dft = llama_init_dft->context();
        llama_set_embeddings(ctx_dft, true);

        const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);
        const llama_vocab * vocab_tgt = draft_only ? vocab_dft : llama_model_get_vocab(model_tgt);
        if (!draft_only && llama_vocab_n_tokens(vocab_tgt) != llama_vocab_n_tokens(vocab_dft)) {
            LOG_ERR("%s: target and draft vocab sizes differ\n", __func__);
            return 1;
        }

        std::vector<llama_token> prompt = common_tokenize(ctx_dft, params.prompt, true, true);
        if ((draft_only ? 0 : ((int) prompt.size() > (int) llama_n_ctx(ctx_tgt))) || (int) prompt.size() > (int) llama_n_ctx(ctx_dft)) {
            LOG_ERR("%s: prompt too long\n", __func__);
            return 1;
        }

        llama_context * ctx_out = draft_only ? ctx_dft : ctx_tgt;

        LOG("\n\n");
        for (auto tok : prompt) {
            LOG("%s", common_token_to_piece(ctx_out, tok).c_str());
        }

        int n_past_tgt = 0;
        int n_past_dft = 0;
        if (!draft_only) {
            decode_tokens(ctx_tgt, prompt, n_past_tgt, false);
        }
        decode_tokens(ctx_dft, prompt, n_past_dft, true);

        if (step_compare.has_value()) {
            if (!mtp) {
                throw std::runtime_error("--gemma4-mtp-step-compare requires --gemma4-mtp-sidecar");
            }
            json report = run_step_compare(ctx_dft, *mtp, n_past_dft, *step_compare, vocab_dft);
            if (report_path) {
                std::ofstream out(*report_path);
                if (!out) {
                    throw std::runtime_error("failed to open report path");
                }
                out << report.dump(2);
            } else {
                LOG("%s\n", report.dump(2).c_str());
            }
            return 0;
        }

        const int draft_max = std::max(1, params.speculative.n_max);
        const int n_predict_max = params.n_predict < 0 ? 256 : params.n_predict;
        int n_predict = 0;
        std::string generated_text;

        while (n_predict < n_predict_max) {
            const float * dft_logits = llama_get_logits_ith(ctx_dft, -1);
            const float * dft_hidden = llama_get_embeddings_ith(ctx_dft, -1);
            if (!dft_logits || !dft_hidden) {
                throw std::runtime_error("draft logits/hidden unavailable");
            }

            metrics.chunks += 1;
            std::vector<llama_token> draft;
            mtp_timing mtp_t{};
            const auto t_draft0 = std::chrono::steady_clock::now();
            if (use_mtp) {
                draft = build_nested_mtp_draft_chunk(ctx_dft, *mtp, n_past_dft, draft_max, vocab_dft, dft_logits, dft_hidden, &mtp_t);
            } else {
                draft = build_plain_draft_chunk(ctx_dft, n_past_dft, draft_max, vocab_dft, dft_logits);
            }
            metrics.draft_build_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t_draft0).count();
            metrics.kv_export_s += mtp_t.kv_export_s;
            metrics.mtp_step_s += mtp_t.mtp_step_s;
            metrics.mtp_steps += mtp_t.mtp_steps;
            metrics.proposed_tokens += (int) draft.size();

            std::vector<llama_token> accepted;
            if (draft_only) {
                accepted = draft;
                metrics.accepted_tokens += (int) accepted.size();
                metrics.accepted_from_draft += (int) accepted.size();
                metrics.full_match_chunks += 1;
            } else {
                const auto t_verify0 = std::chrono::steady_clock::now();
                verify_result vr = verify_draft_chunk(ctx_tgt, n_past_tgt, draft, vocab_tgt);
                metrics.verify_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t_verify0).count();
                accepted = std::move(vr.accepted);
                metrics.accepted_tokens += (int) accepted.size();
                metrics.accepted_from_draft += vr.accepted_from_draft;
                metrics.verifier_substitutions += (int) accepted.size() - vr.accepted_from_draft;
                metrics.first_token_mismatches += vr.first_token_mismatch ? 1 : 0;
                metrics.full_match_chunks += vr.full_match ? 1 : 0;
            }
            const auto t_sync0 = std::chrono::steady_clock::now();
            decode_tokens(ctx_dft, accepted, n_past_dft, true);
            metrics.draft_sync_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t_sync0).count();

            for (llama_token tok : accepted) {
                if (llama_vocab_is_eog(vocab_tgt, tok)) {
                    goto done;
                }
                auto piece = common_token_to_piece(ctx_out, tok);
                LOG("%s", piece.c_str());
                generated_text += piece;
                ++n_predict;
                metrics.emitted_tokens += 1;
                if (n_predict >= n_predict_max) {
                    goto done;
                }
            }
        }

        done:
        LOG("\n\n");
        if (report_path) {
            json report = {
                {"draft_only", metrics.draft_only},
                {"use_mtp", metrics.use_mtp},
                {"chunks", metrics.chunks},
                {"proposed_tokens", metrics.proposed_tokens},
                {"accepted_tokens", metrics.accepted_tokens},
                {"accepted_from_draft", metrics.accepted_from_draft},
                {"verifier_substitutions", metrics.verifier_substitutions},
                {"full_match_chunks", metrics.full_match_chunks},
                {"first_token_mismatches", metrics.first_token_mismatches},
                {"emitted_tokens", metrics.emitted_tokens},
                {"draft_build_s", metrics.draft_build_s},
                {"verify_s", metrics.verify_s},
                {"draft_sync_s", metrics.draft_sync_s},
                {"kv_export_s", metrics.kv_export_s},
                {"mtp_step_s", metrics.mtp_step_s},
                {"mtp_steps", metrics.mtp_steps},
                {"acceptance_rate", metrics.proposed_tokens > 0 ? (double) metrics.accepted_from_draft / (double) metrics.proposed_tokens : 0.0},
                {"emitted_per_chunk", metrics.chunks > 0 ? (double) metrics.emitted_tokens / (double) metrics.chunks : 0.0},
                {"generated_text", generated_text},
            };
            std::ofstream out(*report_path);
            if (!out) {
                throw std::runtime_error("failed to open report path");
            }
            out << report.dump(2);
        }
        return 0;
    } catch (const std::exception & e) {
        LOG_ERR("%s\n", e.what());
        return 1;
    }
}
