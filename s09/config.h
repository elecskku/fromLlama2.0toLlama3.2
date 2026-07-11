#pragma once

#include "typedefs.h"

// ============================================================================
// Llama 3.2 1B configuration (was: Llama2 110M)
//   변경된 값만:  dim 768->2048, hidden_dim 2048->8192, n_layers 12->16,
//                 n_heads 12->32, n_kv_heads 12->8 (GQA 활성화!),
//                 vocab_size 32000->128256
//   head_size = dim/n_heads = 2048/32 = 64  (110M과 동일 -> 누산기/언롤 그대로 유효)
//   kv_dim    = dim*n_kv_heads/n_heads = 512
//   kv_mul    = n_heads/n_kv_heads = 4  (GQA: Q 4개가 KV 1개 공유)
//   seq_len:  데모용 2048. KV캐시 DDR/att BRAM에 직접 비례하므로 필요시 축소 가능.
//   GS=64: Dylan version2_export 기본값과 일치(필수).
// ============================================================================
static constexpr int dim = 2048;          // Model embedding dimension
static constexpr int hidden_dim = 8192;   // Hidden dimension for feed-forward layers
static constexpr int n_layers = 16;       // Number of transformer layers
static constexpr int n_heads = 32;        // Number of attention (query) heads
static constexpr int n_kv_heads = 8;      // Number of key-value heads (GQA)
static constexpr int vocab_size = 128256; // Vocabulary size (Llama 3 tiktoken)
static constexpr int seq_len = 2048;      // Max sequence length (tunable; lower => less memory)
static constexpr int GS = 64;             // Quantization group size (must match export)

constexpr Config config = {
    .dim = dim,
    .hidden_dim = hidden_dim,
    .n_layers = n_layers,
    .n_heads = n_heads,
    .n_kv_heads = n_kv_heads,
    .vocab_size = vocab_size,
    .seq_len = seq_len,
    .GS = GS,
};
