#pragma once

#include "typedefs.h"

// ============================================================================
// [S1/S2] Llama2 110M configuration + Llama3.2용으로 수정된 커널 코드
//   목적: failed1의 "코드 변경"(on-the-fly embed dequant, GQA-generic 경로,
//         8-way score 누산, baseline matmul)만 110M 가중치로 검증.
//   config는 110M 원본과 동일. lut_data.h도 110M(theta=10000) 버전 사용 필수!
//   기대 결과: 기존 improved 110M과 동일한 토큰 시퀀스 출력.
// ============================================================================
static constexpr int dim = 768;          // Model embedding dimension
static constexpr int hidden_dim = 2048;  // Hidden dimension for feed-forward layers
static constexpr int n_layers = 12;      // Number of transformer layers
static constexpr int n_heads = 12;       // Number of attention (query) heads
static constexpr int n_kv_heads = 12;    // Number of key-value heads (=n_heads -> kv_mul=1, GQA 경로가 MHA로 동작)
static constexpr int vocab_size = 32000; // Vocabulary size (Llama 2 sentencepiece)
static constexpr int seq_len = 1024;     // Max sequence length
static constexpr int GS = 64;            // Quantization group size (must match export)

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
