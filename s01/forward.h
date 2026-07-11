#include "typedefs.h"
#include "config.h"
#include <math.h>
#include <cstring>

extern "C" void forward(Transformer<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS> *transformer, 
                        int token, 
                        int pos, 
                        float key_cache[n_layers * seq_len * ((dim * n_kv_heads) / n_heads)], 
                        float value_cache[n_layers * seq_len * ((dim * n_kv_heads) / n_heads)], 
                        float out[vocab_size]);  // Forward pass declaration

template <int S>
void dequantize(QuantizedTensor<S> *qx, float x[S], int GS) {  // Dequantize tensor
    for (int i = 0; i < S; i++) {
        x[i] = qx->q[i] * qx->s[i / GS];
    }
}

// forward.h 안의 quantize<S>() 몸체 — LUT-최소 프로파일
template <int S>
void quantize(QuantizedTensor<S> *qx, float x[S], int GS) {
    const int G = GS;
    const int groups = S / G;

//     // 함수/연산 인스턴스 공유(코드 지시문 버전, 안 먹히면 Tcl로 주기)
// #pragma HLS ALLOCATION instances=fmul limit=1 operation
// #pragma HLS ALLOCATION instances=fadd limit=1 operation
// #pragma HLS ALLOCATION instances=fcmp limit=1 operation
// #pragma HLS ALLOCATION instances=fdiv limit=1 operation
// //  ^ 위 pragma가 버전마다 제한될 수 있음 → 아래 ‘Tcl 대안’ 참고

groups:
    for (int g = 0; g < groups; g++) {
        const int base = g * G;

        float wmax = 0.f;
    // ★ 언롤 금지, 요소 루프에 PIPELINE II=1 (하나의 FP 비교기만 사용)
    max_loop:
        for (int i = 0; i < G; i++) {
#pragma HLS PIPELINE II=1
            float xi = x[base + i];
            float ab = (xi < 0.f) ? -xi : xi;
            wmax = (ab > wmax) ? ab : wmax;
        }

        float inv = (wmax > 0.f) ? (127.0f / wmax) : 0.f;
        qx->s[g]  = (wmax > 0.f) ? (wmax / 127.0f) : 0.f;

    // ★ 언롤 금지, 요소 루프에 PIPELINE II=1 (하나의 FP 곱/덧만 사용)
    q_loop:
        for (int i = 0; i < G; i++) {
#pragma HLS PIPELINE II=1
            float z = x[base + i] * inv;
            int qi  = (int)(z >= 0.f ? z + 0.5f : z - 0.5f);
            if (qi > 127)  qi = 127;
            if (qi < -127) qi = -127;
            qx->q[base + i] = (int8_t)qi;
        }
    }
}
