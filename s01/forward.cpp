#include "forward.h"
#include "config.h"
#include <cstring>
#include <cmath>

#include "rmsnorm.h"
#include "softmax.h"
#include "matmul.h"
#include "lut_data.h"

// ============================================================================
// Fast trigonometric functions using LUT + linear interpolation
// ============================================================================
inline float fast_sin(float x) {
    #pragma HLS INLINE
    const float TWO_PI = 6.28318530718f;
    const float INV_STEP = 40.74366543152521f; // 256 / (2*PI)
    x = x - floorf(x / TWO_PI) * TWO_PI;
    float idx_float = x * INV_STEP;
    int idx = (int)idx_float;
    float frac = idx_float - (float)idx;
    int idx_next = (idx + 1) & 255;
    return SIN_LUT[idx] + frac * (SIN_LUT[idx_next] - SIN_LUT[idx]);
}

inline float fast_cos(float x) {
    #pragma HLS INLINE
    const float TWO_PI = 6.28318530718f;
    const float INV_STEP = 40.74366543152521f; // 256 / (2*PI)
    x = x - floorf(x / TWO_PI) * TWO_PI;
    float idx_float = x * INV_STEP;
    int idx = (int)idx_float;
    float frac = idx_float - (float)idx;
    int idx_next = (idx + 1) & 255;
    return COS_LUT[idx] + frac * (COS_LUT[idx_next] - COS_LUT[idx]);
}

// Forward pass - Timing-Safe Version (100MHz Target)
extern "C" void forward(Transformer<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS> *transformer, 
                        int token, 
                        int pos, 
                        float key_cache[n_layers * seq_len * ((dim * n_kv_heads) / n_heads)], 
                        float value_cache[n_layers * seq_len * ((dim * n_kv_heads) / n_heads)], 
                        float out[vocab_size]) {
    enum {
      KV_DIM          = (dim * n_kv_heads) / n_heads,
      KV_CACHE_DEPTH  = n_layers * seq_len * KV_DIM,
      OUT_DEPTH       = vocab_size
    };
    
    #pragma HLS INTERFACE mode=m_axi port=transformer bundle=gmem0 offset=slave depth=1
    #pragma HLS INTERFACE mode=s_axilite port=token
    #pragma HLS INTERFACE mode=s_axilite port=pos

    #pragma HLS INTERFACE m_axi port=key_cache  bundle=gmem1 offset=slave depth=KV_CACHE_DEPTH
    #pragma HLS INTERFACE m_axi port=value_cache bundle=gmem2 offset=slave depth=KV_CACHE_DEPTH
    #pragma HLS INTERFACE m_axi port=out        bundle=gmem3 offset=slave depth=OUT_DEPTH
    #pragma HLS INTERFACE mode=s_axilite port=return
    
    auto w = &transformer->weights;

    float x[config.dim];
    #pragma HLS BIND_STORAGE variable=x type=ram_2p impl=bram
    float xb[config.dim];
    #pragma HLS BIND_STORAGE variable=xb type=ram_2p impl=bram
    float xb2[config.dim];
    #pragma HLS BIND_STORAGE variable=xb2 type=ram_2p impl=bram
    float hb[config.hidden_dim];
    #pragma HLS BIND_STORAGE variable=hb type=ram_2p impl=bram
    float hb2[config.hidden_dim];
    #pragma HLS BIND_STORAGE variable=hb2 type=ram_2p impl=bram
    QuantizedTensor<config.dim> xq;
    #pragma HLS BIND_STORAGE variable=xq type=ram_2p impl=bram
    QuantizedTensor<config.hidden_dim> hq;
    #pragma HLS BIND_STORAGE variable=hq type=ram_2p impl=bram
    float q[config.dim];
    #pragma HLS BIND_STORAGE variable=q type=ram_2p impl=bram
    float k[(config.dim * config.n_kv_heads) / config.n_heads];
    #pragma HLS BIND_STORAGE variable=k type=ram_2p impl=bram
    float v[(config.dim * config.n_kv_heads) / config.n_heads];
    #pragma HLS BIND_STORAGE variable=v type=ram_2p impl=bram
    float att[config.n_heads * config.seq_len];
    #pragma HLS BIND_STORAGE variable=att type=ram_2p impl=bram

    constexpr int kv_dim = (config.dim * config.n_kv_heads) / config.n_heads;
    constexpr int kv_mul = config.n_heads / config.n_kv_heads;
    constexpr int head_size = dim / config.n_heads;

    // On-the-fly dequant of the current token's embedding row (no 1GB fp32 table).
    // q_tokens layout is row-major (vocab_size, dim); group size GS along the row.
    // scale index for element j of row `token`: (token*dim + j) / GS
    {
        const QuantizedTensor<vocab_size * dim>* qt = w->q_tokens;
        const int row_base = token * dim;
    embed_dequant:
        for (int j = 0; j < dim; j++) {
            #pragma HLS pipeline II=1
            x[j] = qt->q[row_base + j] * qt->s[(row_base + j) / GS];
        }
    }

main_forward_loop:
    for (int l = 0; l < config.n_layers; l++) {
        #pragma HLS loop_tripcount min=1 max=32 avg=12
        
        rmsnorm<dim>(xb, x, w->rms_att_weight + l * dim);
        quantize(&xq, xb, GS);
        matmul<dim, dim>(q, xq.q, xq.s, (w->wq + l)->q, (w->wq + l)->s);
        matmul<dim, kv_dim>(k, xq.q, xq.s, (w->wk + l)->q, (w->wk + l)->s);
        matmul<dim, kv_dim>(v, xq.q, xq.s, (w->wv + l)->q, (w->wv + l)->s);

    rotation1:
        for (int i = 0; i < kv_dim; i += 2) {
            #pragma HLS pipeline II=32
            #pragma HLS loop_tripcount min=64 max=512 avg=256
            
            int head_dim = i % head_size;
            float freq = ROPE_FREQS[head_dim];
            float val = pos * freq;
            float fcr = fast_cos(val);
            float fci = fast_sin(val);
            float v0_q = q[i];
            float v1_q = q[i + 1];
            q[i] = v0_q * fcr - v1_q * fci;
            q[i + 1] = v0_q * fci + v1_q * fcr;
            float v0_k = k[i];
            float v1_k = k[i + 1];
            k[i] = v0_k * fcr - v1_k * fci;
            k[i + 1] = v0_k * fci + v1_k * fcr;
        }

    rotation2:
        for (int i = kv_dim; i < dim; i += 2) {
            #pragma HLS pipeline II=32
            #pragma HLS loop_tripcount min=0 max=512 avg=256
            
            int head_dim = i % head_size;
            float freq = ROPE_FREQS[head_dim];
            float val = pos * freq;
            float fcr = fast_cos(val);
            float fci = fast_sin(val);
            float v0 = q[i];
            float v1 = q[i + 1];
            q[i] = v0 * fcr - v1 * fci;
            q[i + 1] = v0 * fci + v1 * fcr;
        }
        
        int loff = l * config.seq_len * kv_dim;
        float *key_cache_row = key_cache + loff + pos * kv_dim;
        float *value_cache_row = value_cache + loff + pos * kv_dim;
        std::memcpy(key_cache_row, k, kv_dim * sizeof(*key_cache_row));
        std::memcpy(value_cache_row, v, kv_dim * sizeof(*value_cache_row));

#ifndef USE_ONLINE_SOFTMAX
#define USE_ONLINE_SOFTMAX 1
#endif
#ifndef ONLINE_P8
#define ONLINE_P8 0
#endif
#ifndef TAIL_CUT
#define TAIL_CUT 0
#endif

    multihead_attention:
        for (int h = 0; h < n_heads; h++) {
            #pragma HLS loop_tripcount min=1 max=32 avg=12
            
            const int q_offset  = h * head_size;
            const int xb_offset = h * head_size;
            const float inv_sqrt_head = 1.0f / sqrtf(head_size);

            float m = -1e30f;
            float curr_l = 0.f;
            float y_num[head_size];   // was [768]; head_size(=64) entries actually used. saves BRAM per head.
            #pragma HLS BIND_STORAGE variable=y_num type=ram_2p impl=bram
            
        init_y_num:
            for (int i = 0; i < head_size; i++) {
                #pragma HLS pipeline II=1
                #pragma HLS loop_tripcount min=32 max=128 avg=64
                y_num[i] = 0.f;
            }

            float local_k[head_size];
            #pragma HLS BIND_STORAGE variable=local_k type=ram_2p impl=bram
            float local_v[head_size];
            #pragma HLS BIND_STORAGE variable=local_v type=ram_2p impl=bram

        attention_timesteps:
            for (int t = 0; t <= pos; t++) {
                #pragma HLS loop_tripcount min=1 max=257 avg=129
                
                const int key_offset = loff + t * kv_dim + (h / kv_mul) * head_size;
                const int v_offset   = key_offset;

                if (t == pos) {
                load_kv_bram:
                    for (int i = 0; i < head_size; i++) {
                        #pragma HLS pipeline II=1
                        local_k[i] = k[i + (h / kv_mul) * head_size];
                        local_v[i] = v[i + (h / kv_mul) * head_size];
                    }
                } else {
                    std::memcpy(local_k, key_cache   + key_offset, head_size * sizeof(float));
                    std::memcpy(local_v, value_cache + v_offset,   head_size * sizeof(float));
                }

                // [최적화] 단일 score 누산은 float 덧셈 의존사슬로 II=4가 됨.
                //          부분합 4개로 쪼개 의존을 끊고 II=2로. (matmul과 동일 원리)
                //          head_size(=64)는 4의 배수임을 전제로 함.
                float sc0 = 0.0f, sc1 = 0.0f, sc2 = 0.0f, sc3 = 0.0f;
                float sc4 = 0.0f, sc5 = 0.0f, sc6 = 0.0f, sc7 = 0.0f;
            qk_dot_product:
                for (int i = 0; i < head_size; i += 8) {
                    #pragma HLS pipeline II=4
                    #pragma HLS loop_tripcount min=4 max=16 avg=8
                    #pragma HLS BIND_OP variable=sc0 op=fadd latency=2
                    #pragma HLS BIND_OP variable=sc1 op=fadd latency=2
                    #pragma HLS BIND_OP variable=sc2 op=fadd latency=2
                    #pragma HLS BIND_OP variable=sc3 op=fadd latency=2
                    #pragma HLS BIND_OP variable=sc4 op=fadd latency=2
                    #pragma HLS BIND_OP variable=sc5 op=fadd latency=2
                    #pragma HLS BIND_OP variable=sc6 op=fadd latency=2
                    #pragma HLS BIND_OP variable=sc7 op=fadd latency=2
                    sc0 += q[i     + q_offset] * local_k[i];
                    sc1 += q[i + 1 + q_offset] * local_k[i + 1];
                    sc2 += q[i + 2 + q_offset] * local_k[i + 2];
                    sc3 += q[i + 3 + q_offset] * local_k[i + 3];
                    sc4 += q[i + 4 + q_offset] * local_k[i + 4];
                    sc5 += q[i + 5 + q_offset] * local_k[i + 5];
                    sc6 += q[i + 6 + q_offset] * local_k[i + 6];
                    sc7 += q[i + 7 + q_offset] * local_k[i + 7];
                }
                float score = ((sc0 + sc1) + (sc2 + sc3)) + ((sc4 + sc5) + (sc6 + sc7));
                score *= inv_sqrt_head;

                const float m_new = (score > m) ? score : m;
                float diff_m = m - m_new;
                const float exp_m = (diff_m < -80.0f) ? 0.0f : expf(diff_m);
                
    #if ONLINE_P8
                float w = 0.0f;
                float diff_p = score - m_new;
                if (diff_p > TAIL_CUT) {
                    w = (diff_p < -80.0f) ? 0.0f : expf(diff_p);
                }
                const float Pf = 255.0f * w;
                const float l_new = curr_l * exp_m + Pf;
                
            update_y_num_p8:
                for (int i = 0; i < head_size; i++) {
                    #pragma HLS pipeline II=1
                    #pragma HLS DEPENDENCE variable=y_num type=inter false
                    #pragma HLS loop_tripcount min=32 max=128 avg=64
                    y_num[i] = y_num[i] * exp_m + Pf * local_v[i];
                }
    #else
                float diff_p = score - m_new;
                const float p = (diff_p < -80.0f) ? 0.0f : expf(diff_p);
                const float l_new = curr_l * exp_m + p;
                
            update_y_num_standard:
                for (int i = 0; i < head_size; i++) {
                    #pragma HLS pipeline II=1
                    #pragma HLS DEPENDENCE variable=y_num type=inter false
                    #pragma HLS loop_tripcount min=32 max=128 avg=64
                    y_num[i] = y_num[i] * exp_m + p * local_v[i];
                }
    #endif
                curr_l = l_new;  m = m_new;
            }
            
            const float inv_l = (curr_l > 0.f) ? (1.0f / curr_l) : 0.0f;
        normalize_output:
            for (int i = 0; i < head_size; i++) {
                #pragma HLS pipeline II=1
                #pragma HLS loop_tripcount min=32 max=128 avg=64
                xb[i + xb_offset] = y_num[i] * inv_l;
            }
        }

        quantize(&xq, xb, GS);
        matmul<dim, dim>(xb2, xq.q, xq.s, (w->wo + l)->q, (w->wo + l)->s);

    residual:
        for (int i = 0; i < dim; i++) {
            #pragma HLS pipeline II=1
            #pragma HLS loop_tripcount min=256 max=2048 avg=768
            x[i] += xb2[i];
        }

        rmsnorm<dim>(xb, x, w->rms_ffn_weight + l * dim);
        quantize(&xq, xb, GS);
        matmul<dim, hidden_dim>(hb, xq.q, xq.s, (w->w1 + l)->q, (w->w1 + l)->s);
        matmul<dim, hidden_dim>(hb2, xq.q, xq.s, (w->w3 + l)->q, (w->w3 + l)->s);

        float hb_out[hidden_dim];

    swi_glu:
        for (int i = 0; i < hidden_dim; i++) {
            #pragma HLS pipeline II=6
            #pragma HLS loop_tripcount min=512 max=8192 avg=2048
            
            float val = hb[i];
            float silu_out = 0.0f;
            if (val > 20.0f) {
                silu_out = val;
            } else if (val < -20.0f) {
                silu_out = 0.0f;
            } else {
                silu_out = val / (1.0f + expf(-val));
            }
            hb_out[i] = silu_out * hb2[i];
        }

        std::memcpy(hb, hb_out, hidden_dim * sizeof(float));
        quantize(&hq, hb, GS);
        matmul<hidden_dim, dim>(xb, hq.q, hq.s, (w->w2 + l)->q, (w->w2 + l)->s);

    residual2:
        for (int i = 0; i < dim; i++) {
            #pragma HLS pipeline II=1
            #pragma HLS loop_tripcount min=256 max=2048 avg=768
            x[i] += xb[i];
        }
    }

    rmsnorm<dim>(x, x, w->rms_final_weight);
    quantize(&xq, x, GS);
    matmul<dim, vocab_size>(out, xq.q, xq.s, w->wcls->q, w->wcls->s);
}