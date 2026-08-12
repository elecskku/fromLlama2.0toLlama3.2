#pragma once
#include <stdint.h>
#include <ap_int.h>
#include <hls_math.h>
#include "config.h"

// W(D×N) · x(N×1) -> y(D×1)
// xq: [N], xs: [N/GS], wq: [D*N], ws: [D*(N/GS)]
//
// [최적화 노트]
//   기존 버전은 GroupLoop에 #pragma HLS PIPELINE II=GS(=64) 가 박혀 있었다.
//   원인: KLoop(UNROLL)이 한 그룹의 가중치 64개를 동시에 곱하려 하는데,
//         가중치를 int8 한 개씩(8비트) 읽어 메모리 포트(최대 128비트)를 못 채워
//         64개를 모으는 데 64사이클이 걸렸기 때문 → 곱셈기가 63/64 사이클을 놂.
//   개선: 가중치를 128비트(=int8 16개) 단위로 읽어 포트를 꽉 채운다.
//         한 그룹(64개) = 128비트 워드 4개. 워드당 16-MAC을 펼치고,
//         GroupLoop를 II=4 로 파이프라인하여 사이클당 128비트 1워드를 흘린다.
//         (이론상 한 그룹 64사이클 → 4사이클, 약 16배)
//   주의: float 누산(val)의 loop-carried 의존이 작은 II를 막으므로,
//         float 부분합을 NACC개로 분산(회전)시켜 의존성을 끊고 마지막에 합친다.
//         정수 MAC(acc)은 원본과 비트 동일하며, float 합산 순서만 달라진다.
//   전제: wq 가 16바이트 정렬일 것. 본 모델 구조에서 wq[l].q, 행 오프셋(i*N),
//         그룹 오프셋(g*GS)이 모두 16의 배수임을 확인함(N,GS,hidden 모두 16배수).
template <int N, int D>
void matmul(float * __restrict xout,
            const int8_t * __restrict xq,
            const float  * __restrict xs,
            const int8_t * __restrict wq,
            const float  * __restrict ws)
{
    constexpr int GROUPS    = N / GS;     // 그룹 수
    constexpr int LANES     = 16;         // 128비트 = int8 16개
    constexpr int WPG       = GS / LANES; // 그룹당 128비트 워드 수 (64/16 = 4)
    constexpr int WIDE_ROW  = N / LANES;  // 한 행의 128비트 워드 수 (N/16)
    constexpr int NACC      = 4;          // float 부분합 누산기 개수(의존성 분산)

    // 가중치를 128비트 워드로 재해석 (한 번에 int8 16개 읽기)
    const ap_uint<128>* wq128 = reinterpret_cast<const ap_uint<128>*>(wq);

    // x / xs 프리로드 - 모든 행에서 재사용되므로 온칩화
    int8_t xbuf[N];
    #pragma HLS ARRAY_PARTITION variable=xbuf cyclic factor=GS dim=1
    float xsbuf[GROUPS];
    #pragma HLS ARRAY_PARTITION variable=xsbuf complete

load_x:
    for (int j = 0; j < N; j++) {
        #pragma HLS PIPELINE II=1
        xbuf[j] = xq[j];
    }
load_xs:
    for (int g = 0; g < GROUPS; g++) {
        #pragma HLS PIPELINE II=1
        xsbuf[g] = xs[g];
    }

RowLoop:
    for (int i = 0; i < D; i++) {
        #pragma HLS loop_tripcount min=256 max=128256 avg=2048

        // float 부분합 누산기 (loop-carried 의존을 NACC개로 분산)
        float vacc[NACC];
        #pragma HLS ARRAY_PARTITION variable=vacc complete
    init_acc:
        for (int a = 0; a < NACC; a++) {
            #pragma HLS UNROLL
            vacc[a] = 0.f;
        }

        const int row_word_base = i * WIDE_ROW;   // 이 행의 첫 128비트 워드 인덱스

    GroupLoop:
        for (int g = 0; g < GROUPS; g++) {
            #pragma HLS PIPELINE II=4
            #pragma HLS loop_tripcount min=4 max=128 avg=32

            int32_t acc = 0;

        WideLoop:
            for (int w = 0; w < WPG; w++) {          // 그룹당 128비트 워드 4개
                #pragma HLS UNROLL
                ap_uint<128> ww = wq128[row_word_base + g * WPG + w];
                const int xbase = g * GS + w * LANES;

            LaneLoop:
                for (int lane = 0; lane < LANES; lane++) {  // 워드당 int8 16개
                    #pragma HLS UNROLL
                    ap_int<8> wv = (ap_int<8>)ww.range(lane * 8 + 7, lane * 8);
                    acc += (int32_t)xbuf[xbase + lane] * (int32_t)wv;
                }
            }

            // 그룹 스케일 적용 후 회전 누산기에 더함 (g&3 : 4개로 분산)
            vacc[g & (NACC - 1)] += (float)acc * ws[i * GROUPS + g] * xsbuf[g];
        }

        // 부분합 합치기 (행당 NACC-1 번의 덧셈, 무시 가능)
        float val = 0.f;
    reduce_acc:
        for (int a = 0; a < NACC; a++) {
            #pragma HLS UNROLL
            val += vacc[a];
        }

        xout[i] = val;
    }
}
