#pragma once
#include <stdint.h>
#include <ap_int.h>
#include <hls_math.h>
#include "config.h"

// ============================================================================
// [버전 1] baseline 원본 matmul.h
//   - 110M 모델에서 실제 보드 동작이 검증된 코드.
//   - 128비트 wide read, NACC 회전 누산기 등 최적화가 전혀 없는 가장 단순한 형태.
//   - AXI 접근 패턴이 단순해서 하드웨어에서 hang 날 여지가 가장 적다.
//   - 느리지만 "일단 도는가"를 확인하는 기준선.
//   - forward.cpp 호출부는 matmul<N,D>(xout, xq, xs, wq, ws)로 동일 → 그대로 사용.
// ============================================================================

// W(D×N) · x(N×1) -> y(D×1)
// xq: [N], xs: [N/GS], wq: [D*N], ws: [D*(N/GS)]
template <int N, int D>
void matmul(float * __restrict xout,
            const int8_t * __restrict xq,
            const float  * __restrict xs,
            const int8_t * __restrict wq,
            const float  * __restrict ws)
{
    constexpr int GROUPS = N / GS;

    // x / xs 프리로드 - 모든 행에서 재사용되므로 레지스터화
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
        #pragma HLS loop_tripcount min=256 max=4096 avg=768
        float val = 0.f;

    GroupLoop:
        for (int g = 0; g < GROUPS; g++) {
            #pragma HLS PIPELINE II=GS
            #pragma HLS loop_tripcount min=4 max=64 avg=12
            int32_t acc = 0;

        KLoop:
            for (int k = 0; k < GS; k++) {
                #pragma HLS UNROLL
                acc += (int32_t)xbuf[g * GS + k] * (int32_t)wq[i * N + g * GS + k];
            }
            val += (float)acc * ws[i * GROUPS + g] * xsbuf[g];
        }

        xout[i] = val;
    }
}
