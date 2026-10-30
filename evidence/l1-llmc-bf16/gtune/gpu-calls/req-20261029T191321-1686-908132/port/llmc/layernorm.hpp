/*
LayerNorm SYCL kernels, and also Residual, because sometimes they are fused
Port of layernorm.cuh. Reductions use SYCL group algorithms
(sycl::reduce_over_group / joint_reduce) instead of CUB/warp shuffles.
All launches are fire-and-forget on the in-order queue. Reductions that need
to be deterministic across devices accumulate in a fixed order:
- dweight/dbias accumulate via fixed-grid ordered partial sums in device
  memory (partials kernel) followed by an ordered accumulation kernel.
*/
#ifndef SYCL_LAYERNORM_H
#define SYCL_LAYERNORM_H

#include <assert.h>
// llmc internal imports
#include "sycl_common.h"
#include "sycl_utils.hpp"

// ----------------------------------------------------------------------------
// SYCL kernels

// layernorm forward: one work-group per row (N = B*T rows, C columns).
// Local memory holds the row tile for the variance pass.
inline void layernorm_forward(floatX* out, float* mean, float* rstd,
                       floatX* inp, const floatX* weight, const floatX* bias,
                       int B, int T, int C, sycl::queue& q) {
    NVTX_RANGE_FN();
    size_t N = (size_t)B * T;
    if (N == 0) return;
    // work-group of 256 items; each group handles one row
    size_t wg = 256;
    q.parallel_for(sycl::nd_range<1>(N * wg, wg), [=](sycl::nd_item<1> it) {
        size_t row = it.get_group(0);
        size_t lid = it.get_local_id(0);
        sycl::group<1> g = it.get_group();
        const floatX* x = inp + row * (size_t)C;
        // mean
        float sum = 0.0f;
        for (size_t c = lid; c < (size_t)C; c += wg) { sum += (float)x[c]; }
        float total = sycl::reduce_over_group(g, sum, sycl::plus<float>());
        float m = total / (float)C;
        // variance
        float vsum = 0.0f;
        for (size_t c = lid; c < (size_t)C; c += wg) {
            float d = (float)x[c] - m;
            vsum += d * d;
        }
        float vtotal = sycl::reduce_over_group(g, vsum, sycl::plus<float>());
        float s = sycl::rsqrt(vtotal / (float)C + 1e-5f);
        if (lid == 0) {
            if (mean != nullptr) { mean[row] = m; }
            if (rstd != nullptr) { rstd[row] = s; }
        }
        floatX* o = out + row * (size_t)C;
        for (size_t c = lid; c < (size_t)C; c += wg) {
            float nrm = s * ((float)x[c] - m);
            o[c] = (floatX)(nrm * (float)weight[c] + (float)bias[c]);
        }
    });
}

inline void residual_forward(floatX* out, const floatX* inp1, const floatX* inp2, size_t N, sycl::queue& q) {
    NVTX_RANGE_FN();
    size_t vec = x128::size;
    size_t nvec = CEIL_DIV(N, vec);
    q.parallel_for(sycl::range<1>(nvec), [=](sycl::id<1> id) {
        size_t idx = id[0] * vec;
        if (idx >= N) return;
        size_t rem = N - idx;
        if (rem >= vec) {
            x128 packed_out;
            x128 packed_inp1 = load128cs(inp1 + idx);
            x128 packed_inp2 = load128cs(inp2 + idx);
            for (size_t k = 0; k < vec; k++) {
                packed_out[k] = (floatX)((float)packed_inp1[k] + (float)packed_inp2[k]);
            }
            store128(out + idx, packed_out);
        } else {
            for (size_t k = 0; k < rem; k++) {
                out[idx + k] = (floatX)((float)inp1[idx + k] + (float)inp2[idx + k]);
            }
        }
    });
}

inline void fused_residual_forward5(floatX* residual, floatX* normed, float* mean, float* rstd,
                             const floatX* inp1, const floatX* inp2,
                             const floatX* weight, const floatX* bias,
                             int N, int C, sycl::queue& q) {
    // residual = inp1 + inp2, then layernorm(residual) -> normed.
    // Two-pass version (same numerics as the CUDA fallback path).
    residual_forward(residual, inp1, inp2, (size_t)N * C, q);
    layernorm_forward(normed, mean, rstd, residual, weight, bias, N, 1, C, q);
}

// layernorm backward: deterministic.
// CUDA kernel uses cross-block atomics + last-block reduction; here:
// - per-row terms (dnorm_mean, dnorm_norm_mean) via group reductions,
// - dinp row update (one group per row; += semantics preserved),
// - dweight/dbias via fixed-grid ordered partial sums in device memory
//   (`scratch` must hold 256*C floats) + an ordered accumulation pass.
// Pass boundaries depend only on shapes, so results are bit-deterministic.
inline void layernorm_backward(floatX* dinp, floatX* dweight, floatX* dbias, float* scratch,
                        const floatX* dout, const floatX* inp, const floatX* weight, const float* mean, const float* rstd,
                        int B, int T, int C, sycl::queue& q) {
    NVTX_RANGE_FN();
    size_t N = (size_t)B * T;
    if (N == 0) return;
    // Pass 1: per-row dnorm_mean / dnorm_norm_mean into device scratch head.
    float* row_sums = scratch; // N*2 floats
    {
        size_t wg = 256;
        q.parallel_for(sycl::nd_range<1>(N * wg, wg), [=](sycl::nd_item<1> it) {
            size_t row = it.get_group(0);
            size_t lid = it.get_local_id(0);
            sycl::group<1> g = it.get_group();
            const floatX* dout_bt = dout + row * (size_t)C;
            const floatX* inp_bt = inp + row * (size_t)C;
            float dm = 0.0f, dnm = 0.0f;
            for (size_t c = lid; c < (size_t)C; c += wg) {
                float dnorm_i = (float)weight[c] * (float)dout_bt[c];
                dm += dnorm_i;
                dnm += dnorm_i * (float)inp_bt[c];
            }
            dm = sycl::reduce_over_group(g, dm, sycl::plus<float>());
            dnm = sycl::reduce_over_group(g, dnm, sycl::plus<float>());
            if (lid == 0) {
                float mbt = mean[row];
                float rbt = rstd[row];
                row_sums[row*2+0] = dm / (float)C;
                row_sums[row*2+1] = dnm / (float)C * rbt - (dm / (float)C) * mbt * rbt;
            }
        });
    }
    // Pass 2: dinp update (one group per row; += semantics preserved)
    {
        size_t wg = 256;
        q.parallel_for(sycl::nd_range<1>(N * wg, wg), [=](sycl::nd_item<1> it) {
            size_t row = it.get_group(0);
            size_t lid = it.get_local_id(0);
            float dnorm_mean = row_sums[row*2+0];
            float dnorm_norm_mean = row_sums[row*2+1];
            float mbt = mean[row];
            float rbt = rstd[row];
            const floatX* dout_bt = dout + row * (size_t)C;
            const floatX* inp_bt = inp + row * (size_t)C;
            floatX* dinp_bt = dinp + row * (size_t)C;
            for (size_t c = lid; c < (size_t)C; c += wg) {
                float norm_bti = ((float)inp_bt[c] - mbt) * rbt;
                float dval = (float)weight[c] * (float)dout_bt[c]; // term 1
                dval -= dnorm_mean; // term 2
                dval -= norm_bti * dnorm_norm_mean; // term 3
                dval *= rbt; // final scale
                dinp_bt[c] = (floatX)((float)dinp_bt[c] + dval);
            }
        });
    }
    // Pass 3: deterministic dweight/dbias accumulation via fixed-grid ordered
    // partial sums in device memory. `scratch` layout: [row_sums N*2][dw P*C][db P*C].
    constexpr int P = 256;
    int p = (N < (size_t)P) ? (int)N : P;
    size_t chunk = (N + (size_t)p - 1) / (size_t)p;
    float* part_w = scratch + N * 2;
    float* part_b = part_w + (size_t)p * C;
    q.parallel_for(sycl::range<2>((size_t)p, (size_t)C), [=](sycl::id<2> id) {
        size_t r = id[0];
        size_t c = id[1];
        size_t b = r * chunk;
        size_t e = b + chunk;
        if (e > N) { e = N; }
        float aw = 0.0f, ab = 0.0f;
        for (size_t row = b; row < e; row++) {
            float dout_i = (float)dout[row * (size_t)C + c];
            float norm_bti = ((float)inp[row * (size_t)C + c] - mean[row]) * rstd[row];
            ab += dout_i;
            aw += norm_bti * dout_i;
        }
        part_b[r * (size_t)C + c] = ab;
        part_w[r * (size_t)C + c] = aw;
    });
    q.parallel_for(sycl::range<1>((size_t)C), [=](sycl::id<1> id) {
        size_t c = id[0];
        float accb = (float)dbias[c];
        float accw = (float)dweight[c];
        for (int r = 0; r < p; r++) {
            accb += part_b[(size_t)r * C + c];
            accw += part_w[(size_t)r * C + c];
        }
        dbias[c] = (floatX)accb;
        dweight[c] = (floatX)accw;
    });
}

#endif
