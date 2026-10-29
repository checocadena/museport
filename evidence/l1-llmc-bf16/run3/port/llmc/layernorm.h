/*
LayerNorm + Residual kernels (SYCL port of layernorm.cuh).
Forward matches llm.c numerics (fp32 accumulation, 1e-5 eps).
Backward: deterministic; dweight/dbias accumulated with a two-stage
deterministic reduction; dinp does read-modify-write (+=) as in llm.c
(because residual stream gradients must accumulate).
*/
#ifndef LAYERNORM_SYCL_H
#define LAYERNORM_SYCL_H

#include <assert.h>
#include <sycl/sycl.hpp>
// llmc internal imports
#include "sycl_common.h"
#include "sycl_utils.h"

// ----------------------------------------------------------------------------
// forward kernels (1-D range, one logical row-group per row)

inline void layernorm_forward(floatX* out, float* mean, float* rstd,
                              const floatX* inp, const floatX* weight, const floatX* bias,
                              int B, int T, int C, sycl::queue& q) {
    NVTX_RANGE_FN();
    size_t N = (size_t)B * T;
    // one work-group of 32 threads per row gives good occupancy and matches sub-group math
    constexpr size_t WG = 32;
    q.parallel_for(sycl::nd_range<1>(sycl::range<1>(N * WG), sycl::range<1>(WG)),
        [=](sycl::nd_item<1> item) {
        size_t row = item.get_group(0);
        if (row >= N) { return; }
        size_t lid = item.get_local_id(0);
        auto grp = item.get_group();
        const floatX* x = inp + row * (size_t)C;
        // mean (work-group-wide reduce: correct for any sub-group size)
        float sum = 0.f;
        for (int i = (int)lid; i < C; i += (int)WG) { sum += (float)x[i]; }
        sum = sycl::reduce_over_group(grp, sum, sycl::plus<float>());
        float m = sum / (float)C;
        if (lid == 0 && mean != nullptr) { mean[row] = m; }
        // rstd
        float sq = 0.f;
        for (int i = (int)lid; i < C; i += (int)WG) {
            float d = (float)x[i] - m;
            sq += d * d;
        }
        sq = sycl::reduce_over_group(grp, sq, sycl::plus<float>());
        float s = sycl::rsqrt(sq / (float)C + 1e-5f);
        if (lid == 0 && rstd != nullptr) { rstd[row] = s; }
        floatX* o = out + row * (size_t)C;
        for (int c = (int)lid; c < C; c += (int)WG) {
            float nrm = s * ((float)x[c] - m);
            o[c] = (floatX)(nrm * (float)weight[c] + (float)bias[c]);
        }
    }).wait();
}

inline void residual_forward(floatX* out, const floatX* inp1, const floatX* inp2, size_t N, sycl::queue& q) {
    NVTX_RANGE_FN();
    constexpr size_t BLOCK = 256;
    size_t nblocks = CEIL_DIV(N, BLOCK);
    q.parallel_for(sycl::range<1>(nblocks * BLOCK), [=](sycl::id<1> id) {
        size_t idx = id[0];
        if (idx >= N) { return; }
        out[idx] = (floatX)((float)inp1[idx] + (float)inp2[idx]);
    }).wait();
}

inline void fused_residual_forward5(floatX* residual, floatX* normed, float* mean, float* rstd,
                                    const floatX* inp1, const floatX* inp2,
                                    const floatX* weight, const floatX* bias,
                                    int N, int C, sycl::queue& q) {
    NVTX_RANGE_FN();
    // residual = inp1 + inp2
    residual_forward(residual, inp1, inp2, (size_t)N * C, q);
    // normed = layernorm(residual)
    layernorm_forward(normed, mean, rstd, residual, weight, bias, N, 1, C, q);
}

// ----------------------------------------------------------------------------
// backward: deterministic, matches llm.c semantics
//   dinp (BT x C): dinp += f(dout, inp, weight, mean, rstd)
//   dweight/dbias (C): dbias += sum(dout); dweight += sum(norm * dout)

inline void layernorm_backward(floatX* dinp, floatX* dweight, floatX* dbias, float* scratch,
                               const floatX* dout, const floatX* inp, const floatX* weight,
                               const float* mean, const float* rstd,
                               int B, int T, int C, sycl::queue& q) {
    NVTX_RANGE_FN();
    size_t BT = (size_t)B * T;
    // Deterministic two-stage reduction with per-row private partials
    // (no cross-row races: row bt exclusively owns part[bt*C + c]).
    float* part_bias = sycl::malloc_device<float>(BT * (size_t)C, q);
    float* part_weight = sycl::malloc_device<float>(BT * (size_t)C, q);
    q.memset(part_bias, 0, BT * (size_t)C * sizeof(float)).wait();
    q.memset(part_weight, 0, BT * (size_t)C * sizeof(float)).wait();

    // dinp update + partial stat accumulation, one group of 32 per row
    constexpr size_t WG = 32;
    q.parallel_for(sycl::nd_range<1>(sycl::range<1>(BT * WG), sycl::range<1>(WG)),
        [=](sycl::nd_item<1> item) {
        size_t bt = item.get_group(0);
        if (bt >= BT) { return; }
        size_t lid = item.get_local_id(0);
        auto grp = item.get_group();
        const floatX* dout_bt = dout + bt * (size_t)C;
        const floatX* inp_bt = inp + bt * (size_t)C;
        floatX* dinp_bt = dinp + bt * (size_t)C;
        float mean_bt = mean[bt];
        float rstd_bt = rstd[bt];
        // two reductions for dnorm mean terms
        float dm = 0.f, dnm = 0.f;
        for (int i = (int)lid; i < C; i += (int)WG) {
            float dnorm_i = (float)weight[i] * (float)dout_bt[i];
            dm += dnorm_i;
            dnm += dnorm_i * (float)inp_bt[i];
        }
        dm = sycl::reduce_over_group(grp, dm, sycl::plus<float>());
        dnm = sycl::reduce_over_group(grp, dnm, sycl::plus<float>());
        dm /= (float)C;
        dnm = dnm / (float)C * rstd_bt - dm * mean_bt * rstd_bt;
        // dinp (+=) and per-row partial sums for bias/weight (race-free: unique slots)
        for (int c = (int)lid; c < C; c += (int)WG) {
            float dout_i = (float)dout_bt[c];
            float norm_bti = ((float)inp_bt[c] - mean_bt) * rstd_bt;
            float dval = (float)weight[c] * dout_i - dm - norm_bti * dnm;
            dval *= rstd_bt;
            dinp_bt[c] = (floatX)((float)dinp_bt[c] + dval);
            part_bias[bt * (size_t)C + c] = dout_i;
            part_weight[bt * (size_t)C + c] = norm_bti * dout_i;
        }
    }).wait();
    // Stage 2: deterministic sum over rows (ascending bt) into dweight/dbias (+=)
    {
        constexpr size_t BLOCK = 256;
        size_t nblocks = CEIL_DIV((size_t)C, BLOCK);
        q.parallel_for(sycl::range<1>(nblocks * BLOCK), [=](sycl::id<1> id) {
            size_t c = id[0];
            if (c >= (size_t)C) { return; }
            float sb = 0.f, sw = 0.f;
            for (size_t r = 0; r < BT; r++) {
                sb += part_bias[r * (size_t)C + c];
                sw += part_weight[r * (size_t)C + c];
            }
            dbias[c] = (floatX)((float)dbias[c] + sb);
            dweight[c] = (floatX)((float)dweight[c] + sw);
        }).wait();
    }
    sycl::free(part_bias, q);
    sycl::free(part_weight, q);
    (void)scratch;
}

#endif
