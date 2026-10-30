/*
LayerNorm SYCL kernels, and also Residual, because sometimes they are fused.
SYCL 2020 port of llmc/layernorm.cuh. Single device, no NCCL.

Semantics match llm.c: layernorm_backward does += into dinp (residual-stream
accumulation) and += into dweight/dbias, so gradient accumulation works.
*/
#ifndef LAYERNORM_H
#define LAYERNORM_H

#include <assert.h>
// llmc internal imports
#include "sycl_common.h"
#include "sycl_utils.h"

// ----------------------------------------------------------------------------
// SYCL kernels

// out[row*C:(row+1)*C] = w[c]*((x-m)*s) + b[c]; mean/rstd optionally saved.
// One work-group per row; sub-group reductions replace warp reductions.
inline void layernorm_forward_kernel(floatX* out, float* mean, float* rstd,
                                     const floatX* inp, const floatX* weight,
                                     const floatX* bias, size_t N, size_t C) {
    if (N == 0) { return; }
    constexpr size_t WG = 256;
    sycl_queue.parallel_for(sycl::nd_range<1>(sycl::range<1>(N * WG), sycl::range<1>(WG)),
        [=](sycl::nd_item<1> it) {
            size_t row = it.get_group_linear_id();
            auto grp = it.get_group();
            size_t lid = it.get_local_linear_id();
            size_t lsz = it.get_local_range(0);
            const floatX* x = inp + row * C;
            float sum = 0.0f;
            for (size_t i = lid; i < C; i += lsz) { sum += (float)x[i]; }
            sum = group_sum(grp, sum);
            // broadcast mean via group broadcast from lane 0 equivalent:
            float m = sycl::group_broadcast(grp, sum, 0) / (float)C;
            float var = 0.0f;
            for (size_t i = lid; i < C; i += lsz) {
                float d = (float)x[i] - m;
                var += d * d;
            }
            var = group_sum(grp, var);
            float s = sycl::rsqrt(sycl::group_broadcast(grp, var, 0) / (float)C + 1e-5f);
            if (lid == 0) {
                if (mean != nullptr) { mean[row] = m; }
                if (rstd != nullptr) { rstd[row] = s; }
            }
            floatX* o = out + row * C;
            for (size_t c = lid; c < C; c += lsz) {
                float nrm = s * ((float)x[c] - m);
                o[c] = (floatX)(nrm * (float)weight[c] + (float)bias[c]);
            }
        }).wait();
}

inline void residual_forward_kernel(floatX* out, const floatX* inp1, const floatX* inp2, size_t N) {
    if (N == 0) { return; }
    constexpr size_t BLOCK = 512;
    sycl_queue.parallel_for(sycl::nd_range<1>(sycl::range<1>(CEIL_DIV(N, BLOCK) * BLOCK),
                                              sycl::range<1>(BLOCK)),
        [=](sycl::nd_item<1> it) {
            size_t idx = it.get_global_linear_id();
            if (idx >= N) { return; }
            out[idx] = (floatX)((float)inp1[idx] + (float)inp2[idx]);
        }).wait();
}

inline void fused_residual_forward_kernel(floatX* residual, floatX* normed, float* mean, float* rstd,
                                          const floatX* inp1, const floatX* inp2,
                                          const floatX* weight, const floatX* bias,
                                          size_t N, size_t C) {
    if (N == 0) { return; }
    constexpr size_t WG = 256;
    sycl_queue.parallel_for(sycl::nd_range<1>(sycl::range<1>(N * WG), sycl::range<1>(WG)),
        [=](sycl::nd_item<1> it) {
            size_t row = it.get_group_linear_id();
            auto grp = it.get_group();
            size_t lid = it.get_local_linear_id();
            size_t lsz = it.get_local_range(0);
            const floatX* a = inp1 + row * C;
            const floatX* b = inp2 + row * C;
            floatX* r = residual + row * C;
            // residual = a + b (each element written once by strided lanes)
            for (size_t c = lid; c < C; c += lsz) {
                r[c] = (floatX)((float)a[c] + (float)b[c]);
            }
            it.barrier(sycl::access::fence_space::global_space);
            float sum = 0.0f;
            for (size_t i = lid; i < C; i += lsz) { sum += (float)r[i]; }
            sum = group_sum(grp, sum);
            float m = sycl::group_broadcast(grp, sum, 0) / (float)C;
            float var = 0.0f;
            for (size_t i = lid; i < C; i += lsz) {
                float d = (float)r[i] - m;
                var += d * d;
            }
            var = group_sum(grp, var);
            float s = sycl::rsqrt(sycl::group_broadcast(grp, var, 0) / (float)C + 1e-5f);
            if (lid == 0) {
                mean[row] = m;
                rstd[row] = s;
            }
            floatX* o = normed + row * C;
            for (size_t c = lid; c < C; c += lsz) {
                float nrm = s * ((float)r[c] - m);
                o[c] = (floatX)(nrm * (float)weight[c] + (float)bias[c]);
            }
        }).wait();
}

// Layernorm backward. Deterministic single-pass version with the same maths as
// llm.c's layernorm_backward_kernel10:
//   dinp += ... (accumulates into residual stream gradient)
//   dweight/dbias += per-channel sums (read-modify-write)
// Parallelism: one work-group per row for dinp + fp32 partial sums for
// dweight/dbias staged in a device buffer of (num_groups x C) floats, then a
// deterministic串行 host accumulation into dweight/dbias. num_groups is small
// (one group per row, capped) so this is simple and exactly deterministic.
inline void layernorm_backward_kernel(floatX* dinp, floatX* dweight, floatX* dbias, float* scratch,
                                      const floatX* dout, const floatX* inp, const floatX* weight,
                                      const float* mean, const float* rstd,
                                      size_t B, size_t T, size_t C) {
    size_t N = B * T;
    if (N == 0 || C == 0) { return; }
    constexpr size_t WG = 128;
    // partial sums: one fp32 row per input row (N x C may be big: N=256, C=768 -> 786KB, fine)
    float* part_db = scratch;             // N*C floats
    float* part_dw = scratch + N * C;     // N*C floats
    sycl_queue.parallel_for(sycl::nd_range<1>(sycl::range<1>(N * WG), sycl::range<1>(WG)),
        [=](sycl::nd_item<1> it) {
            size_t row = it.get_group_linear_id();
            auto grp = it.get_group();
            size_t lid = it.get_local_linear_id();
            size_t lsz = it.get_local_range(0);
            const floatX* dout_bt = dout + row * C;
            const floatX* inp_bt = inp + row * C;
            floatX* dinp_bt = dinp + row * C;
            float mean_bt = mean[row];
            float rstd_bt = rstd[row];
            // two reductions: sum(dnorm), sum(dnorm*norm)
            float dnorm_mean = 0.0f, dnorm_norm_mean = 0.0f;
            for (size_t i = lid; i < C; i += lsz) {
                float dnorm_i = (float)weight[i] * (float)dout_bt[i];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * (float)inp_bt[i];
            }
            dnorm_mean = group_sum(grp, dnorm_mean);
            dnorm_norm_mean = group_sum(grp, dnorm_norm_mean);
            dnorm_mean = sycl::group_broadcast(grp, dnorm_mean, 0) / (float)C;
            dnorm_norm_mean = sycl::group_broadcast(grp, dnorm_norm_mean, 0) / (float)C * rstd_bt
                              - dnorm_mean * mean_bt * rstd_bt;
            // need broadcast values in all lanes
            float dm = sycl::group_broadcast(grp, dnorm_mean, 0);
            float dnm = sycl::group_broadcast(grp, dnorm_norm_mean, 0);
            float* pdb = part_db + row * C;
            float* pdw = part_dw + row * C;
            for (size_t c = lid; c < C; c += lsz) {
                float dout_i = (float)dout_bt[c];
                float norm = ((float)inp_bt[c] - mean_bt) * rstd_bt;
                pdb[c] = dout_i;
                pdw[c] = norm * dout_i;
                float dval = (float)weight[c] * dout_i - dm - norm * dnm;
                dval *= rstd_bt;
                dinp_bt[c] = (floatX)((float)dinp_bt[c] + dval);
            }
        }).wait();
    // deterministic accumulation over rows into dweight/dbias (host-side, fp32 order fixed)
    std::vector<float> pdb(N * C), pdw(N * C);
    sycl_memcpy_d2h(pdb.data(), part_db, N * C * sizeof(float));
    sycl_memcpy_d2h(pdw.data(), part_dw, N * C * sizeof(float));
    std::vector<floatX> hdw(C), hdb(C);
    sycl_memcpy_d2h(hdw.data(), dweight, C * sizeof(floatX));
    sycl_memcpy_d2h(hdb.data(), dbias, C * sizeof(floatX));
    for (size_t c = 0; c < C; c++) {
        float sb = 0.0f, sw = 0.0f;
        for (size_t r = 0; r < N; r++) {
            sb += pdb[r * C + c];
            sw += pdw[r * C + c];
        }
        hdb[c] = (floatX)(sb + (float)hdb[c]);
        hdw[c] = (floatX)(sw + (float)hdw[c]);
    }
    sycl_memcpy_h2d(dweight, hdw.data(), C * sizeof(floatX));
    sycl_memcpy_h2d(dbias, hdb.data(), C * sizeof(floatX));
}

// ----------------------------------------------------------------------------
// kernel launchers

inline void layernorm_forward(floatX* out, float* mean, float* rstd,
                              floatX* inp, const floatX* weight, const floatX* bias,
                              size_t B, size_t T, size_t C) {
    NVTX_RANGE_FN();
    layernorm_forward_kernel(out, mean, rstd, inp, weight, bias, B * T, C);
}

inline void residual_forward(floatX* out, const floatX* inp1, const floatX* inp2, size_t N) {
    NVTX_RANGE_FN();
    residual_forward_kernel(out, inp1, inp2, N);
}

inline void fused_residual_forward5(floatX* residual, floatX* normed, float* mean, float* rstd,
                                    const floatX* inp1, const floatX* inp2,
                                    const floatX* weight, const floatX* bias,
                                    size_t N, size_t C) {
    fused_residual_forward_kernel(residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C);
}

inline void layernorm_backward(floatX* dinp, floatX* dweight, floatX* dbias, float* scratch,
                               const floatX* dout, const floatX* inp, const floatX* weight,
                               const float* mean, const float* rstd,
                               size_t B, size_t T, size_t C) {
    NVTX_RANGE_FN();
    layernorm_backward_kernel(dinp, dweight, dbias, scratch, dout, inp, weight, mean, rstd, B, T, C);
}

#endif
