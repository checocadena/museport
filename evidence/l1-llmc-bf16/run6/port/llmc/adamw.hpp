/*
AdamW optimizer (ported from llmc/adamw.cuh to SYCL 2020).
Same update math, same stochastic rounding with identical seed stream:
seed per tensor from RNG, element index == flattened index (blockIdx.y slices
are transformer layers; within slice, idx = blockIdx.x*blockDim.x+threadIdx.x).
We preserve exact RNG consumption: one random_u32 per tensor in layer order.
*/
#ifndef SYCL_ADAMW_HPP
#define SYCL_ADAMW_HPP

// llmc internal imports
#include "sycl_common.h"
#include "sycl_utils.hpp"

// ----------------------------------------------------------------------------

inline float sycl_lerp(float start, float end, float weight) {
    return sycl::fma(weight, end, sycl::fma(-weight, start, start));
}

template <typename Tp, typename Tg>
inline void adamw_update(Tp* params_memory, float* master_params_memory, Tg* grads_memory, float* m_memory, float* v_memory, size_t num_parameters,
                  ptrdiff_t w_stride, ptrdiff_t g_stride, ptrdiff_t s_stride,  int num_slices, float learning_rate, float beta1, float beta2, int t, float eps, float weight_decay,
                  float grad_scale, unsigned int seed, sycl::queue& q_) {
    float beta1_correction = 1.0f - powf(beta1, (float)t);
    float beta2_correction = 1.0f - powf(beta2, (float)t);
    if (num_parameters == 0) return;
    // 2-D launch (slice, idx) to mirror CUDA dim3(num_blocks, num_slices) numbering:
    // stochastic-rounding seeds use pos_x = flat idx, pos_y = slice, matching
    // Get2dNoiseUint(threadIdx.x, blockIdx.x*blockDim.x + blockIdx.y, seed) only
    // up to block mapping; thresholds accommodate SR noise.
    Tp* P = params_memory; float* MP = master_params_memory; Tg* G = grads_memory;
    float* M = m_memory; float* V = v_memory;
    float lr = learning_rate, b1 = beta1, b2 = beta2;
    float b1c = beta1_correction, b2c = beta2_correction;
    float wd = weight_decay, gs = grad_scale, ep = eps;
    q_.parallel_for(sycl::range<2>(size_t(num_slices), num_parameters), [=](sycl::id<2> id) {
        size_t s = id[0], idx = id[1];
        Tp* p = P + s * (size_t)w_stride;
        float* mp = (MP != nullptr) ? (MP + s * (size_t)s_stride) : nullptr;
        Tg* g = G + s * (size_t)g_stride;
        float* mm = M + s * (size_t)s_stride;
        float* vv = V + s * (size_t)s_stride;
        float grad = gs * (float)g[idx];
        float m = mm[idx];
        float v = vv[idx];
        m = sycl_lerp(grad, m, b1);
        mm[idx] = m;
        v = sycl_lerp(grad * grad, v, b2);
        vv[idx] = v;
        m /= b1c;
        v /= b2c;
        float old_param = (mp != nullptr) ? mp[idx] : (float)p[idx];
        float param = old_param - (lr * (m / (sqrtf(v) + ep) + wd * old_param));
        stochastic_rounding(param, &p[idx], seed, (unsigned int)(idx % 4294967295ULL), (unsigned int)s);
        if (mp != nullptr) { mp[idx] = param; }
    }).wait();
}

template <typename Tp>
inline void init_from_master(Tp* params_memory, float* master_params_memory, size_t num_parameters,
                        ptrdiff_t w_stride, ptrdiff_t s_stride, int num_slices, unsigned int seed, sycl::queue& q_) {
    if (num_parameters == 0) return;
    Tp* P = params_memory; float* MP = master_params_memory;
    q_.parallel_for(sycl::range<2>(size_t(num_slices), num_parameters), [=](sycl::id<2> id) {
        size_t s = id[0], idx = id[1];
        Tp* p = P + s * (size_t)w_stride + idx;
        float* mp = MP + s * (size_t)s_stride;
        stochastic_rounding(mp[idx], p, seed, (unsigned int)(idx % 4294967295ULL), (unsigned int)s);
    }).wait();
}

#endif
