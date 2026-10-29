/*
AdamW kernel. SYCL port of adamw.cuh.
*/
#ifndef SYCL_ADAMW_H
#define SYCL_ADAMW_H

// llmc internal imports
#include "sycl_common.h"
#include "sycl_utils.hpp"

// ----------------------------------------------------------------------------
// SYCL kernels

// Implements linear interpolation using only two floating-point operations (as opposed to three in a naive implementation).
// Reference: https://developer.nvidia.com/blog/lerp-faster-cuda
inline float lerp(float start, float end, float weight) {
    return sycl::fma(weight, end, sycl::fma(-weight, start, start));
}

template <typename Tp, typename Tg>
inline void adamw_update(Tp* params_memory, float* master_params_memory, Tg* grads_memory, float* m_memory, float* v_memory, size_t num_parameters,
                  ptrdiff_t w_stride, ptrdiff_t g_stride, ptrdiff_t s_stride,  int num_slices, float learning_rate, float beta1, float beta2, int t, float eps, float weight_decay,
                  float grad_scale, unsigned int seed, sycl::queue& q) {
    // AdamW update
    float beta1_correction = 1.0f - powf(beta1, (float)t);
    float beta2_correction = 1.0f - powf(beta2, (float)t);
    size_t total = num_parameters * (size_t)num_slices;
    if (total == 0) return;
    q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> id) {
        size_t gid = id[0];
        size_t j = gid / num_parameters; // slice (layer)
        size_t idx = gid % num_parameters;
        Tp* p = params_memory + j * (size_t)w_stride;
        float* mp = master_params_memory ? master_params_memory + j * (size_t)s_stride : nullptr;
        Tg* g = grads_memory + j * (size_t)g_stride;
        float* mm = m_memory + j * (size_t)s_stride;
        float* vm = v_memory + j * (size_t)s_stride;
        // per-element RNG identical to CUDA: seed + blockIdx.x*blockDim.x+threadIdx.x with blockDim 512
        // gid == blockIdx.x*512+threadIdx.x when num_slices folded into grid-y; replicate exactly:
        size_t lin = j * num_parameters + idx;
        unsigned int rnd = Get2dNoiseUint((int)(lin % 512), (int)(lin / 512), seed);
        // get the gradient, m, and v for this parameter
        float grad = grad_scale * (float)g[idx];
        float m = mm[idx];
        float v = vm[idx];
        // update the first moment (momentum)
        m = lerp(grad, m, beta1);
        mm[idx] = m;
        // update the second moment (RMSprop)
        v = lerp(grad * grad, v, beta2);
        vm[idx] = v;
        m /= beta1_correction;  // m_hat
        v /= beta2_correction;  // v_hat
        // fetch the old value of this parameter as a float, from either source
        float old_param = (mp != nullptr) ? mp[idx] : (float)p[idx];
        // update this parameter
        float param = old_param - (learning_rate * (m / (sqrtf(v) + eps) + weight_decay * old_param));
        // update our low precision version of the parameters using stochastic rounding
        stochastic_rounding_at(param, (sycl::ext::oneapi::bfloat16*)&p[idx], rnd);
        // write the full, float version of the param into our master copy, if we maintain one
        if (mp != nullptr) { mp[idx] = param; }
    }).wait();
}

// NOTE: overloads for fp32 params (Tp=float) need a separate path since
// stochastic_rounding_at targets bf16. Provide generic dispatch:
template <typename Tg>
inline void adamw_update_fp32(float* params_memory, float* master_params_memory, Tg* grads_memory, float* m_memory, float* v_memory, size_t num_parameters,
                  ptrdiff_t w_stride, ptrdiff_t g_stride, ptrdiff_t s_stride,  int num_slices, float learning_rate, float beta1, float beta2, int t, float eps, float weight_decay,
                  float grad_scale, unsigned int seed, sycl::queue& q) {
    float beta1_correction = 1.0f - powf(beta1, (float)t);
    float beta2_correction = 1.0f - powf(beta2, (float)t);
    size_t total = num_parameters * (size_t)num_slices;
    if (total == 0) return;
    q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> id) {
        size_t gid = id[0];
        size_t j = gid / num_parameters;
        size_t idx = gid % num_parameters;
        float* p = params_memory + j * (size_t)w_stride;
        float* mp = master_params_memory ? master_params_memory + j * (size_t)s_stride : nullptr;
        Tg* g = grads_memory + j * (size_t)g_stride;
        float* mm = m_memory + j * (size_t)s_stride;
        float* vm = v_memory + j * (size_t)s_stride;
        float grad = grad_scale * (float)g[idx];
        float m = mm[idx];
        float v = vm[idx];
        m = lerp(grad, m, beta1);
        mm[idx] = m;
        v = lerp(grad * grad, v, beta2);
        vm[idx] = v;
        m /= beta1_correction;
        v /= beta2_correction;
        float old_param = (mp != nullptr) ? mp[idx] : (float)p[idx];
        float param = old_param - (learning_rate * (m / (sqrtf(v) + eps) + weight_decay * old_param));
        p[idx] = param;
        if (mp != nullptr) { mp[idx] = param; }
    }).wait();
}

template <typename Tp>
inline void init_from_master(Tp* params_memory, float* master_params_memory, size_t num_parameters,
                        ptrdiff_t w_stride, ptrdiff_t s_stride, int num_slices, unsigned int seed, sycl::queue& q) {
    size_t total = num_parameters * (size_t)num_slices;
    if (total == 0) return;
    q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> id) {
        size_t gid = id[0];
        size_t j = gid / num_parameters;
        size_t idx = gid % num_parameters;
        Tp* p = params_memory + j * (size_t)w_stride;
        float* mp = master_params_memory + j * (size_t)s_stride;
        size_t lin = j * num_parameters + idx;
        unsigned int rnd = Get2dNoiseUint((int)(lin % 512), (int)(lin / 512), seed);
        stochastic_rounding_at(mp[idx], (sycl::ext::oneapi::bfloat16*)&p[idx], rnd);
    }).wait();
}

#endif
