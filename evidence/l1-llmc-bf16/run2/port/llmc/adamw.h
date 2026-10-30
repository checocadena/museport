/*
AdamW optimizer (SYCL port of adamw.cuh), with llm.c's stochastic rounding
and determinism (per-tensor seed; same update math incl. lerp).
*/
#ifndef ADAMW_SYCL_H
#define ADAMW_SYCL_H

#include <sycl/sycl.hpp>
// llmc internal imports
#include "sycl_common.h"
#include "sycl_utils.h"

inline float sycl_lerp(float start, float end, float weight) {
    return sycl::fma(weight, end, sycl::fma(-weight, start, start));
}

template <typename Tp, typename Tg>
inline void adamw_update(Tp* params_memory, float* master_params_memory, Tg* grads_memory,
                         float* m_memory, float* v_memory, size_t num_parameters,
                         ptrdiff_t w_stride, ptrdiff_t g_stride, ptrdiff_t s_stride, int num_slices,
                         float learning_rate, float beta1, float beta2, int t, float eps, float weight_decay,
                         float grad_scale, unsigned int seed, sycl::queue& q_) {
    float beta1_correction = 1.0f - powf(beta1, t);
    float beta2_correction = 1.0f - powf(beta2, t);
    size_t total = num_parameters * (size_t)num_slices;
    constexpr size_t BLOCK = 512;
    size_t nblocks = CEIL_DIV(total, BLOCK);
    q_.parallel_for(sycl::range<1>(nblocks * BLOCK), [=](sycl::id<1> id) {
        size_t lin = id[0];
        if (lin >= total) { return; }
        size_t slice = lin / num_parameters;
        size_t idx = lin % num_parameters;
        Tp* p = params_memory + slice * (size_t)w_stride;
        float* mp = master_params_memory ? master_params_memory + slice * (size_t)s_stride : nullptr;
        Tg* g = grads_memory + slice * (size_t)g_stride;
        float* m = m_memory + slice * (size_t)s_stride;
        float* v = v_memory + slice * (size_t)s_stride;
        float grad = grad_scale * (float)g[idx];
        float mv = m[idx];
        float vv = v[idx];
        mv = sycl_lerp(grad, mv, beta1);
        m[idx] = mv;
        vv = sycl_lerp(grad * grad, vv, beta2);
        v[idx] = vv;
        mv /= beta1_correction;
        vv /= beta2_correction;
        float old_param = (mp != nullptr) ? mp[idx] : (float)p[idx];
        float param = old_param - (learning_rate * (mv / (sycl::sqrt(vv) + eps) + weight_decay * old_param));
        if constexpr (std::is_same<Tp, float>::value) {
            p[idx] = param;
        } else {
            // match CUDA indexing: Get2dNoiseUint(threadIdx.x, blockIdx.x*blockDim.x + blockIdx.y, seed)
            // here: thread = lin % 512, blockX = lin / 512 ... approximate with same formula:
            unsigned int tx = (unsigned int)(idx % 512);
            unsigned int bx = (unsigned int)(slice + idx / 512);
            unsigned int random = Get2dNoiseUint((int)tx, (int)bx, seed);
            p[idx] = stochastic_rounding_bf16(param, random);
        }
        if (mp != nullptr) { mp[idx] = param; }
    }).wait();
}

template <typename Tp>
inline void init_from_master(Tp* params_memory, float* master_params_memory, size_t num_parameters,
                             ptrdiff_t w_stride, ptrdiff_t s_stride, int num_slices,
                             unsigned int seed, sycl::queue& q_) {
    size_t total = num_parameters * (size_t)num_slices;
    constexpr size_t BLOCK = 512;
    size_t nblocks = CEIL_DIV(total, BLOCK);
    q_.parallel_for(sycl::range<1>(nblocks * BLOCK), [=](sycl::id<1> id) {
        size_t lin = id[0];
        if (lin >= total) { return; }
        size_t slice = lin / num_parameters;
        size_t idx = lin % num_parameters;
        Tp* p = params_memory + slice * (size_t)w_stride;
        float* mp = master_params_memory + slice * (size_t)s_stride;
        if constexpr (std::is_same<Tp, float>::value) {
            p[idx] = mp[idx];
        } else {
            unsigned int tx = (unsigned int)(idx % 512);
            unsigned int bx = (unsigned int)(slice + idx / 512);
            unsigned int random = Get2dNoiseUint((int)tx, (int)bx, seed);
            p[idx] = stochastic_rounding_bf16(mp[idx], random);
        }
    }).wait();
}

#endif
