/*
(Approximate) GeLU non-linearity layer (SYCL port of gelu.cuh)
*/
#ifndef GELU_SYCL_H
#define GELU_SYCL_H

#include <assert.h>
#include <sycl/sycl.hpp>
#include "sycl_common.h"
#include "sycl_utils.h"

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)

inline void gelu_forward(floatX* out, const floatX* inp, size_t N, sycl::queue& q) {
    NVTX_RANGE_FN();
    constexpr size_t BLOCK = 512;
    size_t nblocks = CEIL_DIV(N, BLOCK);
    q.parallel_for(sycl::range<1>(nblocks * BLOCK), [=](sycl::id<1> id) {
        size_t idx = id[0];
        if (idx >= N) { return; }
        float xi = (float)inp[idx];
        float cube = 0.044715f * xi * xi * xi;
        out[idx] = (floatX)(0.5f * xi * (1.0f + sycl::tanh(GELU_SCALING_FACTOR * (xi + cube))));
    }).wait();
}

inline void gelu_backward_inplace(floatX* d_in_out, const floatX* inp, size_t N, sycl::queue& q) {
    NVTX_RANGE_FN();
    constexpr size_t BLOCK = 256;
    size_t nblocks = CEIL_DIV(N, BLOCK);
    q.parallel_for(sycl::range<1>(nblocks * BLOCK), [=](sycl::id<1> id) {
        size_t idx = id[0];
        if (idx >= N) { return; }
        float x = (float)inp[idx];
        float cube = 0.044715f * x * x * x;
        float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
        float tanh_out = sycl::tanh(tanh_arg);
        float coshf_out = sycl::cosh(tanh_arg);
        float sech_out = 1.0f / (coshf_out * coshf_out);
        float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
        d_in_out[idx] = (floatX)(local_grad * (float)d_in_out[idx]);
    }).wait();
}

#endif
