/*
(Approximate) GeLU non-linearity layer (ported from llmc/gelu.cuh)
*/
#ifndef SYCL_GELU_HPP
#define SYCL_GELU_HPP

#include <assert.h>
// llmc internal imports
#include "sycl_common.h"
#include "sycl_utils.hpp"

// ----------------------------------------------------------------------------
// SYCL kernels

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)

inline void gelu_forward(floatX* out, const floatX* inp, size_t N, sycl::queue& q) {
    NVTX_RANGE_FN();
    if (N == 0) return;
    floatX* o = out; const floatX* x = inp;
    // N is a multiple of 128 in all llm.c uses (B*T*4C), but guard anyway
    q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> id) {
        size_t i = id[0];
        float xi = (float)x[i];
        float cube = 0.044715f * xi * xi * xi;
        o[i] = (floatX)(0.5f * xi * (1.0f + sycl::tanh(GELU_SCALING_FACTOR * (xi + cube))));
    }).wait();
}

inline void gelu_backward_inplace(floatX* d_in_out, const floatX* inp, size_t N, sycl::queue& q) {
    NVTX_RANGE_FN();
    if (N == 0) return;
    floatX* d = d_in_out; const floatX* x = inp;
    q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> id) {
        size_t i = id[0];
        float xv = (float)x[i];
        float cube = 0.044715f * xv * xv * xv;
        float tanh_arg = GELU_SCALING_FACTOR * (xv + cube);
        float tanh_out = sycl::tanh(tanh_arg);
        float coshf_out = sycl::cosh(tanh_arg);
        float sech_out = 1.0f / (coshf_out * coshf_out);
        float local_grad = 0.5f * (1.0f + tanh_out) + xv * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * xv * xv);
        d[i] = (floatX)(local_grad * (float)d[i]);
    }).wait();
}

#endif
