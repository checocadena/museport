/*
(Approximate) GeLU non-linearity layer. SYCL 2020 port of llmc/gelu.cuh.
*/
#ifndef GELU_H
#define GELU_H

#include <assert.h>
// llmc internal imports
#include "sycl_common.h"
#include "sycl_utils.h"

// ----------------------------------------------------------------------------
// SYCL kernels

#define GELU_SCALING_FACTOR sqrtf(2.0f / (float)M_PI)

inline void gelu_forward_kernel(floatX* out, const floatX* inp, size_t N) {
    constexpr size_t BLOCK = 512;
    sycl_queue.parallel_for(sycl::nd_range<1>(sycl::range<1>(CEIL_DIV(N, BLOCK) * BLOCK),
                                              sycl::range<1>(BLOCK)),
        [=](sycl::nd_item<1> it) {
            size_t idx = it.get_global_linear_id();
            if (idx >= N) { return; }
            float xi = (float)inp[idx];
            float cube = 0.044715f * xi * xi * xi;
            out[idx] = (floatX)(0.5f * xi * (1.0f + sycl::tanh(GELU_SCALING_FACTOR * (xi + cube))));
        }).wait();
}

inline void gelu_backward_inplace_kernel(floatX* d_in_out, const floatX* inp, size_t N) {
    constexpr size_t BLOCK = 512;
    sycl_queue.parallel_for(sycl::nd_range<1>(sycl::range<1>(CEIL_DIV(N, BLOCK) * BLOCK),
                                              sycl::range<1>(BLOCK)),
        [=](sycl::nd_item<1> it) {
            size_t idx = it.get_global_linear_id();
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

// ----------------------------------------------------------------------------
// kernel launchers

inline void gelu_forward(floatX* out, const floatX* inp, size_t N) {
    NVTX_RANGE_FN();
    gelu_forward_kernel(out, inp, N);
}

inline void gelu_backward_inplace(floatX* d_in_out, const floatX* inp, size_t N) {
    NVTX_RANGE_FN();
    gelu_backward_inplace_kernel(d_in_out, inp, N);
}

#endif
