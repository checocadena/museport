/*
(Approximate) GeLU non-linearity layer. SYCL port of gelu.cuh.
Fire-and-forget launches on the in-order queue (no per-kernel waits).
*/
#ifndef SYCL_GELU_H
#define SYCL_GELU_H

#include <assert.h>
// llmc internal imports
#include "sycl_common.h"
#include "sycl_utils.hpp"

// ----------------------------------------------------------------------------
// SYCL kernels

#define GELU_SCALING_FACTOR sqrtf(2.0f / (float)M_PI)
inline void gelu_forward(floatX* out, const floatX* inp, size_t N, sycl::queue& q) {
    NVTX_RANGE_FN();
    size_t vec = x128::size;
    size_t nvec = CEIL_DIV(N, vec);
    q.parallel_for(sycl::range<1>(nvec), [=](sycl::id<1> id) {
        size_t idx = id[0] * vec;
        if (idx >= N) return;
        size_t rem = N - idx;
        if (rem >= vec) {
            x128 packed_out;
            x128 packed_inp = load128cs(inp + idx);
            for(size_t k = 0; k < vec; ++k) {
                float xi = (float)packed_inp[k];
                float cube = 0.044715f * xi * xi * xi;
                packed_out[k] = (floatX)(0.5f * xi * (1.0f + sycl::tanh(GELU_SCALING_FACTOR * (xi + cube))));
            }
            store128(out + idx, packed_out);
        } else {
            for (size_t k = 0; k < rem; k++) {
                float xi = (float)inp[idx + k];
                float cube = 0.044715f * xi * xi * xi;
                out[idx + k] = (floatX)(0.5f * xi * (1.0f + sycl::tanh(GELU_SCALING_FACTOR * (xi + cube))));
            }
        }
    });
}

inline void gelu_backward_inplace(floatX* d_in_out, const floatX* inp, const size_t N, sycl::queue& q) {
    NVTX_RANGE_FN();
    size_t vec = x128::size;
    size_t nvec = CEIL_DIV(N, vec);
    q.parallel_for(sycl::range<1>(nvec), [=](sycl::id<1> id) {
        size_t idx = id[0] * vec;
        if (idx >= N) return;
        size_t rem = N - idx;
        size_t lim = rem >= vec ? vec : rem;
        for (size_t k = 0; k < lim; ++k) {
            float x = (float)inp[idx + k];
            float cube = 0.044715f * x * x * x;
            float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
            float tanh_out = sycl::tanh(tanh_arg);
            float coshf_out = sycl::cosh(tanh_arg);
            float sech_out = 1.0f / (coshf_out * coshf_out);
            float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
            d_in_out[idx + k] = (floatX)(local_grad * (float)d_in_out[idx + k]);
        }
    });
}

#endif
