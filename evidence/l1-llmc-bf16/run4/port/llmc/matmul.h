/*
Matrix Multiplication with oneMath BLAS. SYCL 2020 port of llmc/matmul.cuh.
(no cuBLASLt: bias and GELU epilogues are separate SYCL kernels; oneMath gemm
takes bf16 A/B with an fp32 C, so we write to an fp32 buffer and convert.)
*/
#ifndef MATMUL_H
#define MATMUL_H

#include <assert.h>
#include <type_traits>      // std::bool_constant
// llmc internal imports
#include "sycl_common.h"
#include "sycl_utils.h"
#include "onemath_common.h"
// GELU can be either fused or non-fused
#include "gelu.h"

// ----------------------------------------------------------------------------
// device-wide fp32 scratch for matmul conversions (like cublaslt workspace)
inline float* matmul_tmp = nullptr;
inline size_t matmul_tmp_n = 0;
inline void matmul_ensure_tmp(size_t n) {
    if (n > matmul_tmp_n) {
        if (matmul_tmp != nullptr) { sycl::free(matmul_tmp, sycl_queue); }
        matmul_tmp = sycl_malloc_device<float>(n);
        matmul_tmp_n = n;
    }
}

// ----------------------------------------------------------------------------
// bias-gradient kernels (replacing matmul_backward_bias_kernel9 +
// reduce_add_sum_kernel): deterministic host-side accumulation.

inline void matmul_backward_bias_kernel(floatX* dbias, const floatX* dout,
                                        size_t B, size_t T, size_t OC) {
    size_t N = B * T;
    if (N == 0 || OC == 0) { return; }
    std::vector<floatX> hdout(N * OC);
    sycl_memcpy_d2h(hdout.data(), dout, N * OC * sizeof(floatX));
    std::vector<floatX> hdb(OC);
    sycl_memcpy_d2h(hdb.data(), dbias, OC * sizeof(floatX));
    for (size_t oc = 0; oc < OC; oc++) {
        float acc = 0.0f;
        for (size_t i = 0; i < N; i++) { acc += (float)hdout[i * OC + oc]; }
        hdb[oc] = (floatX)(acc + (float)hdb[oc]);
    }
    sycl_memcpy_h2d(dbias, hdb.data(), OC * sizeof(floatX));
}

// ----------------------------------------------------------------------------
// kernel launchers

// small wrapper around sycl_matmul_forward for the forward pass (keeping the
// historical order of arguments). GELU fusion flag is accepted for API
// compatibility: gelu_fusion<1 with pre_gelu materialises pre_gelu first.
inline void matmul_forward_cublaslt(floatX* out,
                                    floatX* inp, floatX* weight, floatX* bias,
                                    size_t B, size_t T, size_t C, size_t OC,
                                    floatX* pre_gelu = nullptr, int gelu_fusion = 1) {
    if (gelu_fusion < 1 && pre_gelu) {
        matmul_ensure_tmp(OC * B * T);
        sycl_matmul_forward(pre_gelu, inp, weight, bias, matmul_tmp, OC, B * T, C, false);
        gelu_forward(out, pre_gelu, B * T * OC);
    } else {
        matmul_ensure_tmp(OC * B * T);
        sycl_matmul_forward(out, inp, weight, bias, matmul_tmp, OC, B * T, C, false);
        if (pre_gelu != nullptr && pre_gelu != out) {
            // keep the pre-gelu copy llm.c's fused epilogue would have saved
            sycl_memcpy_d2d(pre_gelu, out, B * T * OC * sizeof(floatX));
            // NOTE: out currently holds the raw matmul output; apply gelu in place
            // to match the fused GELU_AUX path (aux=pre_gelu copy, out=gelu(out))
            gelu_forward(out, pre_gelu, B * T * OC);
        } else if (pre_gelu != nullptr && pre_gelu == out) {
            gelu_forward(out, out, B * T * OC);
        }
    }
}

inline void matmul_backward(floatX* dinp, floatX* dweight, floatX* dbias,
                            floatX* dout, floatX* inp, floatX* weight,
                            float* dbias_buffer,
                            size_t B, size_t T, size_t C, size_t OC,
                            floatX* pre_gelu = nullptr, int gelu_fusion = 1) {
    NVTX_RANGE_FN();
    (void)dbias_buffer;

    // backward to bias, if given, does a +=
    if (dbias != nullptr) {
        matmul_backward_bias_kernel(dbias, dout, B, T, OC);
        dbias = nullptr; // prevent double-counting below
    }

    // backward to input, uses = in the backward pass (set the gradient)
    {
        matmul_ensure_tmp((B * T) * C);
        sycl_matmul_bwd_input(dinp, dout, weight, matmul_tmp, B * T, OC, C);
    }

    // backward GELU (never fused in this port; gelu_fusion only selects recompute)
    if (pre_gelu) {
        gelu_backward_inplace(dinp, pre_gelu, B * T * C);
    }
    (void)gelu_fusion;

    // backward to weight, uses += in the backward pass (accumulate the gradient)
    {
        matmul_ensure_tmp(C * OC);
        sycl_matmul_bwd_weight(dweight, inp, dout, matmul_tmp, OC, B * T, C);
    }
}

#endif
