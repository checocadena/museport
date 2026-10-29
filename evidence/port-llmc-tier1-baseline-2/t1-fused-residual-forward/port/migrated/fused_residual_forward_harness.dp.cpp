/*
Kernels for residual forward pass fused with layernorm

Compile example:
nvcc -O3 --use_fast_math -lcublas -lcublasLt fused_residual_forward.cu -o fused_residual_forward

version 1 is naive port from CPU code to kernel
./fused_residual_forward 1
version 2 packs input into 128 bit memory reads
./fused_residual_forward 2
*/

#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include <stdlib.h>
#include "assert.h"

// museport: fp32 build (#define ENABLE_BF16 removed)
#include "common.h"

// ----------------------------------------------------------------------------
// CPU code reference lol

// museport: CPU reference residual_forward_cpu() removed (it is the hidden check)

// museport: CPU reference layernorm_forward_cpu() removed (it is the hidden check)

// ----------------------------------------------------------------------------
// GPU kernels

// elementwise ops are nice and ez
void residual_forward_kernel1(floatX* out, const floatX* inp1, const floatX* inp2, int N) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
              item_ct1.get_local_id(2);
    if (idx < N) {
        out[idx] = (floatX)((float)inp1[idx] + (float)inp2[idx]);
    }
}

// naive drag and drop implementation into kernel, parallelize over B,T, loop over C
void layernorm_forward_kernel1(floatX* out, floatX* mean, floatX* rstd,
                                          const floatX* inp, const floatX* weight, const floatX* bias,
                                          int N, int C) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
              item_ct1.get_local_id(2);
    float eps = 1e-5f;

    if (idx < N) {
        // seek to the input position inp[idx,:]
        const floatX* x = inp + idx * C;
        // calculate the mean
        float m = 0.0f;
        for (int i = 0; i < C; i++) {
            m += (float)x[i];
        }
        m = m / C;
        // calculate the variance (without any bias correction)
        float v = 0.0f;
        for (int i = 0; i < C; i++) {
            float xshift = (float)x[i] - m;
            v += xshift * xshift;
        }
        v = v / C;
        // calculate the rstd
        float s = 1.0f / sycl::sqrt(v + eps);
        // seek to the output position in out[idx,:]
        floatX* out_idx = out + idx * C;
        for (int i = 0; i < C; i++) {
            float n = (s * ((float)x[i] - m)); // normalized output
            float o = n * (float)weight[i] + (float)bias[i]; // scale and shift it
            out_idx[i] = o; // write
        }
        // cache the mean and rstd for the backward pass later
        mean[idx] = m;
        rstd[idx] = s;
    }
}

// naive fusion; uncoalesced access pattern leads to terrible performance
void fused_residual_forward2(floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                                        const floatX* inp1, const floatX* inp2,
                                        const floatX* weight, const floatX* bias,
                                        int N, int C) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
              item_ct1.get_local_id(2);
    if(idx > N) return;

    // adjust pointers to current token
    residual += C * idx;
    normed += C * idx;
    inp1 += C * idx;
    inp2 += C * idx;

    float eps = 1e-5f;

    float m = 0.0f;
    for(int c = 0; c < C; ++c) {
        float out = (float)inp1[c] + (float)inp2[c];
        m += out;
        residual[c] = (floatX)out;
    }

    m = m / C;
    float v = 0.0f;
    for (int c = 0; c < C; c++) {
        float xshift = (float)residual[c] - m;
        v += xshift * xshift;
    }
    v = v / C;

    // calculate the rstd
    float s = 1.0f / sycl::sqrt(v + eps);
    for (int c = 0; c < C; c++) {
        float n = (s * ((float)residual[c] - m)); // normalized output
        float o = n * (float)weight[c] + (float)bias[c]; // scale and shift it
        normed[c] = (floatX)o; // write
    }
    // cache the mean and rstd for the backward pass later
    mean[idx] = (floatX)m;
    rstd[idx] = (floatX)s;
}

// handle one token per warp for coalesced access
void fused_residual_forward3(floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                                        const floatX* inp1, const floatX* inp2,
                                        const floatX* weight, const floatX* bias,
                                        int N, int C) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr const int WarpSize = 32;
    assert(sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_range(
               2) == WarpSize);
    int idx = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
              item_ct1.get_local_id(1);
    if(idx > N) return;

    // adjust pointers to current token
    residual += C * idx;
    normed += C * idx;
    inp1 += C * idx;
    inp2 += C * idx;

    float eps = 1e-5f;
    float m = 0.0f;
    for (int c = item_ct1.get_local_id(2); c < C; c += WarpSize) {
        float out = (float)inp1[c] + (float)inp2[c];
        m += out;
        residual[c] = out;
    }

    m = warpReduceSum(m);

    m = m / C;
    float v = 0.0f;
    for (int c = item_ct1.get_local_id(2); c < C; c += WarpSize) {
        float xshift = (float)residual[c] - m;
        v += xshift * xshift;
    }

    v = warpReduceSum(v);
    v = v / C;

    // calculate the rstd
    float s = 1.0f / sycl::sqrt(v + eps);
    for (int c = item_ct1.get_local_id(2); c < C; c += WarpSize) {
        float n = (s * ((float)residual[c] - m)); // normalized output
        float o = n * (float)weight[c] + (float)bias[c]; // scale and shift it
        normed[c] = o; // write
    }
    // cache the mean and rstd for the backward pass later
    if (item_ct1.get_local_id(2) == 0) {
        mean[idx] = m;
        rstd[idx] = s;
    }
}

// vectorized loading, single pass stats, streaming access and zigzag loop
/*
DPCT1110:1: The total declared local variable size in device function
fused_residual_forward_kernel4 exceeds 128 bytes and may cause high register
pressure. Consult with your hardware vendor to find the total register size
available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
void fused_residual_forward_kernel4(floatX *residual, floatX *normed,
                                    floatX *mean, floatX *rstd,
                                    const floatX *inp1, const floatX *inp2,
                                    const floatX *weight, const floatX *bias,
                                    int N, int C) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    using x128 = Packed128<floatX>;
    constexpr const int WarpSize = 32;
    assert(sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_range(
               2) == WarpSize);
    int idx = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
              item_ct1.get_local_id(1);
    if(idx > N) return;

    // adjust pointers to current token
    residual += C * idx;
    normed += C * idx;
    inp1 += C * idx;
    inp2 += C * idx;

    const float eps = 1e-5f;
    float sum = 0.0f;
    float sum_sq = 0.0f;
    int c = item_ct1.get_local_id(2) * x128::size;
    for(; c < C; c += WarpSize * x128::size) {
        const x128 in1 = load128cs(inp1 + c);
        const x128 in2 = load128cs(inp2 + c);
        x128 out;
        for(int k = 0; k < x128::size; ++k) {
            out[k] = (floatX)((float)in1[k] + (float)in2[k]);
            sum += (float)out[k];
            sum_sq += (float)out[k] * (float)out[k];
        }
        store128(residual + c, out);
    }

    sum = warpReduceSum(sum);
    sum_sq = warpReduceSum(sum_sq);

    float m = sum / C;
    float v = sum_sq / C - m * m;
    float s = sycl::rsqrt(v + eps);

    c -= WarpSize * x128::size;
    for(; c >= 0; c -= WarpSize * x128::size) {
        const x128 res = load128cs(residual + c);
        const x128 w = load128(weight + c);
        const x128 b = load128(bias + c);
        x128 out;
        for(int k = 0; k < x128::size; ++k) {
            float n = s * ((float)res[k] - m); // normalized output
            float o = n * (float)w[k] + (float)b[k]; // scale and shift it
            out[k] = o;
        }

        store128cs(normed + c, out);
    }
    // cache the mean and rstd for the backward pass later
    if (item_ct1.get_local_id(2) == 0) {
        mean[idx] = m;
        rstd[idx] = s;
    }
}

// what do you want in shared memory? EVERYTHING!
// thus, we no longer require zigzag loops and can do the numerically more stable variance estimation
// needs special attention in the kernel launcher to ensure we have enough smem.
/*
DPCT1110:2: The total declared local variable size in device function
fused_residual_forward_kernel5 exceeds 128 bytes and may cause high register
pressure. Consult with your hardware vendor to find the total register size
available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
void fused_residual_forward_kernel5(floatX *residual, floatX *normed,
                                    floatX *mean, floatX *rstd,
                                    const floatX *inp1, const floatX *inp2,
                                    const floatX *weight, const floatX *bias,
                                    int N, int C, uint8_t *dpct_local) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr const int WarpSize = 32;
    assert(sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_range(
               2) == WarpSize);

    // load weights and biases into shared memory
    // do this before we allow any threads to exit!
    auto params = (char *)dpct_local;
    // load128/store128 sometimes generated multiple instructions when the types here were floatX*, so
    // let's keep everything as x128
    x128* s_weight = reinterpret_cast<x128*>(params);
    x128* s_bias = reinterpret_cast<x128*>(params) + (C / x128::size);
    x128 *s_res = reinterpret_cast<x128 *>(params) +
                  ((2 + item_ct1.get_local_id(1)) * C / x128::size);

    int sidx =
        (item_ct1.get_local_id(2) + WarpSize * item_ct1.get_local_id(1)) *
        x128::size;
    for (int i = sidx; i < C;
         i += item_ct1.get_local_range(1) * WarpSize * x128::size) {
        s_weight[i/x128::size] = load128(weight + i);
        s_bias[i/x128::size] = load128(bias + i);
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);

    int idx = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
              item_ct1.get_local_id(1);
    if(idx > N) return;

    // adjust pointers to current token
    residual += C * idx;
    normed += C * idx;
    inp1 += C * idx;
    inp2 += C * idx;

    const float eps = 1e-5f;
    float sum = 0.0f;
    for (int c = item_ct1.get_local_id(2) * x128::size; c < C;
         c += WarpSize * x128::size) {
        const x128 in1 = load128cs(inp1 + c);
        const x128 in2 = load128cs(inp2 + c);
        x128 out;
        for(int k = 0; k < x128::size; ++k) {
            out[k] = (floatX)((float)in1[k] + (float)in2[k]);
            sum += (float)out[k];
        }
        store128cs(residual + c, out);
        s_res[c / x128::size] = out;
    }

    sum = warpReduceSum(sum);
    float m = sum / C;
    float v = 0.f;

    for (int c = item_ct1.get_local_id(2) * x128::size; c < C;
         c += WarpSize * x128::size) {
        const x128 res = s_res[c / x128::size];
        for(int k = 0; k < x128::size; ++k) {
            v += ((float)res[k] - m) * ((float)res[k] - m);
        }
    }

    v = warpReduceSum(v) / C;
    float s = sycl::rsqrt(v + eps);

    for (int c = item_ct1.get_local_id(2) * x128::size; c < C;
         c += WarpSize * x128::size) {
        const x128 res = s_res[c / x128::size];
        const x128 w = s_weight[c / x128::size];
        const x128 b = s_bias[c / x128::size];
        x128 out;
        for(int k = 0; k < x128::size; ++k) {
            float n = s * ((float)res[k] - m); // normalized output
            float o = n * (float)w[k] + (float)b[k]; // scale and shift it
            out[k] = o;
        }

        store128cs(normed + c, out);
    }
    // cache the mean and rstd for the backward pass later
    if (item_ct1.get_local_id(2) == 0) {
        mean[idx] = m;
        rstd[idx] = s;
    }
}


// using multiple warps per token, and keep threads persistent, so we never have to reload weights and biases
// if we had one warp per token, though, this would require us to use a huge amount of shared memory. Therefore,
// we use multiple warps per token; but generally we cannot use the entire block, because that would give too
// little work per warp to be effective (each warp processes 256 bfloat16 elements, so for C=768 more than 3 warps
// will just mean idle). Therefore, we add a z dimension, where warps with different z handle different tokens.
// all this makes the launcher logic more complicated :(
/*
DPCT1110:3: The total declared local variable size in device function
fused_residual_forward_kernel6 exceeds 128 bytes and may cause high register
pressure. Consult with your hardware vendor to find the total register size
available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
void fused_residual_forward_kernel6(floatX *residual, floatX *normed,
                                    floatX *mean, floatX *rstd,
                                    const floatX *inp1, const floatX *inp2,
                                    const floatX *weight, const floatX *bias,
                                    int N, int C, uint8_t *dpct_local) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr const int WarpSize = 32;
    assert(sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_range(
               2) == WarpSize);

    // load weights and biases into shared memory
    // do this before we allow any threads to exit!
    auto params = (char *)dpct_local;
    // load128/store128 sometimes generated multiple instructions when the types here were floatX*, so
    // let's keep everything as x128
    // weights and biases are  shared among all tokens
    x128* s_weight = reinterpret_cast<x128*>(params);
    x128* s_bias = reinterpret_cast<x128*>(params + C * sizeof(floatX));
    // residual output (input to layernorm) is independent for each sub-block indicates by threadIdx.z
    x128 *s_res = reinterpret_cast<x128 *>(
        params + (2 + item_ct1.get_local_id(0)) * C * sizeof(floatX));
    // similarly, each sub-block needs its own reduction buffers
    float *s_mean = reinterpret_cast<float *>(
        params + (2 + item_ct1.get_local_range(0)) * C * sizeof(floatX) +
        item_ct1.get_local_id(0) * 32 * sizeof(float));
    float *s_var = reinterpret_cast<float *>(
        params + (2 + item_ct1.get_local_range(0)) * C * sizeof(floatX) +
        32 * sizeof(float) *
            (item_ct1.get_local_range(0) + item_ct1.get_local_id(0)));

    int cidx =
        (item_ct1.get_local_id(2) + WarpSize * item_ct1.get_local_id(1)) *
        x128::size;
    int step = item_ct1.get_local_range(1) * WarpSize * x128::size;

    for(int c = cidx; c < C; c += step) {
        s_weight[c / x128::size] = load128(weight + c);
        s_bias[c / x128::size] = load128(bias + c);
    }

    // the block-level reductions will cause sync before the first time we read these
    // => no syncthreads needed here

    // loop over all tokens
    for (int tidx = item_ct1.get_group(2) * item_ct1.get_local_range(0) +
                    item_ct1.get_local_id(0);
         tidx < N;
         tidx += item_ct1.get_group_range(2) * item_ct1.get_local_range(0)) {
        // adjust pointers to current token
        floatX* residual_bt = residual + C * tidx;
        floatX* normed_bt = normed + C * tidx;
        const floatX* inp1_bt = inp1 + C * tidx;
        const floatX* inp2_bt = inp2 + C * tidx;

        const float eps = 1e-5f;
        float sum = 0.0f;
        for (int c = cidx; c < C; c += step) {
            const x128 in1 = load128cs(inp1_bt + c);
            const x128 in2 = load128cs(inp2_bt + c);
            x128 out;
            for (int k = 0; k < x128::size; ++k) {
                out[k] = (float) in1[k] + (float) in2[k];
                sum += (float) out[k];
            }
            store128cs(residual_bt + c, out);
            s_res[c / x128::size] = out;
        }
        sum = warpReduceSum(sum);
        if (item_ct1.get_local_id(2) == 0) {
            s_mean[item_ct1.get_local_id(1)] = sum;
        }
        /*
        DPCT1118:4: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:31: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        float m =
            warpReduceSum(item_ct1.get_local_id(2) < item_ct1.get_local_range(1)
                              ? s_mean[item_ct1.get_local_id(2)]
                              : 0.f) /
            C;
        // normally, we'd syncthread here to make sure that no warp is already at the next
        // iteration of the loop, messing with s_mean. The fact that we interleave s_mean and s_var means
        // we don't need these additional syncs.
        float v = 0.f;

        for (int c = cidx; c < C; c += step) {
            const x128 res = s_res[c / x128::size];
            for (int k = 0; k < x128::size; ++k) {
                v += ((float) res[k] - m) * ((float) res[k] - m);
            }
        }

        v = warpReduceSum(v);
        if (item_ct1.get_local_id(2) == 0) {
            s_var[item_ct1.get_local_id(1)] = v;
        }
        /*
        DPCT1118:5: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:32: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        v = warpReduceSum(item_ct1.get_local_id(2) < item_ct1.get_local_range(1)
                              ? s_var[item_ct1.get_local_id(2)]
                              : 0.f) /
            C;
        float s = sycl::rsqrt(v + eps);

        for (int c = cidx; c < C; c += step) {
            const x128 res = s_res[c / x128::size];
            const x128 w = s_weight[c / x128::size];
            const x128 b = s_bias[c / x128::size];
            x128 out;
            for (int k = 0; k < x128::size; ++k) {
                float n = s * ((float) res[k] - m); // normalized output
                float o = n * (float) w[k] + (float) b[k]; // scale and shift it
                out[k] = o;
            }

            store128(normed_bt + c, out);
        }
        // cache the mean and rstd for the backward pass later
        if (item_ct1.get_local_id(2) == 0 && item_ct1.get_local_id(1) == 0) {
            mean[tidx] = m;
            rstd[tidx] = s;
        }
    }
}



// ----------------------------------------------------------------------------
// kernel launcher

void fused_residual_forward1(floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                             const floatX* inp1, const floatX* inp2,
                             const floatX* weight, const floatX* bias,
                             int N, int C, const int block_size) {
    const int grid_size_resid = ceil_div(N * C, block_size);
    /*
    DPCT1049:6: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
        auto N_C_ct3 = N * C;

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size_resid) *
                                  sycl::range<3>(1, 1, block_size),
                              sycl::range<3>(1, 1, block_size)),
            [=](sycl::nd_item<3> item_ct1) {
                residual_forward_kernel1(residual, inp1, inp2, N_C_ct3);
            });
    });
    /*
    DPCT1010:33: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    cudaCheck(0);
    const int grid_size_ln = ceil_div(N, block_size);
    /*
    DPCT1049:7: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size_ln) *
                              sycl::range<3>(1, 1, block_size),
                          sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) {
            layernorm_forward_kernel1(normed, mean, rstd, residual, weight,
                                      bias, N, C);
        });
    /*
    DPCT1010:34: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    cudaCheck(0);
}

void fused_residual_forward2(floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                             const floatX* inp1, const floatX* inp2,
                             const floatX* weight, const floatX* bias,
                             int N, int C, const int block_size) {
    const int grid_size = ceil_div(N, (int)(block_size));
    /*
    DPCT1049:8: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                              sycl::range<3>(1, 1, block_size),
                          sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) {
            fused_residual_forward2(residual, normed, mean, rstd, inp1, inp2,
                                    weight, bias, N, C);
        });
    /*
    DPCT1010:35: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    cudaCheck(0);
}

void fused_residual_forward3(floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                             const floatX* inp1, const floatX* inp2,
                             const floatX* weight, const floatX* bias,
                             int N, int C, const int block_size) {
    int block_y = block_size / 32;
    const int grid_size = ceil_div(N, block_y);
    /*
    DPCT1049:9: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                              sycl::range<3>(1, block_y, 32),
                          sycl::range<3>(1, block_y, 32)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
            fused_residual_forward3(residual, normed, mean, rstd, inp1, inp2,
                                    weight, bias, N, C);
        });
    /*
    DPCT1010:36: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    cudaCheck(0);
}

void fused_residual_forward4(floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                             const floatX* inp1, const floatX* inp2,
                             const floatX* weight, const floatX* bias,
                             int N, int C, const int block_size) {
    int block_y = block_size / 32;
    const int grid_size = ceil_div(N, block_y);
    /*
    DPCT1049:10: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                              sycl::range<3>(1, block_y, 32),
                          sycl::range<3>(1, block_y, 32)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
            fused_residual_forward_kernel4(residual, normed, mean, rstd, inp1,
                                           inp2, weight, bias, N, C);
        });
    /*
    DPCT1010:37: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    cudaCheck(0);
}

void fused_residual_forward5(floatX *residual, floatX *normed, floatX *mean,
                             floatX *rstd, const floatX *inp1,
                             const floatX *inp2, const floatX *weight,
                             const floatX *bias, int N, int C,
                             const int block_size) try {
    int block_y = block_size / 32;
    const int grid_size = ceil_div(N, block_y);
    /*
    DPCT1083:12: The size of local memory in the migrated code may be different
    from the original code. Check that the allocated memory size in the migrated
    code is correct.
    */
    size_t smem = (2 + block_y) * C * sizeof(floatX);

    // in order to use more than 48 KiB of smem, need to call cudaFuncSetAttribute
    // this may fail, in which case we fall back to the smem free implementation.
    /*
    DPCT1010:39: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    cudaCheck(0);
    /*
    DPCT1027:40: The call to cudaFuncSetAttribute was replaced with 0 because
    SYCL currently does not support corresponding setting.
    */
    auto status = 0;
    /*
    DPCT1026:38: The call to cudaGetLastError was removed because this
    functionality is redundant in SYCL.
    */
    if (status == 0) {
        /*
        DPCT1049:11: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range<1>(smem), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                                      sycl::range<3>(1, block_y, 32),
                                  sycl::range<3>(1, block_y, 32)),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        fused_residual_forward_kernel5(
                            residual, normed, mean, rstd, inp1, inp2, weight,
                            bias, N, C,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                    });
        });
    } else {
        /*
        DPCT1049:13: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::get_in_order_queue().parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                                  sycl::range<3>(1, block_y, 32),
                              sycl::range<3>(1, block_y, 32)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                fused_residual_forward_kernel4(residual, normed, mean, rstd,
                                               inp1, inp2, weight, bias, N, C);
            });
    }
    /*
    DPCT1010:41: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    cudaCheck(0);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void fused_residual_forward6(floatX *residual, floatX *normed, floatX *mean,
                             floatX *rstd, const floatX *inp1,
                             const floatX *inp2, const floatX *weight,
                             const floatX *bias, int N, int C,
                             const int block_size) try {
    int warps_per_token = std::max(1, C / Packed128<floatX>::size / 32);
    int total_warps = block_size / 32;
    int block_z = std::max(1, total_warps / warps_per_token);
    int block_y = std::max(1, total_warps / block_z);
    /*
    DPCT1083:15: The size of local memory in the migrated code may be different
    from the original code. Check that the allocated memory size in the migrated
    code is correct.
    */
    size_t smem =
        (2 + block_z) * C * sizeof(floatX) + 64 * sizeof(float) * block_z;

    // in order to use more than 48 KiB of smem, need to call cudaFuncSetAttribute
    // this may fail, in which case we fall back to the smem free implementation.
    /*
    DPCT1010:43: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    cudaCheck(0);
    /*
    DPCT1027:44: The call to cudaFuncSetAttribute was replaced with 0 because
    SYCL currently does not support corresponding setting.
    */
    auto status = 0;
    /*
    DPCT1026:42: The call to cudaGetLastError was removed because this
    functionality is redundant in SYCL.
    */
    if (status == 0) {
        const int num_blocks =
            std::max(1, cuda_threads_per_SM * cuda_num_SMs / block_size);
        /*
        DPCT1049:14: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range<1>(smem), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                      sycl::range<3>(block_z, block_y, 32),
                                  sycl::range<3>(block_z, block_y, 32)),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        fused_residual_forward_kernel6(
                            residual, normed, mean, rstd, inp1, inp2, weight,
                            bias, N, C,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                    });
        });
    } else {
        const int grid_size = ceil_div(N, total_warps);
        /*
        DPCT1049:16: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::get_in_order_queue().parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                                  sycl::range<3>(1, total_warps, 32),
                              sycl::range<3>(1, total_warps, 32)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                fused_residual_forward_kernel4(residual, normed, mean, rstd,
                                               inp1, inp2, weight, bias, N, C);
            });
    }
    /*
    DPCT1010:45: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    cudaCheck(0);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

// kernel version dispatch
void fused_residual_forward(int kernel_num, floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                            const floatX* inp1, const floatX* inp2,
                            const floatX* weight, const floatX* bias,
                            int N, int C, const int block_size) {
    switch (kernel_num) {
        case 1:
            fused_residual_forward1(residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C, block_size);
            break;
        case 2:
            fused_residual_forward2(residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C, block_size);
            break;
        case 3:
            fused_residual_forward3(residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C, block_size);
            break;
        case 4:
            fused_residual_forward4(residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C, block_size);
            break;
        case 5:
            fused_residual_forward5(residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C, block_size);
            break;
        case 6:
            fused_residual_forward6(residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C, block_size);
            break;
        default:
            printf("Invalid kernel number\n");
            exit(1);
    }
}

// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// museport contract main: <input> <output> sizes...; runs every kernel variant and
// writes each variant's outputs, in order, as raw float32.
int main(int argc, char** argv) {
    setup_main();
    if (argc != 6) {
        fprintf(stderr, "usage: %s <input> <output> B T C\n", argv[0]);
        return 2;
    }
    const int B = atoi(argv[3]);
    const int T = atoi(argv[4]);
    const int C = atoi(argv[5]);
    FILE* fin = fopen(argv[1], "rb");
    if (!fin) { perror(argv[1]); return 2; }
    float* inp1 = (float*)malloc(sizeof(float) * (size_t)(B*T*C));
    if (fread(inp1, sizeof(float), (size_t)(B*T*C), fin) != (size_t)(B*T*C)) {
        fprintf(stderr, "input too short at inp1\n"); return 2;
    }
    float* inp2 = (float*)malloc(sizeof(float) * (size_t)(B*T*C));
    if (fread(inp2, sizeof(float), (size_t)(B*T*C), fin) != (size_t)(B*T*C)) {
        fprintf(stderr, "input too short at inp2\n"); return 2;
    }
    float* weight = (float*)malloc(sizeof(float) * (size_t)(C));
    if (fread(weight, sizeof(float), (size_t)(C), fin) != (size_t)(C)) {
        fprintf(stderr, "input too short at weight\n"); return 2;
    }
    float* bias = (float*)malloc(sizeof(float) * (size_t)(C));
    if (fread(bias, sizeof(float), (size_t)(C), fin) != (size_t)(C)) {
        fprintf(stderr, "input too short at bias\n"); return 2;
    }
    fclose(fin);
    FILE* fout = fopen(argv[2], "wb");
    if (!fout) { perror(argv[2]); return 2; }
    float* d_inp1;
    cudaCheck(
        DPCT_CHECK_ERROR(d_inp1 = sycl::malloc_device<float>(
                             (size_t)(B * T * C), dpct::get_in_order_queue())));
    cudaCheck(DPCT_CHECK_ERROR(
        dpct::get_in_order_queue()
            .memcpy(d_inp1, inp1, sizeof(float) * (size_t)(B * T * C))
            .wait()));
    float* d_inp2;
    cudaCheck(
        DPCT_CHECK_ERROR(d_inp2 = sycl::malloc_device<float>(
                             (size_t)(B * T * C), dpct::get_in_order_queue())));
    cudaCheck(DPCT_CHECK_ERROR(
        dpct::get_in_order_queue()
            .memcpy(d_inp2, inp2, sizeof(float) * (size_t)(B * T * C))
            .wait()));
    float* d_weight;
    cudaCheck(DPCT_CHECK_ERROR(d_weight = sycl::malloc_device<float>(
                                   (size_t)(C), dpct::get_in_order_queue())));
    cudaCheck(DPCT_CHECK_ERROR(
        dpct::get_in_order_queue()
            .memcpy(d_weight, weight, sizeof(float) * (size_t)(C))
            .wait()));
    float* d_bias;
    cudaCheck(DPCT_CHECK_ERROR(d_bias = sycl::malloc_device<float>(
                                   (size_t)(C), dpct::get_in_order_queue())));
    cudaCheck(
        DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                             .memcpy(d_bias, bias, sizeof(float) * (size_t)(C))
                             .wait()));
    float* residual = (float*)malloc(sizeof(float) * (size_t)(B*T*C));
    float* d_residual;
    cudaCheck(
        DPCT_CHECK_ERROR(d_residual = sycl::malloc_device<float>(
                             (size_t)(B * T * C), dpct::get_in_order_queue())));
    float* normed = (float*)malloc(sizeof(float) * (size_t)(B*T*C));
    float* d_normed;
    cudaCheck(
        DPCT_CHECK_ERROR(d_normed = sycl::malloc_device<float>(
                             (size_t)(B * T * C), dpct::get_in_order_queue())));
    float* mean = (float*)malloc(sizeof(float) * (size_t)(B*T));
    float* d_mean;
    cudaCheck(
        DPCT_CHECK_ERROR(d_mean = sycl::malloc_device<float>(
                             (size_t)(B * T), dpct::get_in_order_queue())));
    float* rstd = (float*)malloc(sizeof(float) * (size_t)(B*T));
    float* d_rstd;
    cudaCheck(
        DPCT_CHECK_ERROR(d_rstd = sycl::malloc_device<float>(
                             (size_t)(B * T), dpct::get_in_order_queue())));
    const int variants[] = {1, 2, 3, 4, 5, 6};
    for (int vi = 0; vi < 6; vi++) {
        const int k = variants[vi];
        cudaCheck(DPCT_CHECK_ERROR(
            dpct::get_in_order_queue()
                .memset(d_residual, 0, sizeof(float) * (size_t)(B * T * C))
                .wait()));
        cudaCheck(DPCT_CHECK_ERROR(
            dpct::get_in_order_queue()
                .memset(d_normed, 0, sizeof(float) * (size_t)(B * T * C))
                .wait()));
        cudaCheck(DPCT_CHECK_ERROR(
            dpct::get_in_order_queue()
                .memset(d_mean, 0, sizeof(float) * (size_t)(B * T))
                .wait()));
        cudaCheck(DPCT_CHECK_ERROR(
            dpct::get_in_order_queue()
                .memset(d_rstd, 0, sizeof(float) * (size_t)(B * T))
                .wait()));
        fused_residual_forward(k, d_residual, d_normed, d_mean, d_rstd, d_inp1, d_inp2, d_weight, d_bias, B*T, C, 256);
        /*
        DPCT1010:46: SYCL uses exceptions to report errors and does not use the
        error codes. The cudaGetLastError function call was replaced with 0. You
        need to rewrite this code.
        */
        cudaCheck(0);
        cudaCheck(DPCT_CHECK_ERROR(
            dpct::get_current_device().queues_wait_and_throw()));
        cudaCheck(
            DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                 .memcpy(residual, d_residual,
                                         sizeof(float) * (size_t)(B * T * C))
                                 .wait()));
        fwrite(residual, sizeof(float), (size_t)(B*T*C), fout);
        cudaCheck(DPCT_CHECK_ERROR(
            dpct::get_in_order_queue()
                .memcpy(normed, d_normed, sizeof(float) * (size_t)(B * T * C))
                .wait()));
        fwrite(normed, sizeof(float), (size_t)(B*T*C), fout);
        cudaCheck(DPCT_CHECK_ERROR(
            dpct::get_in_order_queue()
                .memcpy(mean, d_mean, sizeof(float) * (size_t)(B * T))
                .wait()));
        fwrite(mean, sizeof(float), (size_t)(B*T), fout);
        cudaCheck(DPCT_CHECK_ERROR(
            dpct::get_in_order_queue()
                .memcpy(rstd, d_rstd, sizeof(float) * (size_t)(B * T))
                .wait()));
        fwrite(rstd, sizeof(float), (size_t)(B*T), fout);
    }
    fclose(fout);
    return 0;
}
