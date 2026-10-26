/*
Kernels for layernorm forward pass.

Compile example:
nvcc -O3 --use_fast_math -lcublas -lcublasLt layernorm_forward.cu -o layernorm_forward

version 1 is naive port from CPU code to kernel: parallelizes over B,T, loops over C
./layernorm_forward 1

version 2 parallelizes over all of B,T,C
./layernorm_forward 2

version 3 uses cooperative groups to parallelize over all of B,T,C
./layernorm_forward 3

version 4 uses a more clever way to estimate variance, var(x) = mean(x**2) - mean(x)**2
          (allowing us to do a single pass over x on load)
./layernorm_forward 4

verstion 5 allocates blocks per row instead of warps per row, same alg as 4 otherwise
./layernorm_forward 5
*/

#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "common.h"
// ----------------------------------------------------------------------------
// CPU code reference

// GPT-2 layernorm forward pass
// museport: CPU reference layernorm_forward_cpu() removed (it is the hidden check)

// ----------------------------------------------------------------------------
// GPU kernels

// naive drag and drop implementation into kernel, parallelize over B,T, loop over C
void layernorm_forward_kernel1(float* out, float* mean, float* rstd,
                                 const float* inp, const float* weight, const float* bias,
                                 int N, int C) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
              item_ct1.get_local_id(2);
    float eps = 1e-5f;

    if (idx < N) {
        // seek to the input position inp[idx,:]
        const float* x = inp + idx * C;
        // calculate the mean
        float m = 0.0f;
        for (int i = 0; i < C; i++) {
            m += x[i];
        }
        m = m / C;
        // calculate the variance (without any bias correction)
        float v = 0.0f;
        for (int i = 0; i < C; i++) {
            float xshift = x[i] - m;
            v += xshift * xshift;
        }
        v = v / C;
        // calculate the rstd
        float s = 1.0f / sycl::sqrt(v + eps);
        // seek to the output position in out[idx,:]
        float* out_idx = out + idx * C;
        for (int i = 0; i < C; i++) {
            float n = (s * (x[i] - m)); // normalized output
            float o = n * weight[i] + bias[i]; // scale and shift it
            out_idx[i] = o; // write
        }
        // cache the mean and rstd for the backward pass later
        mean[idx] = m;
        rstd[idx] = s;
    }
}

void mean_kernel(float* mean, const float* inp, int N, int C, int block_size,
                 uint8_t *dpct_local) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto shared = (float *)dpct_local;
    int idx = item_ct1.get_group(2);    // range [0, B*T)
    int tid = item_ct1.get_local_id(2); // range [0, block_size)
    const float* x = inp + idx * C;
    // thread coarsening
    float sum = 0.0f;
    for (int i = tid; i < C; i += block_size) {
        sum += x[i];
    }
    shared[tid] = sum;
    item_ct1.barrier(sycl::access::fence_space::local_space);
    // reductions
    for (int stride = block_size / 2; stride >= 1; stride /= 2) {
        /*
        DPCT1118:1: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        item_ct1.barrier(sycl::access::fence_space::local_space);
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
    }
    // write the final result (at thread 0) to global memory
    if (tid == 0) {
        mean[idx] = shared[0] / C;
    }
}

void rstd_kernel(float* rstd, const float* inp, const float* mean, int N, int C, int block_size,
                 uint8_t *dpct_local) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto shared = (float *)dpct_local;
    int idx = item_ct1.get_group(2);    // range [0, B*T)
    int tid = item_ct1.get_local_id(2); // range [0, block_size)
    const float* x = inp + idx * C;
    float m = mean[idx];
    // thread coarsening
    float sum = 0.0f;
    for (int i = tid; i < C; i += block_size) {
        float diff = x[i] - m;
        sum += diff * diff;
    }
    shared[tid] = sum;
    item_ct1.barrier(sycl::access::fence_space::local_space);
    // reductions
    for (int stride = block_size / 2; stride >= 1; stride /= 2) {
        /*
        DPCT1118:2: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        item_ct1.barrier(sycl::access::fence_space::local_space);
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
    }
    // write the final result (at thread 0) to global memory
    if (tid == 0) {
        rstd[idx] = 1.0f / sycl::sqrt(shared[0] / C + 1e-5f);
    }
}

void normalization_kernel(float* out, const float* inp, float* mean, float* rstd,
                                     const float* weight, const float* bias, int B, int T, int C) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
              item_ct1.get_local_id(2);

    int bt = idx / C;
    int c = idx % C;

    float m = mean[bt];
    float s = rstd[bt];
    float xi = inp[idx];
    float n = s * (xi - m);
    float o = n * weight[c] + bias[c];

    out[idx] = o;
}

void layernorm_forward_kernel3(float* __restrict__ out, float* __restrict__ mean, float* __restrict__ rstd,
                                    const float*  __restrict__ inp, const float*  __restrict__ weight,
                                    const float* __restrict__ bias, int N, int C) {

    sycl::group<3> block =
        sycl::ext::oneapi::this_work_item::get_work_group<3>();
    sycl::sub_group warp = sycl::ext::oneapi::this_work_item::get_sub_group();
    // meta_group_size is the number of warps in a block, and meta_group_rank is the warp index
    int idx = sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_group(2) *
                  sycl::ext::oneapi::this_work_item::get_sub_group()
                      .get_group_linear_range() +
              sycl::ext::oneapi::this_work_item::get_sub_group()
                  .get_group_linear_id();
    if(idx >= N) {
        return;
    }

    // the row of input that this group of threads is responsible for
    const float* x = inp + idx * C;

    // mean
    float sum = 0.0f;
    for (int i = sycl::ext::oneapi::this_work_item::get_sub_group()
                     .get_local_linear_id();
         i < C; i += sycl::ext::oneapi::this_work_item::get_sub_group()
                         .get_local_linear_range()) {
        sum += x[i];
    }
    sum = sycl::reduce_over_group(
        sycl::ext::oneapi::this_work_item::get_sub_group(), sum,
        sycl::plus<float>{});
    float m = sum / C;
    if (sycl::ext::oneapi::this_work_item::get_sub_group()
                .get_local_linear_id() == 0 &&
        mean != nullptr) {
        /*
        DPCT1098:27: The '=' expression is used instead of the __stcs call.
        These two expressions do not provide the exact same functionality. Check
        the generated code for potential precision and/or performance issues.
        */
        *(mean + idx) = m;
    }

    // rstd
    sum = 0.0f;
    for (int i = sycl::ext::oneapi::this_work_item::get_sub_group()
                     .get_local_linear_id();
         i < C; i += sycl::ext::oneapi::this_work_item::get_sub_group()
                         .get_local_linear_range()) {
        float diff = x[i] - m;
        sum += diff * diff;
    }
    sum = sycl::reduce_over_group(
        sycl::ext::oneapi::this_work_item::get_sub_group(), sum,
        sycl::plus<float>{});
    float s = sycl::rsqrt(sum / C + 1e-5f);
    if (sycl::ext::oneapi::this_work_item::get_sub_group()
                .get_local_linear_id() == 0 &&
        rstd != nullptr) {
        /*
        DPCT1098:28: The '=' expression is used instead of the __stcs call.
        These two expressions do not provide the exact same functionality. Check
        the generated code for potential precision and/or performance issues.
        */
        *(rstd + idx) = s;
    }

    // final normalization and scaling by weight/bias
    float* o = out + idx * C;
    for (int c = sycl::ext::oneapi::this_work_item::get_sub_group()
                     .get_local_linear_id();
         c < C; c += sycl::ext::oneapi::this_work_item::get_sub_group()
                         .get_local_linear_range()) {
        // load and store using the .cs "streaming" hint to the compiler,
        // indicating that this data will not be reused soon, and can be streamed through the caches
        // this allows the threads to get more cache-hits for the (shared) weight and bias parameters
        /*
        DPCT1098:30: The '*' expression is used instead of the __ldcs call.
        These two expressions do not provide the exact same functionality. Check
        the generated code for potential precision and/or performance issues.
        */
        float n = s * (*(x + c) - m);
        /*
        DPCT1098:29: The '=' expression is used instead of the __stcs call.
        These two expressions do not provide the exact same functionality. Check
        the generated code for potential precision and/or performance issues.
        */
        *(o + c) = n * weight[c] + bias[c];
    }
}

// same as kernel 3 but uses var(x) == mean(x**2) - mean(x)**2
void layernorm_forward_kernel4(float* __restrict__ out, float* __restrict__ mean, float* __restrict__ rstd,
                                    const float*  __restrict__ inp, const float*  __restrict__ weight,
                                    const float* __restrict__ bias, int N, int C) {

    sycl::group<3> block =
        sycl::ext::oneapi::this_work_item::get_work_group<3>();
    sycl::sub_group warp = sycl::ext::oneapi::this_work_item::get_sub_group();
    int idx = sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_group(2) *
                  sycl::ext::oneapi::this_work_item::get_sub_group()
                      .get_group_linear_range() +
              sycl::ext::oneapi::this_work_item::get_sub_group()
                  .get_group_linear_id();
    if(idx >= N) {
        return;
    }

    // the row of input that this group of threads is responsible for
    const float* x = inp + idx * C;

    // thread coarsening through the row, reduce the sum in series
    float sum = 0.0; // stores sum(x)
    float sum2 = 0.0; // stores sum(x**2)
    for (int i = sycl::ext::oneapi::this_work_item::get_sub_group()
                     .get_local_linear_id();
         i < C; i += sycl::ext::oneapi::this_work_item::get_sub_group()
                         .get_local_linear_range()) {
        float xi = x[i];
        sum += xi;
        sum2 += xi * xi;
    }
    // warp-level reduction at the end
    sum = sycl::reduce_over_group(
        sycl::ext::oneapi::this_work_item::get_sub_group(), sum,
        sycl::plus<float>{}); // sum(x)
    sum2 = sycl::reduce_over_group(
        sycl::ext::oneapi::this_work_item::get_sub_group(), sum2,
        sycl::plus<float>{}); // sum(x**2)
    sum /= C; // mean(x)
    sum2 /= C; // mean(x**2)

    // mean, var, rstd
    float m = sum;
    float var = sum2 - sum * sum;
    float s = sycl::rsqrt(var + 1e-5f);

    // store the mean, no need to cache it
    if (sycl::ext::oneapi::this_work_item::get_sub_group()
                .get_local_linear_id() == 0 &&
        mean != nullptr) {
        /*
        DPCT1098:31: The '=' expression is used instead of the __stcs call.
        These two expressions do not provide the exact same functionality. Check
        the generated code for potential precision and/or performance issues.
        */
        *(mean + idx) = m;
    }
    // store the rstd, no need to cache it
    if (sycl::ext::oneapi::this_work_item::get_sub_group()
                .get_local_linear_id() == 0 &&
        rstd != nullptr) {
        /*
        DPCT1098:32: The '=' expression is used instead of the __stcs call.
        These two expressions do not provide the exact same functionality. Check
        the generated code for potential precision and/or performance issues.
        */
        *(rstd + idx) = s;
    }
    // final normalization and scaling by weight/bias
    float* o = out + idx * C;
    for (int c = sycl::ext::oneapi::this_work_item::get_sub_group()
                     .get_local_linear_id();
         c < C; c += sycl::ext::oneapi::this_work_item::get_sub_group()
                         .get_local_linear_range()) {
        /*
        DPCT1098:34: The '*' expression is used instead of the __ldcs call.
        These two expressions do not provide the exact same functionality. Check
        the generated code for potential precision and/or performance issues.
        */
        float n = s * (*(x + c) - m);
        /*
        DPCT1098:33: The '=' expression is used instead of the __stcs call.
        These two expressions do not provide the exact same functionality. Check
        the generated code for potential precision and/or performance issues.
        */
        *(o + c) = n * weight[c] + bias[c];
    }
}

// like 4, but in kernel 5 we have each block doing one row, not just a single warp
void layernorm_forward_kernel5(float* __restrict__ out, float* __restrict__ mean, float* __restrict__ rstd,
                                    const float*  __restrict__ inp, const float*  __restrict__ weight,
                                    const float* __restrict__ bias, int N, int C,
                                    float *shared_sum, float *shared_sum2) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    sycl::group<3> block =
        sycl::ext::oneapi::this_work_item::get_work_group<3>();
    sycl::sub_group warp = sycl::ext::oneapi::this_work_item::get_sub_group();
     // block_size max is 1024 = 32 * 32 warps
     // warps will be writing into shared memeory after warp-reduce
    int num_warps = item_ct1.get_local_range(2) / 32;
    int warp_id = item_ct1.get_local_id(2) / 32;
    int lane_id = item_ct1.get_local_id(2) % 32;
    int idx = item_ct1.get_group(2); // simply one block per row
    // the row of input that this group of threads is responsible for
    const float* x = inp + idx * C;
    // thread coarsening through the row, reduce the sum in series
    float thread_sum = 0.0; // stores sum(x)
    float thread_sum2 = 0.0; // stores sum(x**2)
    // for (int i = C + threadIdx.x - blockDim.x; i >= 0; i -= blockDim.x) {
    for (int i = item_ct1.get_local_id(2); i < C;
         i += item_ct1.get_local_range(2)) {
        float xi = x[i];
        thread_sum += xi;
        thread_sum2 += xi * xi;
    }
    // warp-level reduction
    float warp_sum = sycl::reduce_over_group(
        sycl::ext::oneapi::this_work_item::get_sub_group(), thread_sum,
        sycl::plus<float>{}); // sum(x)
    float warp_sum2 = sycl::reduce_over_group(
        sycl::ext::oneapi::this_work_item::get_sub_group(), thread_sum2,
        sycl::plus<float>{}); // sum(x**2)
    // store the warp-level reduction in shared memory (we could have lane_id == 0 guard but not needed)
    shared_sum[warp_id] = warp_sum;
    shared_sum2[warp_id] = warp_sum2;
    item_ct1.barrier(sycl::access::fence_space::local_space);
    // load results from shared memory to threads, pad with zeros for threads that are out of bounds
    warp_sum = (lane_id < num_warps) ? shared_sum[lane_id] : 0.0f;
    warp_sum2 = (lane_id < num_warps) ? shared_sum2[lane_id] : 0.0f;
    // now reduce the warp-level reductions
    float block_sum = sycl::reduce_over_group(
        sycl::ext::oneapi::this_work_item::get_sub_group(), warp_sum,
        sycl::plus<float>{}); // sum(x)
    float block_sum2 = sycl::reduce_over_group(
        sycl::ext::oneapi::this_work_item::get_sub_group(), warp_sum2,
        sycl::plus<float>{}); // sum(x**2)
    // mean, var, rstd
    block_sum /= C; // mean(x)
    block_sum2 /= C; // mean(x**2)
    float m = block_sum;
    float var = block_sum2 - m * m;
    float s = sycl::rsqrt(var + 1e-5f);
    // store the mean, no need to cache it
    if (item_ct1.get_local_id(2) == 0 && mean != nullptr) {
        /*
        DPCT1098:35: The '=' expression is used instead of the __stcs call.
        These two expressions do not provide the exact same functionality. Check
        the generated code for potential precision and/or performance issues.
        */
        *(mean + idx) = m;
    }
    // store the rstd, no need to cache it
    if (item_ct1.get_local_id(2) == 0 && rstd != nullptr) {
        /*
        DPCT1098:36: The '=' expression is used instead of the __stcs call.
        These two expressions do not provide the exact same functionality. Check
        the generated code for potential precision and/or performance issues.
        */
        *(rstd + idx) = s;
    }
    // final normalization and scaling by weight/bias
    float* o = out + idx * C;
    for (int i = item_ct1.get_local_id(2); i < C;
         i += item_ct1.get_local_range(2)) {
        /*
        DPCT1098:38: The '*' expression is used instead of the __ldcs call.
        These two expressions do not provide the exact same functionality. Check
        the generated code for potential precision and/or performance issues.
        */
        float n = s * (*(x + i) - m);
        /*
        DPCT1098:37: The '=' expression is used instead of the __stcs call.
        These two expressions do not provide the exact same functionality. Check
        the generated code for potential precision and/or performance issues.
        */
        *(o + i) = n * weight[i] + bias[i];
    }
}

// Inspired by `fused_residual_forward_kernel5` in fused_residual_forward.cu
/*
DPCT1110:3: The total declared local variable size in device function
layernorm_forward_kernel6 exceeds 128 bytes and may cause high register
pressure. Consult with your hardware vendor to find the total register size
available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
void layernorm_forward_kernel6(
    float *__restrict__ out, float *__restrict__ mean, float *__restrict__ rstd,
    const float *__restrict__ inp, const float *__restrict__ weight,
    const float *__restrict__ bias, int N, int C, uint8_t *dpct_local) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
assert(sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_range(2) ==
       WARP_SIZE);

    // load weights and biases into shared memory
    // do this before we allow any threads to exit!
    auto params = (char *)dpct_local;
    // load128/store128 sometimes generated multiple instructions when the types here were floatX*, so
    // let's keep everything as x128
    x128* s_weight = reinterpret_cast<x128*>(params);
    x128* s_bias = reinterpret_cast<x128*>(params) + (C / x128::size);
    x128 *s_in = reinterpret_cast<x128 *>(params) +
                 ((2 + item_ct1.get_local_id(1)) * C / x128::size);

    int sidx =
        (item_ct1.get_local_id(2) + WARP_SIZE * item_ct1.get_local_id(1)) *
        x128::size;
    for (int i = sidx; i < C;
         i += item_ct1.get_local_range(1) * WARP_SIZE * x128::size) {
        s_weight[i/x128::size] = load128(weight + i);
        s_bias[i/x128::size] = load128(bias + i);
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);

    int idx = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
              item_ct1.get_local_id(1);
    if(idx >= N) { return; } // guard

    // adjust pointers to current token
    inp += idx * C;
    out += idx * C;

    const float eps = 1e-5f;
    float sum = 0.0f;
    for (int c = item_ct1.get_local_id(2) * x128::size; c < C;
         c += WARP_SIZE * x128::size) {
        const x128 in_data = load128cs(inp + c);
        for(int k = 0; k < x128::size; ++k) {
            sum += (float)in_data[k];
        }
        s_in[c / x128::size] = in_data;
    }

    sum = warpReduceSum(sum);
    float m = sum / C;
    float v = 0.f;

    for (int c = item_ct1.get_local_id(2) * x128::size; c < C;
         c += WARP_SIZE * x128::size) {
        const x128 in_data = s_in[c / x128::size];
        for(int k = 0; k < x128::size; ++k) {
            v += ((float)in_data[k] - m) * ((float)in_data[k] - m);
        }
    }

    v = warpReduceSum(v) / C;
    float s = sycl::rsqrt(v + eps);

    for (int c = item_ct1.get_local_id(2) * x128::size; c < C;
         c += WARP_SIZE * x128::size) {
        const x128 in_data = s_in[c / x128::size];
        const x128 w = s_weight[c / x128::size];
        const x128 b = s_bias[c / x128::size];
        x128 out_data;
        for(int k = 0; k < x128::size; ++k) {
            float n = s * ((float)in_data[k] - m); // normalized output
            float o = n * (float)w[k] + (float)b[k]; // scale and shift it
            out_data[k] = o;
        }

        store128cs(out + c, out_data);
    }
    // cache the mean and rstd for the backward pass later
    if (item_ct1.get_local_id(2) == 0 && mean != nullptr) {
        /*
        DPCT1098:39: The '=' expression is used instead of the __stcs call.
        These two expressions do not provide the exact same functionality. Check
        the generated code for potential precision and/or performance issues.
        */
        *(mean + idx) = m;
    }
    // store the rstd, no need to cache it
    if (item_ct1.get_local_id(2) == 0 && rstd != nullptr) {
        /*
        DPCT1098:40: The '=' expression is used instead of the __stcs call.
        These two expressions do not provide the exact same functionality. Check
        the generated code for potential precision and/or performance issues.
        */
        *(rstd + idx) = s;
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

void layernorm_forward1(float* out, float* mean, float* rstd,
                           const float* inp, const float* weight, const float* bias,
                           int B, int T, int C,
                           const int block_size) {
    const int N = B * T;
    const int grid_size = ceil_div(N, block_size);
    /*
    DPCT1049:4: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                              sycl::range<3>(1, 1, block_size),
                          sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) {
            layernorm_forward_kernel1(out, mean, rstd, inp, weight, bias, N, C);
        });
    /*
    DPCT1010:41: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    cudaCheck(0);
}

void layernorm_forward2(float* out, float* mean, float* rstd,
                       const float* inp, const float* weight, const float* bias,
                       int B, int T, int C,
                       const int block_size) {
    int N = B * T;
    // in mean and rstd, threads cooperate within blocks via reductions
    /*
    DPCT1049:5: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
        /*
        DPCT1083:53: The size of local memory in the migrated code may be
        different from the original code. Check that the allocated memory size
        in the migrated code is correct.
        */
        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
            sycl::range<1>(block_size * sizeof(float)), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, N) *
                                  sycl::range<3>(1, 1, block_size),
                              sycl::range<3>(1, 1, block_size)),
            [=](sycl::nd_item<3> item_ct1) {
                mean_kernel(mean, inp, N, C, block_size,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
            });
    });
    /*
    DPCT1010:42: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    cudaCheck(0);
    /*
    DPCT1049:6: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
        /*
        DPCT1083:54: The size of local memory in the migrated code may be
        different from the original code. Check that the allocated memory size
        in the migrated code is correct.
        */
        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
            sycl::range<1>(block_size * sizeof(float)), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, N) *
                                  sycl::range<3>(1, 1, block_size),
                              sycl::range<3>(1, 1, block_size)),
            [=](sycl::nd_item<3> item_ct1) {
                rstd_kernel(rstd, inp, mean, N, C, block_size,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
            });
    });
    /*
    DPCT1010:43: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    cudaCheck(0);
    // in the normalization, everything just gets flattened out
    const int block_size2 = 256;
    const int grid_size = ceil_div(B * T * C, block_size2);
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                              sycl::range<3>(1, 1, block_size2),
                          sycl::range<3>(1, 1, block_size2)),
        [=](sycl::nd_item<3> item_ct1) {
            normalization_kernel(out, inp, mean, rstd, weight, bias, B, T, C);
        });
    /*
    DPCT1010:44: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    cudaCheck(0);
}

void layernorm_forward3(float* out, float* mean, float* rstd,
                       const float* inp, const float* weight, const float* bias,
                       int B, int T, int C,
                       const int block_size) {
    assert(block_size % 32 == 0);
    const int N = B * T;
    const int grid_size = ceil_div(N * 32, block_size);
    /*
    DPCT1049:7: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                              sycl::range<3>(1, 1, block_size),
                          sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
            layernorm_forward_kernel3(out, mean, rstd, inp, weight, bias, N, C);
        });
    /*
    DPCT1010:45: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    cudaCheck(0);
}

void layernorm_forward4(float* out, float* mean, float* rstd,
                       const float* inp, const float* weight, const float* bias,
                       int B, int T, int C,
                       const int block_size) {
    assert(block_size % 32 == 0);
    const int N = B * T;
    const int grid_size = ceil_div(N * 32, block_size);
    /*
    DPCT1049:8: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                              sycl::range<3>(1, 1, block_size),
                          sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
            layernorm_forward_kernel4(out, mean, rstd, inp, weight, bias, N, C);
        });
    /*
    DPCT1010:46: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    cudaCheck(0);
}

void layernorm_forward5(float* out, float* mean, float* rstd,
                       const float* inp, const float* weight, const float* bias,
                       int B, int T, int C,
                       const int block_size) {
    assert(block_size % 32 == 0);
    assert(block_size <= 1024);
    const int N = B * T;
    const int grid_size = N;
    /*
    DPCT1049:9: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> shared_sum_acc_ct1(sycl::range<1>(32),
                                                          cgh);
        sycl::local_accessor<float, 1> shared_sum2_acc_ct1(sycl::range<1>(32),
                                                           cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                                  sycl::range<3>(1, 1, block_size),
                              sycl::range<3>(1, 1, block_size)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                layernorm_forward_kernel5(
                    out, mean, rstd, inp, weight, bias, N, C,
                    shared_sum_acc_ct1
                        .get_multi_ptr<sycl::access::decorated::no>()
                        .get(),
                    shared_sum2_acc_ct1
                        .get_multi_ptr<sycl::access::decorated::no>()
                        .get());
            });
    });
    /*
    DPCT1010:47: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    cudaCheck(0);
}

void layernorm_forward6(float *out, float *mean, float *rstd, const float *inp,
                        const float *weight, const float *bias, int B, int T,
                        int C, int block_size) try {
    assert(block_size % 32 == 0);
    const int N = B * T;
    int block_y = block_size / WARP_SIZE;
    const int grid_size = ceil_div(N, block_y);
    /*
    DPCT1083:11: The size of local memory in the migrated code may be different
    from the original code. Check that the allocated memory size in the migrated
    code is correct.
    */
    size_t smem = (2 + block_y) * C * sizeof(float);

    // in order to use more than 48 KiB of smem, need to call cudaFuncSetAttribute
    // this may fail, in which case we fall back to the smem free implementation.
    /*
    DPCT1010:49: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    cudaCheck(0);
    /*
    DPCT1027:50: The call to cudaFuncSetAttribute was replaced with 0 because
    SYCL currently does not support corresponding setting.
    */
    auto status = 0;
    /*
    DPCT1026:48: The call to cudaGetLastError was removed because this
    functionality is redundant in SYCL.
    */
    if (status == 0) {
        /*
        DPCT1049:10: The work-group size passed to the SYCL kernel may exceed
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
                        layernorm_forward_kernel6(
                            out, mean, rstd, inp, weight, bias, N, C,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                    });
        });
    } else {
        const int grid_size = N;
        // fall back to the version without shared memory
        /*
        DPCT1049:12: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
            sycl::local_accessor<float, 1> shared_sum_acc_ct1(
                sycl::range<1>(32), cgh);
            sycl::local_accessor<float, 1> shared_sum2_acc_ct1(
                sycl::range<1>(32), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                                      sycl::range<3>(1, 1, block_size),
                                  sycl::range<3>(1, 1, block_size)),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        layernorm_forward_kernel5(
                            out, mean, rstd, inp, weight, bias, N, C,
                            shared_sum_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get(),
                            shared_sum2_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                    });
        });
    }
    /*
    DPCT1010:51: SYCL uses exceptions to report errors and does not use the
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
void layernorm_forward(int kernel_num,
                    float* out, float* mean, float* rstd,
                    const float* inp, const float* weight, const float* bias,
                    int B, int T, int C,
                    const int block_size) {
    switch (kernel_num) {
        case 1:
            layernorm_forward1(out, mean, rstd, inp, weight, bias, B, T, C, block_size);
            break;
        case 2:
            layernorm_forward2(out, mean, rstd, inp, weight, bias, B, T, C, block_size);
            break;
        case 3:
            layernorm_forward3(out, mean, rstd, inp, weight, bias, B, T, C, block_size);
            break;
        case 4:
            layernorm_forward4(out, mean, rstd, inp, weight, bias, B, T, C, block_size);
            break;
        case 5:
            layernorm_forward5(out, mean, rstd, inp, weight, bias, B, T, C, block_size);
            break;
        case 6:
            layernorm_forward6(out, mean, rstd, inp, weight, bias, B, T, C, block_size);
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
    float* inp = (float*)malloc(sizeof(float) * (size_t)(B*T*C));
    if (fread(inp, sizeof(float), (size_t)(B*T*C), fin) != (size_t)(B*T*C)) {
        fprintf(stderr, "input too short at inp\n"); return 2;
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
    float* d_inp;
    cudaCheck(
        DPCT_CHECK_ERROR(d_inp = sycl::malloc_device<float>(
                             (size_t)(B * T * C), dpct::get_in_order_queue())));
    cudaCheck(DPCT_CHECK_ERROR(
        dpct::get_in_order_queue()
            .memcpy(d_inp, inp, sizeof(float) * (size_t)(B * T * C))
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
    float* out = (float*)malloc(sizeof(float) * (size_t)(B*T*C));
    float* d_out;
    cudaCheck(
        DPCT_CHECK_ERROR(d_out = sycl::malloc_device<float>(
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
                .memset(d_out, 0, sizeof(float) * (size_t)(B * T * C))
                .wait()));
        cudaCheck(DPCT_CHECK_ERROR(
            dpct::get_in_order_queue()
                .memset(d_mean, 0, sizeof(float) * (size_t)(B * T))
                .wait()));
        cudaCheck(DPCT_CHECK_ERROR(
            dpct::get_in_order_queue()
                .memset(d_rstd, 0, sizeof(float) * (size_t)(B * T))
                .wait()));
        layernorm_forward(k, d_out, d_mean, d_rstd, d_inp, d_weight, d_bias, B, T, C, 256);
        /*
        DPCT1010:52: SYCL uses exceptions to report errors and does not use the
        error codes. The cudaGetLastError function call was replaced with 0. You
        need to rewrite this code.
        */
        cudaCheck(0);
        cudaCheck(DPCT_CHECK_ERROR(
            dpct::get_current_device().queues_wait_and_throw()));
        cudaCheck(DPCT_CHECK_ERROR(
            dpct::get_in_order_queue()
                .memcpy(out, d_out, sizeof(float) * (size_t)(B * T * C))
                .wait()));
        fwrite(out, sizeof(float), (size_t)(B*T*C), fout);
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
