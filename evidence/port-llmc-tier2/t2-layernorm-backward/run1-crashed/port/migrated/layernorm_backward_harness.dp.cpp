/*
Kernels for layernorm backward pass.

Compile example:
nvcc -O3 --use_fast_math -lcublas -lcublasLt layernorm_backward.cu -o layernorm_backward

version 1 is naive port from CPU code to kernel: parallelizes over B,T, loops over C
./layernorm_backward 1

version 2 moves a lot of reduction to shared memory over global memory
./layernorm_backward 2
*/

#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

// museport: fp32 build (#define ENABLE_BF16 removed)
// museport: needed by the fp32 build
#include "common.h"
#include <cmath>

// ----------------------------------------------------------------------------
// CPU code reference

void layernorm_forward_cpu(float* out, float* mean, float* rstd,
                       const float* inp, const float* weight, const float* bias,
                       int B, int T, int C) {
    // reference: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    // both inp and out are (B,T,C) of the activations
    // mean and rstd are (B,T) buffers, to be used later in backward pass
    // at each position (b,t) of the input, the C-dimensional vector
    // of activations gets normalized, then scaled and shifted
    float eps = 1e-5f;
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // seek to the input position inp[b,t,:]
            const float* x = inp + b * T * C + t * C;
            // calculate the mean
            float m = 0.0f;
            for (int i = 0; i < C; i++) {
                m += x[i];
            }
            m = m/C;
            // calculate the variance (without any bias correction)
            float v = 0.0f;
            for (int i = 0; i < C; i++) {
                float xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = v/C;
            // calculate the rstd (reciprocal standard deviation)
            float s = 1.0f / sqrtf(v + eps);
            // seek to the output position in out[b,t,:]
            float* out_bt = out + b * T * C + t * C;
            for (int i = 0; i < C; i++) {
                float n = (s * (x[i] - m)); // normalize
                float o = n * weight[i] + bias[i]; // scale and shift
                out_bt[i] = o; // write
            }
            // cache the mean and rstd for the backward pass later
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

// museport: CPU reference layernorm_backward_cpu() removed (it is the hidden check)

// ----------------------------------------------------------------------------
// GPU kernels

// GPU helper functions for atomicAdd on smaller than 32-bit types
#ifdef ENABLE_BF16
__device__ void atomicAddX(__nv_bfloat16* addr, __nv_bfloat16 val) {
    uintptr_t ptr_val = reinterpret_cast<uintptr_t>(addr);
    __nv_bfloat162* ptr_bf16 = reinterpret_cast<__nv_bfloat162*>(ptr_val & ~uintptr_t(0x3));

    // Prepare the value to add, setting the other half to zero
    __nv_bfloat162 add_val = (ptr_val & 0x3) ? __halves2bfloat162(__ushort_as_bfloat16(0), val)
                                             : __halves2bfloat162(val, __ushort_as_bfloat16(0));
    atomicAdd(ptr_bf16, add_val);
}
#endif
#ifdef ENABLE_FP16
__device__ void atomicAddX(half* addr, half val) {
    uintptr_t ptr_val = reinterpret_cast<uintptr_t>(addr);
    half2* ptr_fp16 = reinterpret_cast<half2*>(ptr_val & ~uintptr_t(0x3));

    // Prepare the value to add, setting the other half to zero
    half2 add_val = (ptr_val & 0x3) ? __halves2half2(__ushort_as_half(0), val)
                                    : __halves2half2(val, __ushort_as_half(0));
    atomicAdd(ptr_fp16, add_val);
}
#endif
void atomicAddX(float* addr, float val) {
    dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(addr,
                                                                       val);
}

// super naive kernel that just parallelizes over B,T and loops over C
void layernorm_backward_kernel1(float* dinp, float* dweight, float* dbias,
                        const float* dout, const float* inp, const float* weight, const float* mean, const float* rstd,
                        int B, int T, int C) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
              item_ct1.get_local_id(2);
    if (idx >= B*T) return;
    int b = idx / T;
    int t = idx % T;

    const float* dout_bt = dout + b * T * C + t * C;
    const float* inp_bt = inp + b * T * C + t * C;
    float* dinp_bt = dinp + b * T * C + t * C;
    const float mean_bt = mean[b * T + t];
    const float rstd_bt = rstd[b * T + t];

    // first: two reduce operations
    float dnorm_mean = 0.0f;
    float dnorm_norm_mean = 0.0f;
    for (int i = 0; i < C; i++) {
        float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
        float dnorm_i = weight[i] * dout_bt[i];
        dnorm_mean += dnorm_i;
        dnorm_norm_mean += dnorm_i * norm_bti;
    }
    dnorm_mean = dnorm_mean / C;
    dnorm_norm_mean = dnorm_norm_mean / C;

    // now iterate again and accumulate all the gradients
    for (int i = 0; i < C; i++) {
        float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
        float dnorm_i = weight[i] * dout_bt[i];
        // gradient contribution to bias
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            &dbias[i], dout_bt[i]);
        // gradient contribution to weight
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            &dweight[i], norm_bti * dout_bt[i]);
        // gradient contribution to input
        float dval = 0.0f;
        dval += dnorm_i; // term 1
        dval -= dnorm_mean; // term 2
        dval -= norm_bti * dnorm_norm_mean; // term 3
        dval *= rstd_bt; // final scale
        dinp_bt[i] += dval;
    }
}

// uses shared memory instead for the reduces
template <typename Tdinp, typename Tparams, typename Tdout, typename Trest>
/*
DPCT1110:1: The total declared local variable size in device function
layernorm_backward_kernel2 exceeds 128 bytes and may cause high register
pressure. Consult with your hardware vendor to find the total register size
available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
void layernorm_backward_kernel2(Tdinp *dinp, Tparams *dweight, Tparams *dbias,
                                const Tdout *dout, const Trest *inp,
                                const Tparams *weight, const Trest *mean,
                                const Trest *rstd, int B, int T, int C,
                                float *dweight_tmp, float *dbias_tmp,
                                uint8_t *dpct_local) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto shared = (float *)dpct_local; // size = 2 * C

    sycl::group<3> block =
        sycl::ext::oneapi::this_work_item::get_work_group<3>();
    sycl::sub_group warp = sycl::ext::oneapi::this_work_item::get_sub_group();
    int idx = item_ct1.get_group(2) *
                  sycl::ext::oneapi::this_work_item::get_sub_group()
                      .get_group_linear_range() +
              sycl::ext::oneapi::this_work_item::get_sub_group()
                  .get_group_linear_id();
    int N = B * T;
    if(idx >= N) { return; } // thread guards

    int b = idx / T;
    int t = idx % T;

    const Tdout* dout_bt = dout + b * T * C + t * C;
    const Trest* inp_bt = inp + b * T * C + t * C;
    Tdinp* dinp_bt = dinp + b * T * C + t * C;
    const float mean_bt = (float)mean[b * T + t];
    const float rstd_bt = (float)rstd[b * T + t];

    // the first half of shared memory is bias, second is weight
    float* dbias_shared = shared;
    float* dweight_shared = shared + C;

    // init shared memory to zero
    #pragma unroll
    for (int i = item_ct1.get_local_id(2); i < C;
         i += item_ct1.get_local_range(2)) {
       dbias_shared[i] = 0.0f;
       dweight_shared[i] = 0.0f;
    }
    /*
    DPCT1065:51: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // first: two reduce operations
    float dnorm_mean = 0.0f;
    float dnorm_norm_mean = 0.0f;
    for (int i = sycl::ext::oneapi::this_work_item::get_sub_group()
                     .get_local_linear_id();
         i < C; i += sycl::ext::oneapi::this_work_item::get_sub_group()
                         .get_local_linear_range()) {
        float norm_bti = ((float)inp_bt[i] - mean_bt) * rstd_bt;
        float dnorm_i = (float)weight[i] * (float)dout_bt[i];
        dnorm_mean += dnorm_i;
        dnorm_norm_mean += dnorm_i * norm_bti;
    }
    dnorm_mean = sycl::reduce_over_group(
        sycl::ext::oneapi::this_work_item::get_sub_group(), dnorm_mean,
        sycl::plus<float>{});
    dnorm_norm_mean = sycl::reduce_over_group(
        sycl::ext::oneapi::this_work_item::get_sub_group(), dnorm_norm_mean,
        sycl::plus<float>{});
    dnorm_mean = dnorm_mean / C;
    dnorm_norm_mean = dnorm_norm_mean / C;

    // now iterate again and accumulate all the gradients
    for (int i = sycl::ext::oneapi::this_work_item::get_sub_group()
                     .get_local_linear_id();
         i < C; i += sycl::ext::oneapi::this_work_item::get_sub_group()
                         .get_local_linear_range()) {
        float norm_bti = ((float)inp_bt[i] - mean_bt) * rstd_bt;
        float dnorm_i = (float)weight[i] * (float)dout_bt[i];
        // gradient contribution to bias
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            &dbias_shared[i], (float)dout_bt[i]);
        // gradient contribution to weight
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            &dweight_shared[i], norm_bti * (float)dout_bt[i]);
        // gradient contribution to input
        float dval = 0.0f;
        dval += dnorm_i; // term 1
        dval -= dnorm_mean; // term 2
        dval -= norm_bti * dnorm_norm_mean; // term 3
        dval *= rstd_bt; // final scale
        dinp_bt[i] = (Tdinp)((float)dinp_bt[i] + dval);
    }
    /*
    DPCT1065:52: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // write to global memory
    for (int i = item_ct1.get_local_id(2); i < C;
         i += item_ct1.get_local_range(2)) {
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            &dbias_tmp[i], dbias_shared[i]);
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            &dweight_tmp[i], dweight_shared[i]);
    }
}

template <typename Tparams>
void copy_to_dweight_dbias(int C, Tparams* dbias, Tparams* dweight, float* dbias_tmp, float* dweight_tmp) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    for (int i = item_ct1.get_local_id(2) +
                 item_ct1.get_local_range(2) * item_ct1.get_group(2);
         i < C;
         i += item_ct1.get_local_range(2) * item_ct1.get_group_range(2)) {
        dbias[i] = (Tparams)dbias_tmp[i];
        dweight[i] = (Tparams)dweight_tmp[i];
    }
}

// kernel2 is 1 threadblock for all Cs on 32 BTs (assuming threadblock size of 1024 threads = 32 warps)
// To minimise the amount of atomicAdds, we will aim for 1 threadblock per SM, processing (total BTs / threadblocks) BTs
template <typename Tdinp, typename Tparams, typename Tdout, typename Trest>
/*
DPCT1110:2: The total declared local variable size in device function
layernorm_backward_kernel3 exceeds 128 bytes and may cause high register
pressure. Consult with your hardware vendor to find the total register size
available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
void layernorm_backward_kernel3(Tdinp *dinp, Tparams *dweight, Tparams *dbias,
                                const Tdout *dout, const Trest *inp,
                                const Tparams *weight, const Trest *mean,
                                const Trest *rstd, int B, int T, int C,
                                uint8_t *dpct_local) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto shared = (float *)dpct_local; // size = 2 * C

    sycl::group<3> block =
        sycl::ext::oneapi::this_work_item::get_work_group<3>();
    sycl::sub_group warp = sycl::ext::oneapi::this_work_item::get_sub_group();
    int base_idx = item_ct1.get_group(2) *
                       sycl::ext::oneapi::this_work_item::get_sub_group()
                           .get_group_linear_range() +
                   sycl::ext::oneapi::this_work_item::get_sub_group()
                       .get_group_linear_id();

    // the first half of shared memory is bias, second is weight
    float* dbias_shared = shared;
    float* dweight_shared = shared + C;

    // init shared memory to zero
    #pragma unroll 4
    for (int i = item_ct1.get_local_id(2); i < C;
         i += item_ct1.get_local_range(2)) {
       dbias_shared[i] = 0.0f;
       dweight_shared[i] = 0.0f;
    }
    /*
    DPCT1065:53: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    int warps_in_grid = item_ct1.get_group_range(2) *
                        sycl::ext::oneapi::this_work_item::get_sub_group()
                            .get_group_linear_range();
    for (int idx = base_idx; idx < B * T; idx += warps_in_grid) {
        int b = idx / T;
        int t = idx % T;

        const Tdout* dout_bt = dout + b * T * C + t * C;
        const Trest* inp_bt = inp + b * T * C + t * C;
        Tdinp* dinp_bt = dinp + b * T * C + t * C;
        const float mean_bt = (float)mean[b * T + t];
        const float rstd_bt = (float)rstd[b * T + t];

        // first: two reduce operations
        float dnorm_mean = 0.0f;
        float dnorm_norm_mean = 0.0f;
        for (int i = sycl::ext::oneapi::this_work_item::get_sub_group()
                         .get_local_linear_id();
             i < C; i += sycl::ext::oneapi::this_work_item::get_sub_group()
                             .get_local_linear_range()) {
            float norm_bti = ((float)inp_bt[i] - mean_bt) * rstd_bt;
            float dnorm_i = (float)weight[i] * (float)dout_bt[i];
            dnorm_mean += dnorm_i;
            dnorm_norm_mean += dnorm_i * norm_bti;
        }
        dnorm_mean = sycl::reduce_over_group(
            sycl::ext::oneapi::this_work_item::get_sub_group(), dnorm_mean,
            sycl::plus<float>{});
        dnorm_norm_mean = sycl::reduce_over_group(
            sycl::ext::oneapi::this_work_item::get_sub_group(), dnorm_norm_mean,
            sycl::plus<float>{});
        dnorm_mean = dnorm_mean / C;
        dnorm_norm_mean = dnorm_norm_mean / C;

        // now iterate again and accumulate all the gradients
        for (int i = sycl::ext::oneapi::this_work_item::get_sub_group()
                         .get_local_linear_id();
             i < C; i += sycl::ext::oneapi::this_work_item::get_sub_group()
                             .get_local_linear_range()) {
            /*
            DPCT1098:55: The '*' expression is used instead of the __ldcs call.
            These two expressions do not provide the exact same functionality.
            Check the generated code for potential precision and/or performance
            issues.
            */
            float dout_i = (float)dout_bt[i];
            /*
            DPCT1098:56: The '*' expression is used instead of the __ldcs call.
            These two expressions do not provide the exact same functionality.
            Check the generated code for potential precision and/or performance
            issues.
            */
            float norm_bti = ((float)inp_bt[i] - mean_bt) * rstd_bt;
            float dnorm_i = (float)weight[i] * dout_i;
            // gradient contribution to bias
            dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
                &dbias_shared[i], dout_i);
            // gradient contribution to weight
            dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
                &dweight_shared[i], norm_bti * dout_i);
            // gradient contribution to input
            float dval = 0.0f;
            dval += dnorm_i; // term 1
            dval -= dnorm_mean; // term 2
            dval -= norm_bti * dnorm_norm_mean; // term 3
            dval *= rstd_bt; // final scale
            dinp_bt[i] = (Tdinp)((float)dinp_bt[i] + dval);
        }
    }
    /*
    DPCT1065:54: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    for (int i = item_ct1.get_local_id(2); i < C;
         i += item_ct1.get_local_range(2)) {
        atomicAddX(&dbias[i], (Tparams)dbias_shared[i]);
        atomicAddX(&dweight[i], (Tparams)dweight_shared[i]);
    }
}

// atomicCAS version of kernel3
template <typename Tdinp, typename Tparams, typename Tdout, typename Trest>
/*
DPCT1110:3: The total declared local variable size in device function
layernorm_backward_kernel4 exceeds 128 bytes and may cause high register
pressure. Consult with your hardware vendor to find the total register size
available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
void layernorm_backward_kernel4(Tdinp *dinp, Tparams *dweight, Tparams *dbias,
                                const Tdout *dout, const Trest *inp,
                                const Tparams *weight, const Trest *mean,
                                const Trest *rstd, int B, int T, int C,
                                uint8_t *dpct_local) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto shared = (float *)dpct_local; // size = 2 * C

    sycl::group<3> block =
        sycl::ext::oneapi::this_work_item::get_work_group<3>();
    sycl::sub_group warp = sycl::ext::oneapi::this_work_item::get_sub_group();
    int base_idx = item_ct1.get_group(2) *
                       sycl::ext::oneapi::this_work_item::get_sub_group()
                           .get_group_linear_range() +
                   sycl::ext::oneapi::this_work_item::get_sub_group()
                       .get_group_linear_id();

    // the first half of shared memory is bias, second is weight
    float* dbias_shared = shared;
    float* dweight_shared = shared + C;

    // init shared memory to zero
    #pragma unroll 4
    for (int i = item_ct1.get_local_id(2); i < C;
         i += item_ct1.get_local_range(2)) {
       dbias_shared[i] = 0.0f;
       dweight_shared[i] = 0.0f;
    }
    /*
    DPCT1065:57: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    int warps_in_grid = item_ct1.get_group_range(2) *
                        sycl::ext::oneapi::this_work_item::get_sub_group()
                            .get_group_linear_range();
    for (int idx = base_idx; idx < B * T; idx += warps_in_grid) {
        int b = idx / T;
        int t = idx % T;

        const Tdout* dout_bt = dout + b * T * C + t * C;
        const Trest* inp_bt = inp + b * T * C + t * C;
        Tdinp* dinp_bt = dinp + b * T * C + t * C;
        const float mean_bt = (float)mean[b * T + t];
        const float rstd_bt = (float)rstd[b * T + t];

        // first: two reduce operations
        float dnorm_mean = 0.0f;
        float dnorm_norm_mean = 0.0f;
        for (int i = sycl::ext::oneapi::this_work_item::get_sub_group()
                         .get_local_linear_id();
             i < C; i += sycl::ext::oneapi::this_work_item::get_sub_group()
                             .get_local_linear_range()) {
            float norm_bti = ((float)inp_bt[i] - mean_bt) * rstd_bt;
            float dnorm_i = (float)weight[i] * (float)dout_bt[i];
            dnorm_mean += dnorm_i;
            dnorm_norm_mean += dnorm_i * norm_bti;
        }
        dnorm_mean = sycl::reduce_over_group(
            sycl::ext::oneapi::this_work_item::get_sub_group(), dnorm_mean,
            sycl::plus<float>{});
        dnorm_norm_mean = sycl::reduce_over_group(
            sycl::ext::oneapi::this_work_item::get_sub_group(), dnorm_norm_mean,
            sycl::plus<float>{});
        dnorm_mean = dnorm_mean / C;
        dnorm_norm_mean = dnorm_norm_mean / C;

        // now iterate again and accumulate all the gradients
        for (int i = sycl::ext::oneapi::this_work_item::get_sub_group()
                         .get_local_linear_id();
             i < C; i += sycl::ext::oneapi::this_work_item::get_sub_group()
                             .get_local_linear_range()) {
            /*
            DPCT1098:59: The '*' expression is used instead of the __ldcs call.
            These two expressions do not provide the exact same functionality.
            Check the generated code for potential precision and/or performance
            issues.
            */
            float dout_i = (float)dout_bt[i];
            /*
            DPCT1098:60: The '*' expression is used instead of the __ldcs call.
            These two expressions do not provide the exact same functionality.
            Check the generated code for potential precision and/or performance
            issues.
            */
            float norm_bti = ((float)inp_bt[i] - mean_bt) * rstd_bt;
            float dnorm_i = (float)weight[i] * dout_i;
            // gradient contribution to bias
            dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
                &dbias_shared[i], dout_i);
            // gradient contribution to weight
            dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
                &dweight_shared[i], norm_bti * dout_i);
            // gradient contribution to input
            float dval = 0.0f;
            dval += dnorm_i; // term 1
            dval -= dnorm_mean; // term 2
            dval -= norm_bti * dnorm_norm_mean; // term 3
            dval *= rstd_bt; // final scale
            dinp_bt[i] = (Tdinp)((float)dinp_bt[i] + dval);
        }
    }
    /*
    DPCT1065:58: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    sycl::vec<sycl::ext::oneapi::bfloat16, 2> *dbiasVec2 =
        reinterpret_cast<sycl::vec<sycl::ext::oneapi::bfloat16, 2> *>(dbias);
    sycl::vec<sycl::ext::oneapi::bfloat16, 2> *dweightVec2 =
        reinterpret_cast<sycl::vec<sycl::ext::oneapi::bfloat16, 2> *>(dweight);

    // write to global memory
    for (int i = item_ct1.get_local_id(2); i < C / 2;
         i += item_ct1.get_local_range(2)) {
        sycl::vec<sycl::ext::oneapi::bfloat16, 2> add_dbias =
            sycl::vec<sycl::ext::oneapi::bfloat16, 2>(
                (sycl::ext::oneapi::bfloat16)dbias_shared[i * 2],
                (sycl::ext::oneapi::bfloat16)dbias_shared[i * 2 + 1]);
        sycl::vec<sycl::ext::oneapi::bfloat16, 2> add_dweight =
            sycl::vec<sycl::ext::oneapi::bfloat16, 2>(
                (sycl::ext::oneapi::bfloat16)dweight_shared[i * 2],
                (sycl::ext::oneapi::bfloat16)dweight_shared[i * 2 + 1]);

        // Get the current value from L2 cache
        /*
        DPCT1098:61: The '*' expression is used instead of the __ldcg call.
        These two expressions do not provide the exact same functionality. Check
        the generated code for potential precision and/or performance issues.
        */
        sycl::vec<sycl::ext::oneapi::bfloat16, 2> current_dbias = dbiasVec2[i];
        /*
        DPCT1098:62: The '*' expression is used instead of the __ldcg call.
        These two expressions do not provide the exact same functionality. Check
        the generated code for potential precision and/or performance issues.
        */
        sycl::vec<sycl::ext::oneapi::bfloat16, 2> current_dweight =
            dweightVec2[i];

        // Add the two values
        sycl::vec<sycl::ext::oneapi::bfloat16, 2> new_dbias =
            add_dbias + current_dbias;
        sycl::vec<sycl::ext::oneapi::bfloat16, 2> new_dweight =
            add_dweight + current_dweight;

        // Write the result back to L2 cache using 32-bit integer atomic compare and exchange
        unsigned int current_dbias32b = *reinterpret_cast<unsigned int*>(&current_dbias);
        unsigned int current_dweight32b = *reinterpret_cast<unsigned int*>(&current_dweight);

        unsigned int new_dbias32b = *reinterpret_cast<unsigned int*>(&new_dbias);
        unsigned int new_dweight32b = *reinterpret_cast<unsigned int*>(&new_dweight);

        unsigned int old_dbias32b = dpct::atomic_compare_exchange_strong<
            sycl::access::address_space::generic_space>(
            (unsigned int *)&dbiasVec2[i], current_dbias32b, new_dbias32b);
        unsigned int old_dweight32b = dpct::atomic_compare_exchange_strong<
            sycl::access::address_space::generic_space>(
            (unsigned int *)&dweightVec2[i], current_dweight32b,
            new_dweight32b);

        // If the value has changed between read and atomic, we need to try again
        while (old_dbias32b != current_dbias32b) {
            current_dbias32b = old_dbias32b;
            new_dbias =
                *reinterpret_cast<sycl::vec<sycl::ext::oneapi::bfloat16, 2> *>(
                    &current_dbias32b) +
                add_dbias;
            new_dbias32b = *reinterpret_cast<unsigned int*>(&new_dbias);
            old_dbias32b = dpct::atomic_compare_exchange_strong<
                sycl::access::address_space::generic_space>(
                (unsigned int *)&dbiasVec2[i], current_dbias32b, new_dbias32b);
        }

        while (old_dweight32b != current_dweight32b) {
            current_dweight32b = old_dweight32b;
            new_dweight =
                *reinterpret_cast<sycl::vec<sycl::ext::oneapi::bfloat16, 2> *>(
                    &current_dweight32b) +
                add_dweight;
            new_dweight32b = *reinterpret_cast<unsigned int*>(&new_dweight);
            old_dweight32b = dpct::atomic_compare_exchange_strong<
                sycl::access::address_space::generic_space>(
                (unsigned int *)&dweightVec2[i], current_dweight32b,
                new_dweight32b);
        }
    }
}

// FP32 scratchpad per threadgroup, zero atomics except atomicAdd on unsigned int for the flag (based on kernel3)
template <typename Tdinp, typename Tparams, typename Tdout, typename Trest>
/*
DPCT1110:4: The total declared local variable size in device function
layernorm_backward_kernel5 exceeds 128 bytes and may cause high register
pressure. Consult with your hardware vendor to find the total register size
available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
void layernorm_backward_kernel5(Tdinp *dinp, Tparams *dweight, Tparams *dbias,
                                float *scratch, const Tdout *dout,
                                const Trest *inp, const Tparams *weight,
                                const Trest *mean, const Trest *rstd, int B,
                                int T, int C, uint8_t *dpct_local) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto shared = (float *)dpct_local; // size = 2 * C + 1

    sycl::group<3> block =
        sycl::ext::oneapi::this_work_item::get_work_group<3>();
    sycl::sub_group warp = sycl::ext::oneapi::this_work_item::get_sub_group();
    int base_idx = item_ct1.get_group(2) *
                       sycl::ext::oneapi::this_work_item::get_sub_group()
                           .get_group_linear_range() +
                   sycl::ext::oneapi::this_work_item::get_sub_group()
                       .get_group_linear_id();

    // the first half of shared memory is bias, second is weight
    float* dbias_shared = shared;
    float* dweight_shared = shared + C;

    // init shared memory to zero
    #pragma unroll 4
    for (int i = item_ct1.get_local_id(2); i < C;
         i += item_ct1.get_local_range(2)) {
       dbias_shared[i] = 0.0f;
       dweight_shared[i] = 0.0f;
    }
    unsigned int *tmp_flag = (unsigned int*)(shared + C*2);
    /*
    DPCT1065:63: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    int warps_in_grid = item_ct1.get_group_range(2) *
                        sycl::ext::oneapi::this_work_item::get_sub_group()
                            .get_group_linear_range();
    for (int idx = base_idx; idx < B * T; idx += warps_in_grid) {
        int b = idx / T;
        int t = idx % T;

        const Tdout* dout_bt = dout + b * T * C + t * C;
        const Trest* inp_bt = inp + b * T * C + t * C;
        Tdinp* dinp_bt = dinp + b * T * C + t * C;
        const float mean_bt = (float)mean[b * T + t];
        const float rstd_bt = (float)rstd[b * T + t];

        // first: two reduce operations
        float dnorm_mean = 0.0f;
        float dnorm_norm_mean = 0.0f;
        for (int i = sycl::ext::oneapi::this_work_item::get_sub_group()
                         .get_local_linear_id();
             i < C; i += sycl::ext::oneapi::this_work_item::get_sub_group()
                             .get_local_linear_range()) {
            float norm_bti = ((float)inp_bt[i] - mean_bt) * rstd_bt;
            float dnorm_i = (float)weight[i] * (float)dout_bt[i];
            dnorm_mean += dnorm_i;
            dnorm_norm_mean += dnorm_i * norm_bti;
        }
        dnorm_mean = sycl::reduce_over_group(
            sycl::ext::oneapi::this_work_item::get_sub_group(), dnorm_mean,
            sycl::plus<float>{});
        dnorm_norm_mean = sycl::reduce_over_group(
            sycl::ext::oneapi::this_work_item::get_sub_group(), dnorm_norm_mean,
            sycl::plus<float>{});
        dnorm_mean = dnorm_mean / C;
        dnorm_norm_mean = dnorm_norm_mean / C;

        // now iterate again and accumulate all the gradients
        for (int i = sycl::ext::oneapi::this_work_item::get_sub_group()
                         .get_local_linear_id();
             i < C; i += sycl::ext::oneapi::this_work_item::get_sub_group()
                             .get_local_linear_range()) {
            /*
            DPCT1098:67: The '*' expression is used instead of the __ldcs call.
            These two expressions do not provide the exact same functionality.
            Check the generated code for potential precision and/or performance
            issues.
            */
            float dout_i = (float)dout_bt[i];
            /*
            DPCT1098:68: The '*' expression is used instead of the __ldcs call.
            These two expressions do not provide the exact same functionality.
            Check the generated code for potential precision and/or performance
            issues.
            */
            float norm_bti = ((float)inp_bt[i] - mean_bt) * rstd_bt;
            float dnorm_i = (float)weight[i] * dout_i;
            // gradient contribution to bias
            dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
                &dbias_shared[i], dout_i);
            // gradient contribution to weight
            dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
                &dweight_shared[i], norm_bti * dout_i);
            // gradient contribution to input
            float dval = 0.0f;
            dval += dnorm_i; // term 1
            dval -= dnorm_mean; // term 2
            dval -= norm_bti * dnorm_norm_mean; // term 3
            dval *= rstd_bt; // final scale
            dinp_bt[i] = (Tdinp)((float)dinp_bt[i] + dval);
        }
    }
    /*
    DPCT1065:64: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    float* scratch_dbias = scratch;
    float *scratch_dweight = scratch + C * item_ct1.get_group_range(2);
    unsigned int *scratchFlag =
        (unsigned int *)(scratch + (2 * C * item_ct1.get_group_range(2)));

    for (int i = item_ct1.get_local_id(2); i < C;
         i += item_ct1.get_local_range(2)) {
        scratch_dbias[i + C * item_ct1.get_group(2)] = dbias_shared[i];
        scratch_dweight[i + C * item_ct1.get_group(2)] = dweight_shared[i];
    }
    /*
    DPCT1078:5: Consider replacing memory_order::acq_rel with
    memory_order::seq_cst for correctness if strong memory order restrictions
    are needed.
    */
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
    /*
    DPCT1065:65: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (item_ct1.get_local_id(2) == 0) {
        *tmp_flag =
            dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
                scratchFlag, 1);
    }
    /*
    DPCT1065:66: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (*tmp_flag == item_ct1.get_group_range(2) - 1) {
        // last block to finish, accumulate the scratchpad
        for (int i = item_ct1.get_local_id(2); i < C;
             i += item_ct1.get_local_range(2)) {
            float dbias_sum = 0.0f;
            float dweight_sum = 0.0f;
            #pragma unroll 8
            for (int j = 0; j < item_ct1.get_group_range(2); j++) {
                dbias_sum += scratch_dbias[i + j*C];
                dweight_sum += scratch_dweight[i + j*C];
            }
            dbias[i] = (Tparams)((float)dbias[i] + dbias_sum);
            dweight[i] = (Tparams)((float)dweight[i] + dweight_sum);
        }
    }
}

// single FP32 scratchpad shared by all the threadblocks (based on kernels 3 & 5)
template <typename Tdinp, typename Tparams, typename Tdout, typename Trest>
/*
DPCT1110:6: The total declared local variable size in device function
layernorm_backward_kernel6 exceeds 128 bytes and may cause high register
pressure. Consult with your hardware vendor to find the total register size
available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
void layernorm_backward_kernel6(Tdinp *dinp, Tparams *dweight, Tparams *dbias,
                                float *scratch, const Tdout *dout,
                                const Trest *inp, const Tparams *weight,
                                const Trest *mean, const Trest *rstd, int B,
                                int T, int C, uint8_t *dpct_local) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto shared = (float *)dpct_local; // size = 2 * C + 1

    sycl::group<3> block =
        sycl::ext::oneapi::this_work_item::get_work_group<3>();
    sycl::sub_group warp = sycl::ext::oneapi::this_work_item::get_sub_group();
    int base_idx = item_ct1.get_group(2) *
                       sycl::ext::oneapi::this_work_item::get_sub_group()
                           .get_group_linear_range() +
                   sycl::ext::oneapi::this_work_item::get_sub_group()
                       .get_group_linear_id();

    // the first half of shared memory is bias, second is weight
    float* dbias_shared = shared;
    float* dweight_shared = shared + C;

    // init shared memory to zero
    #pragma unroll 4
    for (int i = item_ct1.get_local_id(2); i < C;
         i += item_ct1.get_local_range(2)) {
       dbias_shared[i] = 0.0f;
       dweight_shared[i] = 0.0f;
    }
    unsigned int *tmp_flag = (unsigned int*)(shared + C*2);
    /*
    DPCT1065:69: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    int warps_in_grid = item_ct1.get_group_range(2) *
                        sycl::ext::oneapi::this_work_item::get_sub_group()
                            .get_group_linear_range();
    for (int idx = base_idx; idx < B * T; idx += warps_in_grid) {
        int b = idx / T;
        int t = idx % T;

        const Tdout* dout_bt = dout + b * T * C + t * C;
        const Trest* inp_bt = inp + b * T * C + t * C;
        Tdinp* dinp_bt = dinp + b * T * C + t * C;
        const float mean_bt = (float)mean[b * T + t];
        const float rstd_bt = (float)rstd[b * T + t];

        // first: two reduce operations
        float dnorm_mean = 0.0f;
        float dnorm_norm_mean = 0.0f;
        for (int i = sycl::ext::oneapi::this_work_item::get_sub_group()
                         .get_local_linear_id();
             i < C; i += sycl::ext::oneapi::this_work_item::get_sub_group()
                             .get_local_linear_range()) {
            float norm_bti = ((float)inp_bt[i] - mean_bt) * rstd_bt;
            float dnorm_i = (float)weight[i] * (float)dout_bt[i];
            dnorm_mean += dnorm_i;
            dnorm_norm_mean += dnorm_i * norm_bti;
        }
        dnorm_mean = sycl::reduce_over_group(
            sycl::ext::oneapi::this_work_item::get_sub_group(), dnorm_mean,
            sycl::plus<float>{});
        dnorm_norm_mean = sycl::reduce_over_group(
            sycl::ext::oneapi::this_work_item::get_sub_group(), dnorm_norm_mean,
            sycl::plus<float>{});
        dnorm_mean = dnorm_mean / C;
        dnorm_norm_mean = dnorm_norm_mean / C;

        // now iterate again and accumulate all the gradients
        for (int i = sycl::ext::oneapi::this_work_item::get_sub_group()
                         .get_local_linear_id();
             i < C; i += sycl::ext::oneapi::this_work_item::get_sub_group()
                             .get_local_linear_range()) {
            /*
            DPCT1098:73: The '*' expression is used instead of the __ldcs call.
            These two expressions do not provide the exact same functionality.
            Check the generated code for potential precision and/or performance
            issues.
            */
            float dout_i = (float)dout_bt[i];
            /*
            DPCT1098:74: The '*' expression is used instead of the __ldcs call.
            These two expressions do not provide the exact same functionality.
            Check the generated code for potential precision and/or performance
            issues.
            */
            float norm_bti = ((float)inp_bt[i] - mean_bt) * rstd_bt;
            float dnorm_i = (float)weight[i] * dout_i;
            // gradient contribution to bias
            dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
                &dbias_shared[i], dout_i);
            // gradient contribution to weight
            dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
                &dweight_shared[i], norm_bti * dout_i);
            // gradient contribution to input
            float dval = 0.0f;
            dval += dnorm_i; // term 1
            dval -= dnorm_mean; // term 2
            dval -= norm_bti * dnorm_norm_mean; // term 3
            dval *= rstd_bt; // final scale
            dinp_bt[i] = (Tdinp)((float)dinp_bt[i] + dval);
        }
    }

    // Accumulate into a FP32 scratchpad
    // BF16 atomics are potentially much slower... and this is more precise!
    /*
    DPCT1065:70: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    float* scratch_dbias = scratch;
    float* scratch_dweight = scratch + C;
    unsigned int* scratchFlag = (unsigned int*)(scratch + (2 * C));
    for (int i = item_ct1.get_local_id(2); i < C;
         i += item_ct1.get_local_range(2)) {
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            &scratch_dbias[i], dbias_shared[i]);
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            &scratch_dweight[i], dweight_shared[i]);
    }
    /*
    DPCT1065:71: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (item_ct1.get_local_id(2) == 0) {
        *tmp_flag =
            dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
                scratchFlag, 1);
    }
    /*
    DPCT1065:72: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (*tmp_flag == item_ct1.get_group_range(2) - 1) {
        for (int i = item_ct1.get_local_id(2); i < C;
             i += item_ct1.get_local_range(2)) {
            // todo - potentially do stochastic rounding here as well
            dbias[i] = (Tparams)scratch_dbias[i];
            dweight[i] = (Tparams)scratch_dweight[i];
        }
    }
}


// Same as kernel 6 but without cooperative groups or templates
/*
DPCT1110:7: The total declared local variable size in device function
layernorm_backward_kernel7 exceeds 128 bytes and may cause high register
pressure. Consult with your hardware vendor to find the total register size
available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
void layernorm_backward_kernel7(floatX *dinp, floatX *dweight, floatX *dbias,
                                float *scratch, const floatX *dout,
                                const floatX *inp, const floatX *weight,
                                const floatX *mean, const floatX *rstd, int B,
                                int T, int C, uint8_t *dpct_local) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto shared = (float *)dpct_local; // size = 2 * C + 1
    int warpId = item_ct1.get_local_id(2) /
                 sycl::ext::oneapi::this_work_item::get_sub_group()
                     .get_local_range()
                     .get(0); // warp index within a block
    int warpsInBlock = item_ct1.get_local_range(2) /
                       sycl::ext::oneapi::this_work_item::get_sub_group()
                           .get_local_range()
                           .get(0);
    int base_idx = item_ct1.get_group(2) * warpsInBlock + warpId;
    int warpThreadIdx = item_ct1.get_local_id(2) %
                        sycl::ext::oneapi::this_work_item::get_sub_group()
                            .get_local_range()
                            .get(0); // Thread index within the warp
    int warps_in_grid = item_ct1.get_group_range(2) * warpsInBlock;

    // the first half of shared memory is bias, second is weight
    float* dbias_shared = shared;
    float* dweight_shared = shared + C;

    // init shared memory to zero
    #pragma unroll 4
    for (int i = item_ct1.get_local_id(2); i < C;
         i += item_ct1.get_local_range(2)) {
       dbias_shared[i] = 0.0f;
       dweight_shared[i] = 0.0f;
    }
    unsigned int *tmp_flag = (unsigned int*)(shared + C*2);
    /*
    DPCT1065:75: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    for (int idx = base_idx; idx < B * T; idx += warps_in_grid) {
        int b = idx / T;
        int t = idx % T;

        const floatX* dout_bt = dout + b * T * C + t * C;
        const floatX* inp_bt = inp + b * T * C + t * C;
        floatX* dinp_bt = dinp + b * T * C + t * C;
        const float mean_bt = (float)mean[b * T + t];
        const float rstd_bt = (float)rstd[b * T + t];

        // first: two reduce operations
        float dnorm_mean = 0.0f;
        float dnorm_norm_mean = 0.0f;
        for (int i = warpThreadIdx; i < C;
             i += sycl::ext::oneapi::this_work_item::get_sub_group()
                      .get_local_range()
                      .get(0)) {
            float norm_bti = ((float)inp_bt[i] - mean_bt) * rstd_bt;
            float dnorm_i = (float)weight[i] * (float)dout_bt[i];
            dnorm_mean += dnorm_i;
            dnorm_norm_mean += dnorm_i * norm_bti;
        }
        dnorm_mean = warpReduceSum(dnorm_mean);
        dnorm_norm_mean = warpReduceSum(dnorm_norm_mean);

        dnorm_mean = dnorm_mean / C;
        dnorm_norm_mean = dnorm_norm_mean / C;

        // now iterate again and accumulate all the gradients
        for (int i = warpThreadIdx; i < C;
             i += sycl::ext::oneapi::this_work_item::get_sub_group()
                      .get_local_range()
                      .get(0)) {
            /*
            DPCT1098:79: The '*' expression is used instead of the __ldcs call.
            These two expressions do not provide the exact same functionality.
            Check the generated code for potential precision and/or performance
            issues.
            */
            float dout_i = (float)dout_bt[i];
            /*
            DPCT1098:80: The '*' expression is used instead of the __ldcs call.
            These two expressions do not provide the exact same functionality.
            Check the generated code for potential precision and/or performance
            issues.
            */
            float norm_bti = ((float)inp_bt[i] - mean_bt) * rstd_bt;
            float dnorm_i = (float)weight[i] * dout_i;
            // gradient contribution to bias
            dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
                &dbias_shared[i], dout_i);
            // gradient contribution to weight
            dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
                &dweight_shared[i], norm_bti * dout_i);
            // gradient contribution to input
            float dval = 0.0f;
            dval += dnorm_i; // term 1
            dval -= dnorm_mean; // term 2
            dval -= norm_bti * dnorm_norm_mean; // term 3
            dval *= rstd_bt; // final scale
            dinp_bt[i] = (floatX)((float)dinp_bt[i] + dval);
        }
    }

    // Accumulate into a FP32 scratchpad
    // BF16 atomics are potentially much slower... and this is more precise!
    /*
    DPCT1065:76: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    float* scratch_dbias = scratch;
    float* scratch_dweight = scratch + C;
    unsigned int* scratchFlag = (unsigned int*)(scratch + (2 * C));
    for (int i = item_ct1.get_local_id(2); i < C;
         i += item_ct1.get_local_range(2)) {
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            &scratch_dbias[i], dbias_shared[i]);
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            &scratch_dweight[i], dweight_shared[i]);
    }
    /*
    DPCT1065:77: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (item_ct1.get_local_id(2) == 0) {
        *tmp_flag =
            dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
                scratchFlag, 1);
    }
    /*
    DPCT1065:78: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (*tmp_flag == item_ct1.get_group_range(2) - 1) {
        for (int i = item_ct1.get_local_id(2); i < C;
             i += item_ct1.get_local_range(2)) {
            // todo - potentially do stochastic rounding here as well
            dbias[i] = (floatX)scratch_dbias[i];
            dweight[i] = (floatX)scratch_dweight[i];
        }
    }
}

/*
DPCT1110:8: The total declared local variable size in device function
layernorm_backward_kernel8 exceeds 128 bytes and may cause high register
pressure. Consult with your hardware vendor to find the total register size
available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
void layernorm_backward_kernel8(floatX *dinp, floatX *dweight, floatX *dbias,
                                float *scratch, const floatX *dout,
                                const floatX *inp, const floatX *weight,
                                const floatX *mean, const floatX *rstd, int B,
                                int T, int C, uint8_t *dpct_local) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto shared = (float *)dpct_local; // size = 2 * C + 1
    int warpId = item_ct1.get_local_id(2) /
                 sycl::ext::oneapi::this_work_item::get_sub_group()
                     .get_local_range()
                     .get(0); // warp index within a block
    int warpsInBlock = item_ct1.get_local_range(2) /
                       sycl::ext::oneapi::this_work_item::get_sub_group()
                           .get_local_range()
                           .get(0); // number of warps in block
    int baseIdx = item_ct1.get_group(2) * warpsInBlock + warpId;
    int warpThreadIdx = item_ct1.get_local_id(2) %
                        sycl::ext::oneapi::this_work_item::get_sub_group()
                            .get_local_range()
                            .get(0); // Thread index within the warp
    int warpsInGrid = item_ct1.get_group_range(2) * warpsInBlock;
    int C_per_iteration = sycl::ext::oneapi::this_work_item::get_sub_group()
                              .get_local_range()
                              .get(0) *
                          x128::size;
    int iterations_C = C / C_per_iteration;

    // the first half of shared memory is bias, second is weight
    float* dbias_shared = shared;
    float* dweight_shared = shared + C;

    // init shared memory to zero
    for (int i = item_ct1.get_local_id(2); i < C;
         i += item_ct1.get_local_range(2)) {
       dbias_shared[i] = 0.0f;
       dweight_shared[i] = 0.0f;
    }
    unsigned int *tmp_flag = (unsigned int*)(shared + C*2);
    /*
    DPCT1065:81: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    for (int idx = baseIdx; idx < B * T; idx += warpsInGrid) {
        int b = idx / T;
        int t = idx % T;

        const floatX* dout_bt = dout + b * T * C + t * C;
        const floatX* inp_bt = inp + b * T * C + t * C;
        floatX* dinp_bt = dinp + b * T * C + t * C;
        const float mean_bt = (float)mean[b * T + t];
        const float rstd_bt = (float)rstd[b * T + t];

        // first: two reduce operations
        float dnorm_mean = 0.0f;
        float dnorm_norm_mean = 0.0f;
        for (int i = warpThreadIdx * x128::size; i < C;
             i += sycl::ext::oneapi::this_work_item::get_sub_group()
                      .get_local_range()
                      .get(0) *
                  x128::size) {
            x128 dout128_i   = load128(dout_bt + i);
            x128 inp128_i    = load128(inp_bt  + i);
            x128 weight128_i = load128(weight  + i);
            for (int k = 0; k < x128::size; k++) {
                float norm_bti = ((float)inp128_i[k] - mean_bt) * rstd_bt;
                float dnorm_i = (float)weight128_i[k] * (float)dout128_i[k];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
            }
        }
        dnorm_mean = warpReduceSum(dnorm_mean) / C;
        dnorm_norm_mean = warpReduceSum(dnorm_norm_mean) / C;

        // now iterate again and accumulate all the gradients
        // unfortunately we cannot use the same index for x128 arrays and shared memory
        // as atomics can only be 32-bit rather than 128-bit (at least pre-SM90/Hopper)
        // so this would result in an 8-way bank conflict, and kill performance
        // so instead, we use a shared memory friendly index, and reorder before the final write
        for (int i = 0; i < iterations_C; i++) {
            int global_index = (warpThreadIdx * x128::size) + (i * C_per_iteration);
            int shared_index = warpThreadIdx + (i * C_per_iteration);
            x128 dout128   = load128cs(dout_bt + global_index);
            x128 inp128    = load128cs(inp_bt  + global_index);
            x128 dinp128   = load128(dinp_bt   + global_index);
            x128 weight128 = load128(weight    + global_index);

            for (int x = 0; x < x128::size; x++) {
                float dout_i = (float)dout128[x];
                float norm_bti = ((float)inp128[x] - mean_bt) * rstd_bt;
                float dnorm_i = (float)weight128[x] * dout_i;
                // gradient contribution to bias (using shared memory friendly index)
                dpct::atomic_fetch_add<
                    sycl::access::address_space::generic_space>(
                    &dbias_shared
                        [shared_index +
                         x * sycl::ext::oneapi::this_work_item::get_sub_group()
                                 .get_local_range()
                                 .get(0)],
                    dout_i);
                // gradient contribution to weight (using shared memory friendly index)
                dpct::atomic_fetch_add<
                    sycl::access::address_space::generic_space>(
                    &dweight_shared
                        [shared_index +
                         x * sycl::ext::oneapi::this_work_item::get_sub_group()
                                 .get_local_range()
                                 .get(0)],
                    norm_bti * dout_i);
                // gradient contribution to input
                float dval = 0.0f;
                dval += dnorm_i; // term 1
                dval -= dnorm_mean; // term 2
                dval -= norm_bti * dnorm_norm_mean; // term 3
                dval *= rstd_bt; // final scale
                dinp128[x] = (floatX)((float)dinp128[x] + dval);
            }
            // cache in L2 as this is read by the next kernel, but bypass L1 to minimise thrashing
            store128cg(dinp_bt + global_index, dinp128);
        }
    }
    // Accumulate into a FP32 scratchpad
    // BF16 atomics are potentially much slower... and this is more precise!
    // todo - could potentially avoid the extra copy if floatX is FP32, fairly negligible though
    /*
    DPCT1065:82: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    float* scratch_dbias = scratch;
    float* scratch_dweight = scratch + C;
    unsigned int* scratchFlag = (unsigned int*)(scratch + (2 * C));
    for (int i = item_ct1.get_local_id(2); i < C;
         i += item_ct1.get_local_range(2)) {
        // global atomics in the same "shared memory banking friendly" order
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            &scratch_dbias[i], dbias_shared[i]);
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            &scratch_dweight[i], dweight_shared[i]);
    }
    /*
    DPCT1065:83: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (item_ct1.get_local_id(2) == 0) {
        *tmp_flag = dpct::atomic_fetch_compare_inc<
            sycl::access::address_space::generic_space>(
            scratchFlag, item_ct1.get_group_range(2));
    }
    /*
    DPCT1065:84: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (*tmp_flag == item_ct1.get_group_range(2) - 1) {
        for (int i = warpId; i < iterations_C; i += warpsInBlock) {
            // reorder from atomic/shared memory-friendly index to real global memory index
            // and convert from float/FP32 to floatX/BF16 for the final write
            int global_index = (warpThreadIdx * x128::size) + (i * C_per_iteration);
            int shared_index = warpThreadIdx + (i * C_per_iteration);

            x128 dbias128 = load128(dbias + global_index);
            x128 dweight128 = load128(dweight + global_index);
            for (int x = 0; x < x128::size; x++) {
                float s_db = scratch_dbias
                    [shared_index +
                     x * sycl::ext::oneapi::this_work_item::get_sub_group()
                             .get_local_range()
                             .get(0)];
                float s_dw = scratch_dweight
                    [shared_index +
                     x * sycl::ext::oneapi::this_work_item::get_sub_group()
                             .get_local_range()
                             .get(0)];
                dbias128[x] = (floatX)(s_db + (float)dbias128[x]);
                dweight128[x] = (floatX)(s_dw + (float)dweight128[x]);
            }
            store128(dbias + global_index, dbias128);
            store128(dweight + global_index, dweight128);
        }
    }
}

/*
DPCT1110:9: The total declared local variable size in device function
layernorm_backward_kernel9 exceeds 128 bytes and may cause high register
pressure. Consult with your hardware vendor to find the total register size
available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
void layernorm_backward_kernel9(floatX *dinp, floatX *dweight, floatX *dbias,
                                float *scratch, const floatX *dout,
                                const floatX *inp, const floatX *weight,
                                const floatX *mean, const floatX *rstd, int B,
                                int T, int C, const sycl::stream &stream_ct1,
                                uint8_t *dpct_local) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    if (C % (32 * x128::size) != 0) {
        if (item_ct1.get_local_id(2) == 0 && item_ct1.get_group(2) == 0) {
            stream_ct1
                << "Number of channels is not a multiple of 32 * x128::size";
        }
        assert(0); // prefer to crash here than run into a deadlock later on
    }
    int BLOCK_SIZE = item_ct1.get_local_range(2);
    int warpsInBlock = BLOCK_SIZE / WARP_SIZE; //number of warps in block
    auto shared = (float *)dpct_local;         // size = 2 * C + 1

    int warpId =
        item_ct1.get_local_id(2) / WARP_SIZE; // warp index within a block
    int baseIdx = item_ct1.get_group(2) * warpsInBlock + warpId;
    int warpThreadIdx =
        item_ct1.get_local_id(2) % WARP_SIZE; // Thread index within the warp
    int warpsInGrid = item_ct1.get_group_range(2) * warpsInBlock;
    int C_per_iteration = WARP_SIZE * x128::size;
    int iterations_C = ceil_div(C, C_per_iteration) + 2;

    // the first half of shared memory is bias, second is weight
    float* dbias_shared = shared;
    float* dweight_shared = shared + C;
    float* dbias_tmp_shared = shared + 2 * C;
    float* dweight_tmp_shared = shared + 2 * C + BLOCK_SIZE;

    // init shared memory to zero
    for (int i = item_ct1.get_local_id(2); i < C; i += BLOCK_SIZE) {
       dbias_shared[i] = 0.0f;
       dweight_shared[i] = 0.0f;
    }
    unsigned int *tmp_flag = (unsigned int*)(shared + 2*C + 2*BLOCK_SIZE);
    item_ct1.barrier(sycl::access::fence_space::local_space);

    for (int idx = baseIdx; idx < B * T; idx += warpsInGrid) {
        int b = idx / T;
        int t = idx % T;

        const floatX* dout_bt = dout + b * T * C + t * C;
        const floatX* inp_bt = inp + b * T * C + t * C;
        floatX* dinp_bt = dinp + b * T * C + t * C;
        const float mean_bt = (float)mean[b * T + t];
        const float rstd_bt = (float)rstd[b * T + t];

        // first: two reduce operations
        float dnorm_mean = 0.0f;
        float dnorm_norm_mean = 0.0f;
        for (int i = warpThreadIdx * x128::size; i < C; i += WARP_SIZE * x128::size) {
            x128 dout128_i   = load128(dout_bt + i);
            x128 inp128_i    = load128(inp_bt  + i);
            x128 weight128_i = load128(weight  + i);
            for (int k = 0; k < x128::size; k++) {
                float norm_bti = ((float)inp128_i[k] - mean_bt) * rstd_bt;
                float dnorm_i = (float)weight128_i[k] * (float)dout128_i[k];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
            }
        }
        dnorm_mean = warpReduceSum(dnorm_mean) / C;
        dnorm_norm_mean = warpReduceSum(dnorm_norm_mean) / C;

        // now iterate again and accumulate all the gradients
        // unfortunately we cannot use the same index for x128 arrays and shared memory
        // as atomics can only be 32-bit rather than 128-bit (at least pre-SM90/Hopper)
        // so this would result in an 8-way bank conflict, and kill performance
        // so instead, we use a shared memory friendly index, and reorder before the final write
        for (int i = 0; i < iterations_C; i++) {
            int global_index = (warpThreadIdx * x128::size) + (i * C_per_iteration);
            int shared_index = warpThreadIdx + (i * C_per_iteration);
            if (global_index >= C) {
                break;
            }

            x128 dout128   = load128cs(dout_bt + global_index);
            x128 inp128    = load128cs(inp_bt  + global_index);
            x128 dinp128   = load128(dinp_bt   + global_index);
            x128 weight128 = load128(weight    + global_index);

            for (int x = 0; x < x128::size; x++) {
                float dout_i = (float)dout128[x];
                float norm_bti = ((float)inp128[x] - mean_bt) * rstd_bt;
                float dnorm_i = (float)weight128[x] * dout_i;

                // sum up the gradients for bias and weight across the entire block
                // this is basically a reduction (but only inter-warp, not intra-warp)
                // doing it this way allows us to avoid using atomics while using many warps
                if (warpId != 0) {
                    dbias_tmp_shared[item_ct1.get_local_id(2)] = dout_i;
                    dweight_tmp_shared[item_ct1.get_local_id(2)] =
                        norm_bti * dout_i;
                }
                /*
                DPCT1118:10: SYCL group functions and algorithms must be
                encountered in converged control flow. You may need to adjust
                the code.
                */
                item_ct1.barrier(sycl::access::fence_space::local_space);
                if (warpId == 0) {
                    float dbias_tmp = dout_i;
                    float dweight_tmp = norm_bti * dout_i;
                    for (int j = 1; j < warpsInBlock; j++) {
                        dbias_tmp += dbias_tmp_shared[item_ct1.get_local_id(2) +
                                                      j * WARP_SIZE];
                        dweight_tmp +=
                            dweight_tmp_shared[item_ct1.get_local_id(2) +
                                               j * WARP_SIZE];
                    }
                    // gradient contribution to bias (using shared memory friendly index)
                    dbias_shared[shared_index + x*WARP_SIZE] += dbias_tmp;
                    // gradient contribution to weight (using shared memory friendly index)
                    dweight_shared[shared_index + x*WARP_SIZE] += dweight_tmp;
                }
                /*
                DPCT1118:11: SYCL group functions and algorithms must be
                encountered in converged control flow. You may need to adjust
                the code.
                */
                item_ct1.barrier(sycl::access::fence_space::local_space);

                // gradient contribution to input
                float dval = 0.0f;
                dval += dnorm_i; // term 1
                dval -= dnorm_mean; // term 2
                dval -= norm_bti * dnorm_norm_mean; // term 3
                dval *= rstd_bt; // final scale
                dinp128[x] = (floatX)((float)dinp128[x] + dval);
            }
            // cache in L2 as this is read by the next kernel, but bypass L1 to minimise thrashing
            store128cg(dinp_bt + global_index, dinp128);
        }
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    // Each block writes its partial sum to global memory
    // The last block to finish becomes responsible for summing up all the partial sums
    // This is done by atomically incrementing a flag (cleared to 0 before launching the kernel)
    unsigned int* scratchFlag = (unsigned int*)(scratch);
    // Increment scratch pointer by a full cacheline so that everything remains cacheline aligned
    scratch += 32;
    float* scratch_dbias = scratch;
    float* scratch_dweight = scratch + C;
    for (int i = item_ct1.get_local_id(2); i < C; i += BLOCK_SIZE) {
        // Write to global memory in the same "shared memory banking friendly" order
        scratch_dbias[i + 2 * C * item_ct1.get_group(2)] = dbias_shared[i];
        scratch_dweight[i + 2 * C * item_ct1.get_group(2)] = dweight_shared[i];
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    if (item_ct1.get_local_id(2) == 0) {
        *tmp_flag = dpct::atomic_fetch_compare_inc<
            sycl::access::address_space::generic_space>(
            scratchFlag, item_ct1.get_group_range(2));
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    if (*tmp_flag == item_ct1.get_group_range(2) - 1) {
        // Reduction of the partial sums by the final block
        // todo - there isn't enough parallelism even inside that single SM...
        // ==> so could maybe split into another kernel with YET ANOTHER level of reduction?!
        for (int i = item_ct1.get_local_id(2) * f128::size; i < C;
             i += BLOCK_SIZE * f128::size) {
            f128 dbias_accum = f128::zeros();
            f128 dweight_accum = f128::zeros();

            for (int read_block_idx = 0;
                 read_block_idx < item_ct1.get_group_range(2);
                 read_block_idx++) {
                int offset = i + 2*C*read_block_idx;
                f128 dbias128 = load128(scratch_dbias + offset);
                f128 dweight128 = load128(scratch_dweight + offset);
                for(int k = 0; k < f128::size; k++) {
                    dbias_accum[k] += dbias128[k];
                    dweight_accum[k] += dweight128[k];
                }
            }
            store128(dbias_shared + i, dbias_accum);
            store128(dweight_shared + i, dweight_accum);
        }
        /*
        DPCT1118:12: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        item_ct1.barrier(sycl::access::fence_space::local_space);

        // reorder from atomic/shared memory-friendly index to real global memory index
        // and convert from float/FP32 to floatX/BF16 for the final write
        // this is separate also because it cannot use as many warps as the above (f128 vs x128)
        // todo - if we split this code into another kernel, we could maybe do it at the same time?
        for (int i = warpId; i < iterations_C; i += warpsInBlock) {
            int global_index = (warpThreadIdx * x128::size) + (i * C_per_iteration);
            int shared_index = warpThreadIdx + (i * C_per_iteration);
            if (global_index >= C) {
                break;
            }

            x128 dbias128 = load128(dbias + global_index);
            x128 dweight128 = load128(dweight + global_index);
            for (int x = 0; x < x128::size; x++) {
                float s_db = dbias_shared[shared_index + x*WARP_SIZE];
                float s_dw = dweight_shared[shared_index + x*WARP_SIZE];
                dbias128[x] = (floatX)(s_db + (float)dbias128[x]);
                dweight128[x] = (floatX)(s_dw + (float)dweight128[x]);
            }
            store128(dbias + global_index, dbias128);
            store128(dweight + global_index, dweight128);
        }
    }
}


// similar to kernel 9, but uses vectors to access shared memory, which also avoids the bank conflict problems,
// and makes use require fewer barriers, at the cost of increased shared memory consumption.
// warning: this kernel is _extremely_ close to getting register spills, so many "optimizations" turn out to be unhelpful
// or need to be implemented in a very specific way.
/*
DPCT1110:13: The total declared local variable size in device function
layernorm_backward_kernel10 exceeds 128 bytes and may cause high register
pressure. Consult with your hardware vendor to find the total register size
available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
void layernorm_backward_kernel10(floatX *dinp, floatX *dweight, floatX *dbias,
                                 float *scratch, const floatX *dout,
                                 const floatX *inp, const floatX *weight,
                                 const floatX *mean, const floatX *rstd, int B,
                                 int T, int C, uint8_t *dpct_local) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int BLOCK_SIZE = item_ct1.get_local_range(2);
    int warpsInBlock = BLOCK_SIZE / WARP_SIZE; //number of warps in block
    auto shared = (float *)dpct_local;         // size = 2 * C + 1

    int warpId =
        item_ct1.get_local_id(2) / WARP_SIZE; // warp index within a block
    int baseIdx = item_ct1.get_group(2) * warpsInBlock + warpId;
    int warpThreadIdx =
        item_ct1.get_local_id(2) % WARP_SIZE; // Thread index within the warp
    int warpsInGrid = item_ct1.get_group_range(2) * warpsInBlock;
    int C_per_iteration = WARP_SIZE * x128::size;
    int iterations_C = ceil_div(C, C_per_iteration); // + 2;

    // the first half of shared memory is bias, second is weight
    size_t rounded_C = ceil_div(C, (32 * x128::size)) * (32 * x128::size);
    float* dbias_shared = shared;
    float* dweight_shared = shared + rounded_C;
    // warp zero doesn't actually write to the _tmp_shared memory locations, so we don't need to reserve memory
    // the obvious solution is to change the addressing below to use (threadId.x-32) as offset, but that causes
    // register spills, so instead we mess with the base pointer here, which doesn't increase register usage.
    float* dbias_tmp_shared = shared + 2 * rounded_C - WARP_SIZE * f128::size;
    float* dweight_tmp_shared = shared + 2 * rounded_C + f128::size * BLOCK_SIZE - 2 * WARP_SIZE * f128::size;

    // init shared memory to zero
    for (int i = item_ct1.get_local_id(2) * f128::size; i < rounded_C;
         i += BLOCK_SIZE * f128::size) {
        store128(dbias_shared + i, f128::zeros());
        store128(dweight_shared + i, f128::zeros());
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);

    for (int bt = baseIdx; bt < B * T; bt += warpsInGrid) {
        const floatX* dout_bt = dout + bt * C;
        const floatX* inp_bt = inp +bt * C;
        floatX* dinp_bt = dinp + bt * C;

        // first: two reduce operations
        float dnorm_mean = 0.0f;
        float dnorm_norm_mean = 0.0f;
        for (int i = warpThreadIdx * x128::size; i < C; i += WARP_SIZE * x128::size) {
            x128 dout128_i   = load128(dout_bt + i);
            x128 inp128_i    = load128(inp_bt  + i);
            x128 weight128_i = load128(weight  + i);
            for (int k = 0; k < x128::size; k++) {
                float dnorm_i = (float)weight128_i[k] * (float)dout128_i[k];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * (float)inp128_i[k];
            }
        }

        const float mean_bt = (float)mean[bt];
        const float rstd_bt = (float)rstd[bt];
        dnorm_mean = warpReduceSum(dnorm_mean) / C;
        dnorm_norm_mean = warpReduceSum(dnorm_norm_mean) / C * rstd_bt - dnorm_mean * mean_bt * rstd_bt;

        for (int c = 0; c < iterations_C; c++) {
            int global_index = (warpThreadIdx * x128::size) + (c * C_per_iteration);

            x128 dout128   = x128::zeros();
            x128 inp128    = x128::zeros();
            x128 dinp128   = x128::zeros();
            x128 weight128 = x128::zeros();

            if(global_index < C) {
                dout128 = load128cs(dout_bt + global_index);
                inp128 = load128cs(inp_bt + global_index);
                dinp128 = load128(dinp_bt + global_index);
                weight128 = load128(weight + global_index);
            }

            for(int o = 0; o < x128::size / f128::size; ++o) {
                f128 dbias_f;
                f128 dweight_f;
                for(int i = 0; i < f128::size; ++i) {
                    int x = o * f128::size + i;
                    float dout_i = (float)dout128[x];
                    float norm_bti = ((float)inp128[x] - mean_bt) * rstd_bt;
                    dbias_f[i] = dout_i;
                    dweight_f[i] = norm_bti * dout_i;

                    float dval = 0.0f;
                    dval += (float) weight128[x] * (float)dout128[x]; // term 1
                    dval -= dnorm_mean; // term 2
                    dval -= norm_bti * dnorm_norm_mean; // term 3
                    dval *= rstd_bt; // final scale
                    dinp128[x] = (floatX) ((float) dinp128[x] + dval);
                }

                if (warpId != 0) {
                    store128(dbias_tmp_shared +
                                 item_ct1.get_local_id(2) * f128::size,
                             dbias_f);
                    // this seems to generate a 64-bit store, instead of 128-bit.
                    // however, forcing 128-bit (e.g., using inline ptx), results in register
                    // spilling and much worse performance, so we'll keep it like this for now
                    // but ideally, we could reduce the register pressure a little.
                    store128(dweight_tmp_shared +
                                 item_ct1.get_local_id(2) * f128::size,
                             dweight_f);
                }
                /*
                DPCT1118:14: SYCL group functions and algorithms must be
                encountered in converged control flow. You may need to adjust
                the code.
                */
                item_ct1.barrier(sycl::access::fence_space::local_space);
                if (warpId == 0) {
                    for (int j = 1; j < warpsInBlock; j++) {
                        f128 dbias_tmp =
                            load128(dbias_tmp_shared +
                                    f128::size * (item_ct1.get_local_id(2) +
                                                  j * WARP_SIZE));
                        f128 dweight_tmp =
                            load128(dweight_tmp_shared +
                                    f128::size * (item_ct1.get_local_id(2) +
                                                  j * WARP_SIZE));
                        for(int i = 0; i < f128::size; ++i) {
                            dbias_f[i] += dbias_tmp[i];
                            dweight_f[i] += dweight_tmp[i];
                        }
                    }
                }
                /*
                DPCT1118:15: SYCL group functions and algorithms must be
                encountered in converged control flow. You may need to adjust
                the code.
                */
                item_ct1.barrier(sycl::access::fence_space::local_space);
                if (warpId == 0) {
                    f128 db_old = load128(dbias_shared + global_index + f128::size * o);
                    f128 dw_old = load128(dweight_shared + global_index + f128::size * o);
                    for(int i = 0; i < f128::size; ++i) {
                        dbias_f[i] += db_old[i];
                        dweight_f[i] += dw_old[i];
                    }
                    store128(dbias_shared + global_index + f128::size * o, dbias_f);
                    store128(dweight_shared + global_index + f128::size * o, dweight_f);
                }
            }
            if(global_index < C) {
                // cache in L2 as this is read by the next kernel, but bypass L1 to minimise thrashing
                store128cg(dinp_bt + global_index, dinp128);
            }
        }
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    // Each block writes its partial sum to global memory
    // The last block to finish becomes responsible for summing up all the partial sums
    // This is done by atomically incrementing a flag (cleared to 0 before launching the kernel)
    unsigned int* scratchFlag = (unsigned int*)(scratch);
    // Increment scratch pointer by a full cacheline so that everything remains cacheline aligned
    scratch += 32;
    float* scratch_dbias = scratch;
    float* scratch_dweight = scratch + C;
    for (int i = item_ct1.get_local_id(2) * f128::size; i < C;
         i += BLOCK_SIZE * f128::size) {
        // Write to global memory in the same "shared memory banking friendly" order
        store128(scratch_dbias + i + 2 * C * item_ct1.get_group(2),
                 load128(dbias_shared + i));
        store128(scratch_dweight + i + 2 * C * item_ct1.get_group(2),
                 load128(dweight_shared + i));
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    // that portion of shared memory is no longer used, so we can repurpose it for the scratch flag.
    unsigned int *tmp_flag = (unsigned int*)(shared + 2*rounded_C);
    if (item_ct1.get_local_id(2) == 0) {
        *tmp_flag = dpct::atomic_fetch_compare_inc<
            sycl::access::address_space::generic_space>(
            scratchFlag, item_ct1.get_group_range(2));
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    if (*tmp_flag == item_ct1.get_group_range(2) - 1) {
        // Reduction of the partial sums by the final block
        // todo - there isn't enough parallelism even inside that single SM...
        // ==> so could maybe split into another kernel with YET ANOTHER level of reduction?!
        for (int i = item_ct1.get_local_id(2) * f128::size; i < C;
             i += BLOCK_SIZE * f128::size) {
            f128 dbias_accum = f128::zeros();
            f128 dweight_accum = f128::zeros();

            for (int read_block_idx = 0;
                 read_block_idx < item_ct1.get_group_range(2);
                 read_block_idx++) {
                int offset = i + 2*C*read_block_idx;
                f128 dbias128 = load128(scratch_dbias + offset);
                f128 dweight128 = load128(scratch_dweight + offset);
                for(int k = 0; k < f128::size; k++) {
                    dbias_accum[k] += dbias128[k];
                    dweight_accum[k] += dweight128[k];
                }
            }
            store128(dbias_shared + i, dbias_accum);
            store128(dweight_shared + i, dweight_accum);
        }
        /*
        DPCT1118:16: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        item_ct1.barrier(sycl::access::fence_space::local_space);

        // convert from float/FP32 to floatX/BF16 for the final write
        // this is separate because it cannot use as many warps as the above (f128 vs x128)
        // todo - if we split this code into another kernel, we could maybe do it at the same time?
        for (int c = warpId; c < iterations_C; c += warpsInBlock) {
            int global_index = (warpThreadIdx * x128::size) + (c * C_per_iteration);
            if (global_index >= C) {
                break;
            }

            x128 dbias128 = load128(dbias + global_index);
            x128 dweight128 = load128(dweight + global_index);
            for(int o = 0; o < x128::size / f128::size; ++o) {
                f128 s_db = load128(dbias_shared + global_index + o * f128::size);
                f128 s_dw = load128(dweight_shared + global_index + o * f128::size);
                for(int i = 0; i < f128::size; ++i) {
                    int x = o * f128::size + i;
                    dbias128[x] = (floatX)(s_db[i] + (float)dbias128[x]);
                    dweight128[x] = (floatX)(s_dw[i] + (float)dweight128[x]);
                }
            }
            store128(dbias + global_index, dbias128);
            store128(dweight + global_index, dweight128);
        }
    }
}


// ----------------------------------------------------------------------------
// kernel launchers

void layernorm_backward1(float* dinp, float* dweight, float* dbias,
                        const float* dout, const float* inp, const float* weight, const float* mean, const float* rstd,
                        int B, int T, int C, const int block_size) {
    const int N = B * T;
    const int grid_size = ceil_div(N, block_size);
    /*
    DPCT1049:17: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
        dpct::get_in_order_queue().parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                                  sycl::range<3>(1, 1, block_size),
                              sycl::range<3>(1, 1, block_size)),
            [=](sycl::nd_item<3> item_ct1) {
                    layernorm_backward_kernel1(dinp, dweight, dbias, dout, inp,
                                               weight, mean, rstd, B, T, C);
            });
}

template <typename Tdinp, typename Tparams, typename Tdout, typename Trest>
void layernorm_backward2(Tdinp* dinp, Tparams* dweight, Tparams* dbias,
                        const Tdout* dout, const Trest* inp, const Tparams* weight, const Trest* mean, const Trest* rstd,
                        int B, int T, int C, int block_size) {
    const int N = B * T;
    const int grid_size = ceil_div(32*N, block_size);
    /*
    DPCT1083:19: The size of local memory in the migrated code may be different
    from the original code. Check that the allocated memory size in the migrated
    code is correct.
    */
    size_t shared_mem_size = 2 * C * sizeof(float);
    float* dweight_tmp;
    float* dbias_tmp;
    cudaCheck(DPCT_CHECK_ERROR(dweight_tmp = sycl::malloc_device<float>(
                                   C, dpct::get_in_order_queue())));
    cudaCheck(DPCT_CHECK_ERROR(
        dbias_tmp = sycl::malloc_device<float>(C, dpct::get_in_order_queue())));
    dpct::get_in_order_queue().memset(dweight_tmp, 0, C * sizeof(float)).wait();
    dpct::get_in_order_queue().memset(dbias_tmp, 0, C * sizeof(float)).wait();
    /*
    DPCT1049:18: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
        dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                    sycl::range<1>(shared_mem_size), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                                          sycl::range<3>(1, 1, block_size),
                                      sycl::range<3>(1, 1, block_size)),
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                                layernorm_backward_kernel2(
                                    dinp, dweight, dbias, dout, inp, weight,
                                    mean, rstd, B, T, C, dweight_tmp, dbias_tmp,
                                    dpct_local_acc_ct1
                                        .get_multi_ptr<
                                            sycl::access::decorated::no>()
                                        .get());
                        });
        });
    /*
    DPCT1049:20: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
        dpct::get_in_order_queue().parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, 512),
                              sycl::range<3>(1, 1, 512)),
            [=](sycl::nd_item<3> item_ct1) {
                    copy_to_dweight_dbias(C, dweight, dbias, dweight_tmp,
                                          dbias_tmp);
            });
    cudaCheck(DPCT_CHECK_ERROR(
        dpct::dpct_free(dweight_tmp, dpct::get_in_order_queue())));
    cudaCheck(DPCT_CHECK_ERROR(
        dpct::dpct_free(dbias_tmp, dpct::get_in_order_queue())));
}

template <typename Tdinp, typename Tparams, typename Tdout, typename Trest>
void layernorm_backward3(Tdinp* dinp, Tparams* dweight, Tparams* dbias,
                        const Tdout* dout, const Trest* inp, const Tparams* weight, const Trest* mean, const Trest* rstd,
                        int B, int T, int C, int block_size) {
    const int grid_size = (1024/block_size) * cuda_num_SMs;
    /*
    DPCT1083:22: The size of local memory in the migrated code may be different
    from the original code. Check that the allocated memory size in the migrated
    code is correct.
    */
    size_t shared_mem_size = 2 * C * sizeof(float);
    /*
    DPCT1049:21: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
        dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                    sycl::range<1>(shared_mem_size), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                                          sycl::range<3>(1, 1, block_size),
                                      sycl::range<3>(1, 1, block_size)),
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                                layernorm_backward_kernel3(
                                    dinp, dweight, dbias, dout, inp, weight,
                                    mean, rstd, B, T, C,
                                    dpct_local_acc_ct1
                                        .get_multi_ptr<
                                            sycl::access::decorated::no>()
                                        .get());
                        });
        });
}

template <typename Tdinp, typename Tparams, typename Tdout, typename Trest>
void layernorm_backward4(Tdinp* dinp, Tparams* dweight, Tparams* dbias,
                        const Tdout* dout, const Trest* inp, const Tparams* weight, const Trest* mean, const Trest* rstd,
                        int B, int T, int C, int block_size) {
        const int grid_size = (1024/block_size) * cuda_num_SMs;
        /*
        DPCT1083:24: The size of local memory in the migrated code may be
        different from the original code. Check that the allocated memory size
        in the migrated code is correct.
        */
        size_t shared_mem_size = 2 * C * sizeof(float);
        /*
        DPCT1049:23: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                    sycl::range<1>(shared_mem_size), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                                          sycl::range<3>(1, 1, block_size),
                                      sycl::range<3>(1, 1, block_size)),
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                                layernorm_backward_kernel4(
                                    dinp, dweight, dbias, dout, inp, weight,
                                    mean, rstd, B, T, C,
                                    dpct_local_acc_ct1
                                        .get_multi_ptr<
                                            sycl::access::decorated::no>()
                                        .get());
                        });
        });
}

template <typename Tdinp, typename Tparams, typename Tdout, typename Trest>
void layernorm_backward5(Tdinp* dinp, Tparams* dweight, Tparams* dbias, float* scratch,
                        const Tdout* dout, const Trest* inp, const Tparams* weight, const Trest* mean, const Trest* rstd,
                        int B, int T, int C, int block_size) {
        const int grid_size = 1 * cuda_num_SMs; // only support 1 block per SM for simplicity, 1024 threads is best anyway
        /*
        DPCT1083:26: The size of local memory in the migrated code may be
        different from the original code. Check that the allocated memory size
        in the migrated code is correct.
        */
        size_t shared_mem_size = (2 * C + 1) * sizeof(float);
        dpct::get_in_order_queue()
            .memset(scratch, 0, (grid_size * 2 * C + 1) * sizeof(float))
            .wait();
        /*
        DPCT1049:25: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                    sycl::range<1>(shared_mem_size), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                                          sycl::range<3>(1, 1, block_size),
                                      sycl::range<3>(1, 1, block_size)),
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                                layernorm_backward_kernel5(
                                    dinp, dweight, dbias, scratch, dout, inp,
                                    weight, mean, rstd, B, T, C,
                                    dpct_local_acc_ct1
                                        .get_multi_ptr<
                                            sycl::access::decorated::no>()
                                        .get());
                        });
        });
}

template <typename Tdinp, typename Tparams, typename Tdout, typename Trest>
void layernorm_backward6(Tdinp* dinp, Tparams* dweight, Tparams* dbias, float* scratch,
                        const Tdout* dout, const Trest* inp, const Tparams* weight, const Trest* mean, const Trest* rstd,
                        int B, int T, int C, int block_size) {
        const int grid_size = (1024/block_size) * cuda_num_SMs;
        /*
        DPCT1083:28: The size of local memory in the migrated code may be
        different from the original code. Check that the allocated memory size
        in the migrated code is correct.
        */
        size_t shared_mem_size = (2 * C + 1) * sizeof(float);

        // Including this as part of the timing until we can parallelise it
        // It should fully hide the cost and improve kernel perf by >5% if done in parallel using CUDA streams
        dpct::get_in_order_queue()
            .memset(scratch, 0, (1 + 2 * C) * sizeof(float))
            .wait();

        /*
        DPCT1049:27: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                    sycl::range<1>(shared_mem_size), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                                          sycl::range<3>(1, 1, block_size),
                                      sycl::range<3>(1, 1, block_size)),
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                                layernorm_backward_kernel6(
                                    dinp, dweight, dbias, scratch, dout, inp,
                                    weight, mean, rstd, B, T, C,
                                    dpct_local_acc_ct1
                                        .get_multi_ptr<
                                            sycl::access::decorated::no>()
                                        .get());
                        });
        });
}

template <typename Tdinp, typename Tparams, typename Tdout, typename Trest>
void layernorm_backward7(Tdinp* dinp, Tparams* dweight, Tparams* dbias, float* scratch,
                        const Tdout* dout, const Trest* inp, const Tparams* weight, const Trest* mean, const Trest* rstd,
                        int B, int T, int C, int block_size) {
        const int grid_size = (1024/block_size) * cuda_num_SMs;
        /*
        DPCT1083:30: The size of local memory in the migrated code may be
        different from the original code. Check that the allocated memory size
        in the migrated code is correct.
        */
        size_t shared_mem_size = (2 * C + 1) * sizeof(float);

        // Including this as part of the timing until we can parallelise it
        // It should fully hide the cost and improve kernel perf by >5% if done in parallel using CUDA streams
        dpct::get_in_order_queue()
            .memset(scratch, 0, (1 + 2 * C) * sizeof(float))
            .wait();

        /*
        DPCT1049:29: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                    sycl::range<1>(shared_mem_size), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                                          sycl::range<3>(1, 1, block_size),
                                      sycl::range<3>(1, 1, block_size)),
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                                layernorm_backward_kernel7(
                                    dinp, dweight, dbias, scratch, dout, inp,
                                    weight, mean, rstd, B, T, C,
                                    dpct_local_acc_ct1
                                        .get_multi_ptr<
                                            sycl::access::decorated::no>()
                                        .get());
                        });
        });
}

template <typename Tdinp, typename Tparams, typename Tdout, typename Trest>
void layernorm_backward8(Tdinp* dinp, Tparams* dweight, Tparams* dbias, float* scratch,
                        const Tdout* dout, const Trest* inp, const Tparams* weight, const Trest* mean, const Trest* rstd,
                        int B, int T, int C, int block_size) {
        const int grid_size = (1024/block_size) * cuda_num_SMs;
        /*
        DPCT1083:32: The size of local memory in the migrated code may be
        different from the original code. Check that the allocated memory size
        in the migrated code is correct.
        */
        size_t shared_mem_size = (2 * C + 1) * sizeof(float);

        // Including this as part of the timing until we can parallelise it
        // It should fully hide the cost and improve kernel perf by >5% if done in parallel using CUDA streams
        dpct::get_in_order_queue()
            .memset(scratch, 0, (1 + 2 * C) * sizeof(float))
            .wait();

        /*
        DPCT1049:31: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                    sycl::range<1>(shared_mem_size), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                                          sycl::range<3>(1, 1, block_size),
                                      sycl::range<3>(1, 1, block_size)),
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                                layernorm_backward_kernel8(
                                    dinp, dweight, dbias, scratch, dout, inp,
                                    weight, mean, rstd, B, T, C,
                                    dpct_local_acc_ct1
                                        .get_multi_ptr<
                                            sycl::access::decorated::no>()
                                        .get());
                        });
        });
}

template <typename Tdinp, typename Tparams, typename Tdout, typename Trest>
void layernorm_backward9(Tdinp* dinp, Tparams* dweight, Tparams* dbias, float* scratch,
                        const Tdout* dout, const Trest* inp, const Tparams* weight, const Trest* mean, const Trest* rstd,
                        int B, int T, int C, int block_size) {

        assert(C % (32 * x128::size) == 0  && "Channels must be divisible by (32 * x128::size)");
        const int grid_size = (1024/block_size) * cuda_num_SMs; // todo - heuristics for other GPUs?
        /*
        DPCT1083:34: The size of local memory in the migrated code may be
        different from the original code. Check that the allocated memory size
        in the migrated code is correct.
        */
        size_t shared_mem_size = (2 * C + 2 * block_size + 1) * sizeof(float);

        dpct::get_in_order_queue()
            .memset(scratch, 0, 1 * sizeof(float))
            .wait(); // just need to memset the flag for this version
        /*
        DPCT1049:33: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
                sycl::stream stream_ct1(64 * 1024, 80, cgh);

                sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                    sycl::range<1>(shared_mem_size), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                                          sycl::range<3>(1, 1, block_size),
                                      sycl::range<3>(1, 1, block_size)),
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                                layernorm_backward_kernel9(
                                    dinp, dweight, dbias, scratch, dout, inp,
                                    weight, mean, rstd, B, T, C, stream_ct1,
                                    dpct_local_acc_ct1
                                        .get_multi_ptr<
                                            sycl::access::decorated::no>()
                                        .get());
                        });
        });
}

template <typename Tdinp, typename Tparams, typename Tdout, typename Trest>
void layernorm_backward10(Tdinp* dinp, Tparams* dweight, Tparams* dbias, float* scratch,
                         const Tdout* dout, const Trest* inp, const Tparams* weight, const Trest* mean, const Trest* rstd,
                         int B, int T, int C, int block_size) {
        if(block_size == 1024) {
            block_size = 512;
        }
        //assert(C % (32 * x128::size) == 0  && "Channels must be divisible by (32 * x128::size)");
        const int grid_size = (1024/block_size) * cuda_num_SMs; // todo - heuristics for other GPUs?
        size_t rounded_C = ceil_div(C, (32 * x128::size)) * (32 * x128::size);
        /*
        DPCT1083:36: The size of local memory in the migrated code may be
        different from the original code. Check that the allocated memory size
        in the migrated code is correct.
        */
        size_t shared_mem_size =
            (2 * rounded_C + 2 * (block_size - 32) * f128::size) *
            sizeof(float);

        cudaCheck(DPCT_CHECK_ERROR(
            dpct::get_in_order_queue()
                .memset(scratch, 0, 1 * sizeof(float))
                .wait())); // just need to memset the flag for this version
        /*
        DPCT1049:35: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                    sycl::range<1>(shared_mem_size), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                                          sycl::range<3>(1, 1, block_size),
                                      sycl::range<3>(1, 1, block_size)),
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                                layernorm_backward_kernel10(
                                    dinp, dweight, dbias, scratch, dout, inp,
                                    weight, mean, rstd, B, T, C,
                                    dpct_local_acc_ct1
                                        .get_multi_ptr<
                                            sycl::access::decorated::no>()
                                        .get());
                        });
        });
        /*
        DPCT1010:85: SYCL uses exceptions to report errors and does not use the
        error codes. The cudaGetLastError function call was replaced with 0. You
        need to rewrite this code.
        */
        cudaCheck(0);
}

// kernel version dispatch
void layernorm_backward(int kernel_num,
                        floatX* dinp, floatX* dweight, floatX* dbias, float* scratch,
                        const floatX* dout, const floatX* inp, const floatX* weight, const floatX* mean, const floatX* rstd,
                        int B, int T, int C,
                        const int block_size) {
    switch (kernel_num) {
#if !defined(ENABLE_BF16) && !defined(ENABLE_FP16)
        case 1:
            layernorm_backward1(dinp, dweight, dbias, dout, inp, weight, mean, rstd, B, T, C, block_size);
            break;
#endif
        case 2:
            layernorm_backward2(dinp, dweight, dbias, dout, inp, weight, mean, rstd, B, T, C, block_size);
            break;
        case 3:
            layernorm_backward3(dinp, dweight, dbias, dout, inp, weight, mean, rstd, B, T, C, block_size);
            break;
#if defined(ENABLE_BF16)
        case 4:
            layernorm_backward4(dinp, dweight, dbias, dout, inp, weight, mean, rstd, B, T, C, block_size);
            break;
#endif
        case 5:
            layernorm_backward5(dinp, dweight, dbias, scratch, dout, inp, weight, mean, rstd, B, T, C, block_size);
            break;
        case 6:
            layernorm_backward6(dinp, dweight, dbias, scratch, dout, inp, weight, mean, rstd, B, T, C, block_size);
            break;
        case 7:
            layernorm_backward7(dinp, dweight, dbias, scratch, dout, inp, weight, mean, rstd, B, T, C, block_size);
            break;
        case 8:
            layernorm_backward8(dinp, dweight, dbias, scratch, dout, inp, weight, mean, rstd, B, T, C, block_size);
            break;
        case 9:
            layernorm_backward9(dinp, dweight, dbias, scratch, dout, inp, weight, mean, rstd, B, T, C, block_size);
            break;
        case 10:
            layernorm_backward10(dinp, dweight, dbias, scratch, dout, inp, weight, mean, rstd, B, T, C, block_size);
            break;
    default:
            printf("Invalid kernel number\n");
            exit(1);
    }
    /*
    DPCT1010:86: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    cudaCheck(0);
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
    float* dout = (float*)malloc(sizeof(float) * (size_t)(B*T*C));
    if (fread(dout, sizeof(float), (size_t)(B*T*C), fin) != (size_t)(B*T*C)) {
        fprintf(stderr, "input too short at dout\n"); return 2;
    }
    fclose(fin);
    FILE* fout = fopen(argv[2], "wb");
    if (!fout) { perror(argv[2]); return 2; }
    float* ln_out = (float*)malloc(sizeof(float) * (size_t)B * T * C); float* mean = (float*)malloc(sizeof(float) * (size_t)B * T); float* rstd = (float*)malloc(sizeof(float) * (size_t)B * T); layernorm_forward_cpu(ln_out, mean, rstd, inp, weight, bias, B, T, C);
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
    float* d_dout;
    cudaCheck(
        DPCT_CHECK_ERROR(d_dout = sycl::malloc_device<float>(
                             (size_t)(B * T * C), dpct::get_in_order_queue())));
    cudaCheck(DPCT_CHECK_ERROR(
        dpct::get_in_order_queue()
            .memcpy(d_dout, dout, sizeof(float) * (size_t)(B * T * C))
            .wait()));
    float* dinp = (float*)malloc(sizeof(float) * (size_t)(B*T*C));
    float* d_dinp;
    cudaCheck(
        DPCT_CHECK_ERROR(d_dinp = sycl::malloc_device<float>(
                             (size_t)(B * T * C), dpct::get_in_order_queue())));
    float* dweight = (float*)malloc(sizeof(float) * (size_t)(C));
    float* d_dweight;
    cudaCheck(DPCT_CHECK_ERROR(d_dweight = sycl::malloc_device<float>(
                                   (size_t)(C), dpct::get_in_order_queue())));
    float* dbias = (float*)malloc(sizeof(float) * (size_t)(C));
    float* d_dbias;
    cudaCheck(DPCT_CHECK_ERROR(d_dbias = sycl::malloc_device<float>(
                                   (size_t)(C), dpct::get_in_order_queue())));
    float *d_mean; float * d_rstd; float * d_scratch;
        cudaCheck(DPCT_CHECK_ERROR(
            d_mean = (float *)sycl::malloc_device(sizeof(float) * (size_t)B * T,
                                                  dpct::get_in_order_queue())));
        cudaCheck(DPCT_CHECK_ERROR(
            d_rstd = (float *)sycl::malloc_device(sizeof(float) * (size_t)B * T,
                                                  dpct::get_in_order_queue())));
        cudaCheck(DPCT_CHECK_ERROR(
            dpct::get_in_order_queue()
                .memcpy(d_mean, mean, sizeof(float) * (size_t)B * T)
                .wait()));
        cudaCheck(DPCT_CHECK_ERROR(
            dpct::get_in_order_queue()
                .memcpy(d_rstd, rstd, sizeof(float) * (size_t)B * T)
                .wait()));
        cudaCheck(DPCT_CHECK_ERROR(d_scratch = sycl::malloc_device<float>(
                                       (1024 / 32) * cuda_num_SMs * (2 * C + 1),
                                       dpct::get_in_order_queue())));
    const int variants[] = {1, 2, 3, 5, 6, 7, 8, 9, 10};
    for (int vi = 0; vi < 9; vi++) {
        const int k = variants[vi];
        cudaCheck(DPCT_CHECK_ERROR(
            dpct::get_in_order_queue()
                .memset(d_dinp, 0, sizeof(float) * (size_t)(B * T * C))
                .wait()));
        cudaCheck(DPCT_CHECK_ERROR(
            dpct::get_in_order_queue()
                .memset(d_dweight, 0, sizeof(float) * (size_t)(C))
                .wait()));
        cudaCheck(DPCT_CHECK_ERROR(
            dpct::get_in_order_queue()
                .memset(d_dbias, 0, sizeof(float) * (size_t)(C))
                .wait()));
        layernorm_backward(k, d_dinp, d_dweight, d_dbias, d_scratch, d_dout, d_inp, d_weight, d_mean, d_rstd, B, T, C, 256);
        /*
        DPCT1010:87: SYCL uses exceptions to report errors and does not use the
        error codes. The cudaGetLastError function call was replaced with 0. You
        need to rewrite this code.
        */
        cudaCheck(0);
        cudaCheck(DPCT_CHECK_ERROR(
            dpct::get_current_device().queues_wait_and_throw()));
        cudaCheck(DPCT_CHECK_ERROR(
            dpct::get_in_order_queue()
                .memcpy(dinp, d_dinp, sizeof(float) * (size_t)(B * T * C))
                .wait()));
        fwrite(dinp, sizeof(float), (size_t)(B*T*C), fout);
        cudaCheck(DPCT_CHECK_ERROR(
            dpct::get_in_order_queue()
                .memcpy(dweight, d_dweight, sizeof(float) * (size_t)(C))
                .wait()));
        fwrite(dweight, sizeof(float), (size_t)(C), fout);
        cudaCheck(DPCT_CHECK_ERROR(
            dpct::get_in_order_queue()
                .memcpy(dbias, d_dbias, sizeof(float) * (size_t)(C))
                .wait()));
        fwrite(dbias, sizeof(float), (size_t)(C), fout);
    }
    fclose(fout);
    return 0;
}
