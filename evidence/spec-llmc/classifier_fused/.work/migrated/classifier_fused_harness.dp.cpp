/*  Kernels for fused forward/backward classifier part
This fuses softmax, crossentropy, and logit gradients into a single pass, so we don't have to write unnecessary
(B, T, V) tensors. Such an operation is only possible if `dloss` can be known beforehand, which doesn't seem like
much of a restriction: In pretraining, it is just a constant 1/batch_size tensor, for fine-tuning we might zero
out the input prompt, but that is known in advance.

Compile example:
nvcc -O3 --use_fast_math -lcublas -lcublasLt classifier_fused.cu -o classifier_fused

./classifier_fused 1
./classifier_fused 2
./classifier_fused 3
./classifier_fused 4
*/

#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include "common.h"

// todo - this file does not properly support anything but FP32
// kernel 5 can be run in fp16/bf16 to test performance, but the outputs will be wrong
#if defined(ENABLE_BF16)
typedef __nv_bfloat16 floatX;
#elif defined(ENABLE_FP16)
typedef half floatX;
#else
typedef float floatX;
#endif
typedef Packed128<floatX> x128;

// ----------------------------------------------------------------------------
// CPU code reference

// museport: CPU reference softmax_forward_cpu() removed (it is the hidden check)


// museport: CPU reference crossentropy_forward_cpu() removed (it is the hidden check)

// museport: CPU reference crossentropy_softmax_backward_cpu() removed (it is the hidden check)

// ----------------------------------------------------
// Kernel Utils

// warp-level reduction for finding the maximum value
SYCL_EXTERNAL float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        /*
        DPCT1096:35: The right-most dimension of the work-group used in the SYCL
        kernel that calls this function may be less than "32". The function
        "dpct::permute_sub_group_by_xor" may return an unexpected result on the
        CPU device. Modify the size of the work-group to ensure that the value
        of the right-most dimension is a multiple of "32".
        */
        val = sycl::fmax(val,
                         dpct::permute_sub_group_by_xor(
                             sycl::ext::oneapi::this_work_item::get_sub_group(),
                             val, offset));
    }
    return val;
}

// ----------------------------------------------------------------------------
// GPU kernels

struct SoftmaxParams {
    float Scale;
    float Offset;
};

SoftmaxParams prepare_softmax(sycl::sub_group &warp, int64_t idx,
                              const float *inp, int V, int P) {
    // this warp (of 32) threads processes one row of inp, i.e. inp[idx, :] of shape (V,)
    // note that inp is actually (B * T, P) but we only use the first V elements
    // this function then calculates:
    // 1) the max value to subtract for numerical stability and
    // 2) the sum normalization factor
    const float* x = inp + idx * P;
    // thread coarsening loop, where the 32 threads serially process all V elements
    // thread_rank() is in [0, 31], warp.size() is 32
    float maxval = -INFINITY;
    float sumval = 0.0f;
    for (int i = sycl::ext::oneapi::this_work_item::get_sub_group()
                     .get_local_linear_id();
         i < V; i += sycl::ext::oneapi::this_work_item::get_sub_group()
                         .get_local_linear_range()) {
        float v = x[i];
        float old_maxval = maxval;
        // online softmax recurrence from "Online normalizer calculation for softmax" paper
        maxval = sycl::fmax(maxval, v);
        sumval *= sycl::native::exp((old_maxval - maxval));
        sumval += sycl::native::exp(v - maxval);
    }
    // warp-level reduction to get the maxval across the 32 threads
    float global_maxval = sycl::reduce_over_group(
        sycl::ext::oneapi::this_work_item::get_sub_group(), maxval,
        sycl::maximum<float>{});
    // all 32 threads do a final shift of the sum considering the global max in this row
    sumval *= sycl::native::exp((maxval - global_maxval));
    // warp-level reduction to get the sumval across the 32 threads
    float global_sumval = sycl::reduce_over_group(
        sycl::ext::oneapi::this_work_item::get_sub_group(), sumval,
        sycl::plus<float>{});
    // the final normalization factor
    float norm = 1.0f / global_sumval;
    return SoftmaxParams{norm, global_maxval};
}

void fused_classifier_kernel1(float* dlogits, float* losses,
                             const float* logits, const float* dlosses, const int* targets,
                             int B, int T, int V, int P) {

    sycl::group<3> block =
        sycl::ext::oneapi::this_work_item::get_work_group<3>();
    sycl::sub_group warp = sycl::ext::oneapi::this_work_item::get_sub_group();
    // example: B = 4, T = 1024, block_size = 128 => we'd have grid_size = 1024
    // each block of 4 warps is in charge of 4 rows of the input, one warp per row
    // meta_group_size is the number of warps per block (e.g. 4)
    // meta_group_rank is the index of the warp in the block (e.g. 0, 1, 2, 3)
    int64_t idx =
        sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_group(2) *
            sycl::ext::oneapi::this_work_item::get_sub_group()
                .get_group_linear_range() +
        sycl::ext::oneapi::this_work_item::get_sub_group()
            .get_group_linear_id();
    if (idx >= B * T) { // there are B * T rows in the input
        return;
    }
    int b = idx / T;
    int t = idx % T;

    // calculate the offset (maxval) and scale (sumval) for the softmax
    SoftmaxParams sp = prepare_softmax(warp, idx, logits, V, P);

    // in each row (handled by one warp), thread 0 calculates the loss
    // calculate the probability needed for the loss and update losses
    if (sycl::ext::oneapi::this_work_item::get_sub_group()
            .get_local_linear_id() == 0) {
        int ix = targets[b * T + t];
        float prob =
            sycl::native::exp(logits[idx * P + ix] - sp.Offset) * sp.Scale;
        losses[b * T + t] = -sycl::log(prob);
    }

    // finally all threads calculate the gradients
    // prob is only materialized here temporarily and in registers, never
    // as a full tensor that gets written to global memory
    for (int i = sycl::ext::oneapi::this_work_item::get_sub_group()
                     .get_local_linear_id();
         i < V; i += sycl::ext::oneapi::this_work_item::get_sub_group()
                         .get_local_linear_range()) {
        float prob =
            sycl::native::exp(logits[idx * P + i] - sp.Offset) * sp.Scale;
        float* dlogits_bt = dlogits + b * T * P + t * P;
        float dloss = dlosses[b * T + t];
        int ix = targets[b * T + t];
        float indicator = i == ix ? 1.0f : 0.0f;
        dlogits_bt[i] = (prob - indicator) * dloss;
    }
}

float vec_at(const sycl::float4 &vec, int index) {
    return reinterpret_cast<const float*>(&vec)[index];
}

SoftmaxParams prepare_softmax_blockwide(sycl::sub_group &warp, int64_t idx,
                                        const float *inp, int V, int P,
                                        float *shared_maxval,
                                        float *shared_sumval) {
    // one row of inp, i.e. inp[idx, :] of shape (V,)
    // float4 to get 128-bit loads and memory level parallelism
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const sycl::float4 *x_vec4 =
        reinterpret_cast<const sycl::float4 *>(inp + idx * P);

    float thread_maxval = -INFINITY;
    float thread_sumval = 0.0f;
    // do the loop in reverse to maximise probability of L2 cache hits
    // so even small L2s get some hits on the 2nd read of the same thread
    for (int i = ceil_div(V, 4) + item_ct1.get_local_id(2) -
                 item_ct1.get_local_range(2);
         i >= 0; i -= item_ct1.get_local_range(2)) {
        sycl::float4 v4 = x_vec4[i];
#pragma unroll
        for(int k = 0; k < 4; k++) {
            if (i*4+k >= V) {  // bounds checking against real V
                continue;
            }
            float old_maxval = thread_maxval;
            thread_maxval = sycl::fmax(thread_maxval, vec_at(v4, k));
            thread_sumval *= sycl::native::exp(old_maxval - thread_maxval);
            thread_sumval += sycl::native::exp(vec_at(v4, k) - thread_maxval);
        }
    }

    // two reductions of up to 1024 threads:
    // 1) inside warp (shuffle), 2) cross-warp (shared memory), 3) inside warp (shuffle)
    // this results in much cleaner assembly than a multi-warp cg::reduce

    int num_warps = item_ct1.get_local_range(2) / 32;
    int warp_id = item_ct1.get_local_id(2) / 32;
    int lane_id = item_ct1.get_local_id(2) % 32;

    // reduce maxval within each warp
    float warp_maxval = sycl::reduce_over_group(
        sycl::ext::oneapi::this_work_item::get_sub_group(), thread_maxval,
        sycl::maximum<float>{});
    // thread 0 in each warp writes to shared memory
    if (lane_id == 0) { shared_maxval[warp_id] = warp_maxval; }
    /*
    DPCT1065:20: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    // each thread now loads the maxval across previous warps
    // if the thread is "out of range" of data, use -FLT_MAX as the maxval
    warp_maxval = (lane_id < num_warps) ? shared_maxval[lane_id] : -FLT_MAX;
    // now reduce the maxval among the warp threads
    float block_maxval = sycl::reduce_over_group(
        sycl::ext::oneapi::this_work_item::get_sub_group(), warp_maxval,
        sycl::maximum<float>{});
    // each thread uses maxval to scale sumval to avoid numerical instability / overflow
    thread_sumval *= sycl::native::exp(thread_maxval - block_maxval);
    // (warp-level) reduce sumval, thread 0 in each warp saves result in shared memory
    float warp_sumval = sycl::reduce_over_group(
        sycl::ext::oneapi::this_work_item::get_sub_group(), thread_sumval,
        sycl::plus<float>{});
    if (lane_id == 0) { shared_sumval[warp_id] = warp_sumval; }
    /*
    DPCT1065:21: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    // same strategy, now reduce sumval across warps
    warp_sumval = (lane_id < num_warps) ? shared_sumval[lane_id] : 0.0f;
    float block_sumval = sycl::reduce_over_group(
        sycl::ext::oneapi::this_work_item::get_sub_group(), warp_sumval,
        sycl::plus<float>{});
    // return the softmax parameters
    return SoftmaxParams{1.f / block_sumval, block_maxval};
}

// Fused forward and backward pass for classifier including softmax, and logit gradients
// Writes to both probs (only for debugging) and dlogits (only for training) are optional
// N.B.: We may want to reuse the logits memory for dlogits, so they should *not* be __restrict__!
void fused_classifier_kernel2(float* dlogits, float* losses, float* probs,
                                         const float* logits, const float* dlosses, const int* targets,
                                         int B, int T, int V, int P,
                                         float *shared_maxval,
                                         float *shared_sumval) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    sycl::group<3> block =
        sycl::ext::oneapi::this_work_item::get_work_group<3>();
    sycl::sub_group warp = sycl::ext::oneapi::this_work_item::get_sub_group();
    int64_t idx = item_ct1.get_group(2);
    int ix = targets[idx];

    // softmax (reading B * T * V, same logits read again below, hopefully still in cache)
    SoftmaxParams sp = prepare_softmax_blockwide(warp, idx, logits, V, P,
                                                 shared_maxval, shared_sumval);

    // calculate the probability needed for the loss and update (single-threaded)
    if (item_ct1.get_local_id(2) == 0) {
        float prob =
            sycl::native::exp(logits[idx * P + ix] - sp.Offset) * sp.Scale;
        losses[idx] = -sycl::log(prob);
    }

    // very sensible default for dlosses is 1/(B*T), which is the uniform loss
    float dloss = dlosses != NULL ? dlosses[idx] : 1.0f / (B*T);
    // calculate the gradients directly, saves bandwidth from probs during training
    // but also supports writing probs for inference-only and debugging
    const sycl::float4 *logits_vec4 =
        reinterpret_cast<const sycl::float4 *>(logits + idx * P);
    for (int i = item_ct1.get_local_id(2); i < ceil_div(V, 4);
         i += item_ct1.get_local_range(2)) {
        // this is the 2nd read of logits after the one in prepare_softmax2
        // this data will never be needed again, so we reduce cache persistence
        /*
        DPCT1098:22: The '*' expression is used instead of the __ldcs call.
        These two expressions do not provide the exact same functionality. Check
        the generated code for potential precision and/or performance issues.
        */
        sycl::float4 v4 = logits_vec4[i];

#pragma unroll
        for(int k = 0; k < 4; ++k) {
            int element = i*4 + k;
            float prob =
                sycl::native::exp(vec_at(v4, k) - sp.Offset) * sp.Scale;
            prob = (element < V) ? prob : 0.0f; // bounds checking against real V

            // this kernel is DRAM limited so cost of inner branch is ~zero
            if (probs != NULL) {
                probs[idx * P + element] = prob;
            }
            if (dlogits != NULL) {
                float indicator = element == ix ? 1.0f : 0.0f;
                dlogits[idx * P + element] = (prob - indicator) * dloss;
            }
        }
    }
}

SoftmaxParams prepare_softmax_blockwide_nofloat4(sycl::sub_group &warp,
                                                 int64_t idx, const float *inp,
                                                 int V, int P,
                                                 float *shared_maxval,
                                                 float *shared_sumval) {
    // same but not float4
    // one row of inp, i.e. inp[idx, :] of shape (V,)

    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const float *x = inp + idx * P;
    float thread_maxval = -INFINITY;
    float thread_sumval = 0.0f;
    // do the loop in reverse to maximise probability of L2 cache hits
    // so even small L2s get some hits on the 2nd read of the same thread
    for (int i = V + item_ct1.get_local_id(2) - item_ct1.get_local_range(2);
         i >= 0; i -= item_ct1.get_local_range(2)) {
        float v = x[i];
        float old_maxval = thread_maxval;
        thread_maxval = sycl::fmax(thread_maxval, v);
        thread_sumval *= sycl::native::exp(old_maxval - thread_maxval);
        thread_sumval += sycl::native::exp(v - thread_maxval);
    }

    // two reductions of up to 1024 threads:
    // 1) inside warp (shuffle), 2) cross-warp (shared memory), 3) inside warp (shuffle)
    // this results in much cleaner assembly than a multi-warp cg::reduce

    int num_warps = item_ct1.get_local_range(2) / 32;
    int warp_id = item_ct1.get_local_id(2) / 32;
    int lane_id = item_ct1.get_local_id(2) % 32;

    // reduce maxval within each warp
    float warp_maxval = sycl::reduce_over_group(
        sycl::ext::oneapi::this_work_item::get_sub_group(), thread_maxval,
        sycl::maximum<float>{});
    // thread 0 in each warp writes to shared memory
    if (lane_id == 0) { shared_maxval[warp_id] = warp_maxval; }
    /*
    DPCT1065:23: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    // each thread now loads the maxval across previous warps
    // if the thread is "out of range" of data, use -FLT_MAX as the maxval
    warp_maxval = (lane_id < num_warps) ? shared_maxval[lane_id] : -FLT_MAX;
    // now reduce the maxval among the warp threads
    float block_maxval = sycl::reduce_over_group(
        sycl::ext::oneapi::this_work_item::get_sub_group(), warp_maxval,
        sycl::maximum<float>{});
    // each thread uses maxval to scale sumval to avoid numerical instability / overflow
    thread_sumval *= sycl::native::exp(thread_maxval - block_maxval);
    // (warp-level) reduce sumval, thread 0 in each warp saves result in shared memory
    float warp_sumval = sycl::reduce_over_group(
        sycl::ext::oneapi::this_work_item::get_sub_group(), thread_sumval,
        sycl::plus<float>{});
    if (lane_id == 0) { shared_sumval[warp_id] = warp_sumval; }
    /*
    DPCT1065:24: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    // same strategy, now reduce sumval across warps
    warp_sumval = (lane_id < num_warps) ? shared_sumval[lane_id] : 0.0f;
    float block_sumval = sycl::reduce_over_group(
        sycl::ext::oneapi::this_work_item::get_sub_group(), warp_sumval,
        sycl::plus<float>{});
    // return the softmax parameters
    return SoftmaxParams{1.f / block_sumval, block_maxval};
}

// same as 2 but not using float4
void fused_classifier_kernel3(float* dlogits, float* losses, float* probs,
                                         const float* logits, const float* dlosses, const int* targets,
                                         int B, int T, int V, int P,
                                         float *shared_maxval,
                                         float *shared_sumval) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    sycl::group<3> block =
        sycl::ext::oneapi::this_work_item::get_work_group<3>();
    sycl::sub_group warp = sycl::ext::oneapi::this_work_item::get_sub_group();
    int64_t idx = item_ct1.get_group(2);
    int ix = targets[idx];

    // softmax (reading B * T * V, same logits read again below, hopefully still in cache)
    SoftmaxParams sp = prepare_softmax_blockwide_nofloat4(
        warp, idx, logits, V, P, shared_maxval, shared_sumval);

    // calculate the probability needed for the loss and update (single-threaded)
    if (item_ct1.get_local_id(2) == 0) {
        float prob =
            sycl::native::exp(logits[idx * P + ix] - sp.Offset) * sp.Scale;
        losses[idx] = -sycl::log(prob);
    }

    // very sensible default for dlosses is 1/(B*T), which is the uniform loss
    float dloss = dlosses != NULL ? dlosses[idx] : 1.0f / (B*T);
    // calculate the gradients directly, saves bandwidth from probs during training
    // but also supports writing probs for inference-only and debugging
    const float* logits_vec = logits + idx * P;
    for (int i = item_ct1.get_local_id(2); i < V;
         i += item_ct1.get_local_range(2)) {
        // this is the 2nd read of logits after the one in prepare_softmax2
        // this data will never be needed again, so we reduce cache persistence
        /*
        DPCT1098:25: The '*' expression is used instead of the __ldcs call.
        These two expressions do not provide the exact same functionality. Check
        the generated code for potential precision and/or performance issues.
        */
        float v = logits_vec[i];
        float prob = sycl::native::exp(v - sp.Offset) * sp.Scale;
        if (probs != NULL) {
            probs[idx * P + i] = prob;
        }
        if (dlogits != NULL) {
            float indicator = (i == ix) ? 1.0f : 0.0f;
            dlogits[idx * P + i] = (prob - indicator) * dloss;
        }
    }
}

SoftmaxParams prepare_softmax_blockwide2(int64_t idx, const floatX* inp, int V, int P,
                                         float *shared_maxval,
                                         float *shared_sumval) {
    // one row of inp, i.e. inp[idx, :] of shape (V,)

    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const floatX *x = inp + idx * P;
    float thread_maxval = -INFINITY;
    float thread_sumval = 0.0f;
    // do the loop in reverse to maximise probability of L2 cache hits
    // so even small L2s get some hits on the 2nd read of the same thread
    for (int i = ceil_div(V, x128::size) + item_ct1.get_local_id(2) -
                 item_ct1.get_local_range(2);
         i >= 0; i -= item_ct1.get_local_range(2)) {
        x128 packed_x = load128cs(x + i * x128::size); // load and do not keep in cache
        for(int k = 0; k < packed_x.size; ++k) {
            if (i*x128::size+k >= V) {  // bounds checking against real V
                continue;
            }
            float v = (float)packed_x[k];
            float old_maxval = thread_maxval;
            thread_maxval = sycl::fmax(thread_maxval, v);
            thread_sumval *= sycl::native::exp(old_maxval - thread_maxval);
            thread_sumval += sycl::native::exp(v - thread_maxval);
        }
    }
    // two reductions of up to 1024 threads:
    // 1) inside warp (shuffle), 2) cross-warp (shared memory), 3) inside warp (shuffle)
    // this results in much cleaner assembly than a multi-warp cg::reduce

    int num_warps = item_ct1.get_local_range(2) / 32;
    int warp_id = item_ct1.get_local_id(2) / 32;
    int lane_id = item_ct1.get_local_id(2) % 32;

    // reduce maxval within each warp
    float warp_maxval = warpReduceMax(thread_maxval);
    // thread 0 in each warp writes to shared memory
    if (lane_id == 0) { shared_maxval[warp_id] = warp_maxval; }
    /*
    DPCT1065:26: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    // each thread now loads the maxval across previous warps
    // if the thread is "out of range" of data, use -FLT_MAX as the maxval
    warp_maxval = (lane_id < num_warps) ? shared_maxval[lane_id] : -FLT_MAX;
    // now reduce the maxval among the warp threads
    float block_maxval = warpReduceMax(warp_maxval);
    // each thread uses maxval to scale sumval to avoid numerical instability / overflow
    thread_sumval *= sycl::native::exp(thread_maxval - block_maxval);
    // (warp-level) reduce sumval, thread 0 in each warp saves result in shared memory
    float warp_sumval = warpReduceSum(thread_sumval); //cg::reduce(warp, thread_sumval, cg::plus<float>{});

    if (lane_id == 0) { shared_sumval[warp_id] = warp_sumval; }
    /*
    DPCT1065:27: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    // same strategy, now reduce sumval across warps
    warp_sumval = (lane_id < num_warps) ? shared_sumval[lane_id] : 0.0f;
    float block_sumval = warpReduceSum(warp_sumval); //cg::reduce(warp, thread_sumval, cg::plus<float>{});
    // return the softmax parameters
    return SoftmaxParams{1.f / block_sumval, block_maxval};
}

// same as 2 but using x128
void fused_classifier_kernel4(floatX* dlogits, floatX* losses, floatX* probs,
                                         const floatX* logits, const floatX* dlosses, const int* targets,
                                         int B, int T, int V, int P,
                                         float *shared_maxval,
                                         float *shared_sumval) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int64_t idx = item_ct1.get_group(2);
    int ix = targets[idx];

    // softmax (reading B * T * V, same logits read again below, hopefully still in cache)
    SoftmaxParams sp = prepare_softmax_blockwide2(idx, logits, V, P,
                                                  shared_maxval, shared_sumval);

    // calculate the probability needed for the loss and update (single-threaded)
    if (item_ct1.get_local_id(2) == 0) {
        float prob =
            sycl::native::exp((float)logits[idx * P + ix] - sp.Offset) *
            sp.Scale;
        losses[idx] = -sycl::log(prob);
    }

    // very sensible default for dlosses is 1/(B*T), which is the uniform loss
    float dloss = dlosses != NULL ? (float)dlosses[idx] : 1.0f / (B*T);
    // calculate the gradients directly, saves bandwidth from probs during training
    // but also supports writing probs for inference-only and debugging
    const floatX* logits_vec = logits + idx * P;
    for (int i = item_ct1.get_local_id(2); i < ceil_div(V, x128::size);
         i += item_ct1.get_local_range(2)) {
        // this is the 2nd read of logits after the one in prepare_softmax2
        // this data will never be needed again, so we reduce cache persistence
        x128 packed_logits_vec = load128cs(logits_vec + i * x128::size); // load and do not keep in cache
        x128 packed_probs;
        x128 packed_dlogits;
        for(int k = 0; k < packed_logits_vec.size; ++k) {
            int element = i*packed_logits_vec.size + k;
            if (element >= V) {  // bounds checking against real V
                continue;
            }
            float v = packed_logits_vec[k];
            float prob = sycl::native::exp(v - sp.Offset) * sp.Scale;
            packed_probs[k] = prob;
            float indicator = (element == ix) ? 1.0f : 0.0f;
            packed_dlogits[k] = (prob - indicator) * dloss;
        }
        // Note: missing .cs hint hurts our performance due to cache thrashing, fixed in kernel5
        store128(dlogits + idx * P + i * packed_logits_vec.size, packed_dlogits);
        if (probs != NULL) {
            store128(probs + idx * P + i * packed_logits_vec.size, packed_probs);
        }
    }
}

SoftmaxParams prepare_softmax_blockwide3(int64_t idx, const floatX* inp, int V, int P,
                                         float *shared_val) {
    // same but not float4
    // one row of inp, i.e. inp[idx, :] of shape (V,)

    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const floatX *x = inp + idx * P;
    float thread_maxval = -INFINITY;
    float thread_sumval = 0.0f;
    int i = (V + x128::size - 1) / x128::size + item_ct1.get_local_id(2) -
            item_ct1.get_local_range(2);

    // special-case loop to handle the unaligned elements at the end of the array
    // this lets us skip the bounds check in the main loop below, which improves performance
    while ((i+1)*x128::size > V) {
        for(int k = 0; k < x128::size; ++k) {
            if (i*x128::size+k >= V) {
                break; // bounds checking against real V (rather than padded P)
            }
            float v = (float)x[i*x128::size+k];
            float old_maxval = thread_maxval;
            thread_maxval = sycl::fmax(thread_maxval, v);
            thread_sumval *= sycl::native::exp((old_maxval - thread_maxval));
            thread_sumval += sycl::native::exp(v - thread_maxval);
        }
        i -= item_ct1.get_local_range(2);
    }

    // main loop for the bulk of the iterations (no bounds checking required!)
    for (; i >= 0; i -= item_ct1.get_local_range(2)) {
        x128 packed_x = load128(x + i * x128::size); // load and keep in cache until fused_classifier loop
        for(int k = 0; k < x128::size; ++k) {
            float v = (float)packed_x[k];
            float old_maxval = thread_maxval;
            thread_maxval = sycl::fmax(thread_maxval, v);
            thread_sumval *= sycl::native::exp((old_maxval - thread_maxval));
            thread_sumval += sycl::native::exp(v - thread_maxval);
        }
    }

    // Block Max Reduction -> Maths -> Block Sum Reduction
    float block_maxval =
        blockReduce<warpReduceMax>(thread_maxval, false, -FLT_MAX, shared_val);
    thread_sumval *= sycl::native::exp(thread_maxval - block_maxval);
    float block_sumval = blockReduce<warpReduceSum>(thread_sumval, shared_val);

    // return the softmax parameters
    return SoftmaxParams{1.f / block_sumval, block_maxval};
}

// will _update_ logits to logit gradients
// uses template to decide whether to write logits and probs
// split both loops in "multiple-of-x128-size" and "bounds-checked remainder" parts
template <bool WriteLogits = true, bool WriteProbs = false>
void 
                fused_classifier_kernel5(floatX* dlogits, floatX* losses, floatX* probs,
                                         const floatX* logits, const floatX* dlosses, const int* targets,
                                         int B, int T, int V, int P,
                                         float *shared_val) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int64_t idx = item_ct1.get_group(2);
    int ix = targets[idx];

    // softmax (reading B * T * V, same logits read again below, hopefully still in cache)
    SoftmaxParams sp =
        prepare_softmax_blockwide3(idx, logits, V, P, shared_val);

    // calculate the probability needed for the loss and update (single-threaded)
    if (item_ct1.get_local_id(2) == 0) {
        float prob =
            sycl::native::exp((float)logits[idx * P + ix] - sp.Offset) *
            sp.Scale;
        losses[idx] = (floatX)(-sycl::log(prob));
    }

    // very sensible default for dlosses is 1/(B*T), which is the uniform loss
    float dloss = (dlosses != NULL) ? (float)dlosses[idx] : 1.0f / (B*T);
    // calculate the gradients directly, saves bandwidth from probs during training
    // but also supports writing probs for inference-only and debugging
    const floatX* logits_vec = logits + idx * P;
    for (int i = item_ct1.get_local_id(2); i < V / x128::size;
         i += item_ct1.get_local_range(2)) {
        // this is the 2nd read of logits after the one in prepare_softmax2
        // it will be overwritten by the logits gradients which is when we reduce cache persistence
        x128 packed_logits_vec = load128(logits_vec + i * x128::size); // rely on cs of store128cs
        x128 packed_probs;
        for(int k = 0; k < x128::size; ++k) {
            int element = i*x128::size + k;
            float prob =
                sycl::native::exp((float)packed_logits_vec[k] - sp.Offset) *
                sp.Scale;
            packed_probs[k] = (floatX)prob;
            float indicator = (element == ix) ? 1.0f : 0.0f;
            packed_logits_vec[k] = (floatX)((prob - indicator) * dloss);
        }
        if (WriteLogits){
            // reduce cache persistence for the overwritten logits
            // to maximise probability that logits remain in cache between prepare_softmax and here
            store128cs(dlogits + idx * P + i * x128::size, packed_logits_vec);
        }
        if (WriteProbs) {
            store128(probs + idx * P + i * x128::size, packed_probs);
        }
    }

    // handle remaining elements after the last multiple of x128::size
    // e.g. if V = 8003, and x128::size = 8, we need to handle the last 3 elements
    int unaligned_start = V & ~(x128::size - 1); // round down to multiple of x128::size
    for (int i = item_ct1.get_local_id(2) + unaligned_start; i < V; i++) {
        float prob =
            sycl::native::exp((float)logits_vec[i] - sp.Offset) * sp.Scale;
        float indicator = (i == ix) ? 1.0f : 0.0f;
        float dlogit = (prob - indicator) * dloss;
        if (WriteLogits){
            /*
            DPCT1098:28: The '=' expression is used instead of the __stcs call.
            These two expressions do not provide the exact same functionality.
            Check the generated code for potential precision and/or performance
            issues.
            */
            *(dlogits + idx * P + i) = (floatX)dlogit;
        }
        if (WriteProbs) {
            probs[idx * P + i] = (floatX)prob;
        }
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

void fused_classifier1(float* dlogits, float* losses,
                      const float* logits, const float* dlosses, const int* targets,
                      int B, int T, int V, int P, int block_size) {
    const int N = B * T; // total number of rows in the input
    // how many rows of the input can each block of threads process?
    // e.g. in block_size=128, 4 rows get handled by 4 warps (of 32 threads each)
    const int rows_per_block = block_size / 32;
    const int grid_size = N / rows_per_block; // total number of blocks needed
    /*
    DPCT1049:1: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                              sycl::range<3>(1, 1, block_size),
                          sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
            fused_classifier_kernel1(dlogits, losses, logits, dlosses, targets,
                                     B, T, V, P);
        });
    /*
    DPCT1010:29: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    cudaCheck(0);
}

void fused_classifier2(float* dlogits, float* losses,
                      const float* logits, const float* dlosses, const int* targets,
                      int B, int T, int V, int P, int block_size) {
    const int N = B * T;
    const int grid_size = N;
    /*
    DPCT1049:2: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> shared_maxval_acc_ct1(sycl::range<1>(32),
                                                             cgh);
        sycl::local_accessor<float, 1> shared_sumval_acc_ct1(sycl::range<1>(32),
                                                             cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                                  sycl::range<3>(1, 1, block_size),
                              sycl::range<3>(1, 1, block_size)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                fused_classifier_kernel2(
                    dlogits, losses, NULL, logits, dlosses, targets, B, T, V, P,
                    shared_maxval_acc_ct1
                        .get_multi_ptr<sycl::access::decorated::no>()
                        .get(),
                    shared_sumval_acc_ct1
                        .get_multi_ptr<sycl::access::decorated::no>()
                        .get());
            });
    });
    /*
    DPCT1010:30: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    cudaCheck(0);
}

void fused_classifier3(float* dlogits, float* losses,
                      const float* logits, const float* dlosses, const int* targets,
                      int B, int T, int V, int P, int block_size) {
    const int N = B * T;
    const int grid_size = N;
    /*
    DPCT1049:3: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> shared_maxval_acc_ct1(sycl::range<1>(32),
                                                             cgh);
        sycl::local_accessor<float, 1> shared_sumval_acc_ct1(sycl::range<1>(32),
                                                             cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                                  sycl::range<3>(1, 1, block_size),
                              sycl::range<3>(1, 1, block_size)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                fused_classifier_kernel3(
                    dlogits, losses, NULL, logits, dlosses, targets, B, T, V, P,
                    shared_maxval_acc_ct1
                        .get_multi_ptr<sycl::access::decorated::no>()
                        .get(),
                    shared_sumval_acc_ct1
                        .get_multi_ptr<sycl::access::decorated::no>()
                        .get());
            });
    });
    /*
    DPCT1010:31: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    cudaCheck(0);
}

void fused_classifier4(float* dlogits, float* losses,
                      const float* logits, const float* dlosses, const int* targets,
                      int B, int T, int V, int P, int block_size) {
    const int N = B * T;
    const int grid_size = N;
    /*
    DPCT1049:4: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> shared_maxval_acc_ct1(sycl::range<1>(32),
                                                             cgh);
        sycl::local_accessor<float, 1> shared_sumval_acc_ct1(sycl::range<1>(32),
                                                             cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                                  sycl::range<3>(1, 1, block_size),
                              sycl::range<3>(1, 1, block_size)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                fused_classifier_kernel4(
                    (floatX *)dlogits, (floatX *)losses, NULL, (floatX *)logits,
                    (floatX *)dlosses, targets, B, T, V, P,
                    shared_maxval_acc_ct1
                        .get_multi_ptr<sycl::access::decorated::no>()
                        .get(),
                    shared_sumval_acc_ct1
                        .get_multi_ptr<sycl::access::decorated::no>()
                        .get());
            });
    });
    /*
    DPCT1010:32: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    cudaCheck(0);
}

void fused_classifier5(float* dlogits, float* losses,
                      const float* logits, const float* dlosses, const int* targets,
                      int B, int T, int V, int P, int block_size) {
    const int N = B * T;
    const int grid_size = N;
    /*
    DPCT1049:5: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
        /*
        DPCT1101:37: 'WARP_SIZE' expression was replaced with a value. Modify
        the code to use the original expression, provided in comments, if it is
        correct.
        */
        sycl::local_accessor<float, 1> shared_val_acc_ct1(
            sycl::range<1>(32 /*WARP_SIZE*/), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                                  sycl::range<3>(1, 1, block_size),
                              sycl::range<3>(1, 1, block_size)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                fused_classifier_kernel5<true, false>(
                    (floatX *)dlogits, (floatX *)losses, NULL, (floatX *)logits,
                    (floatX *)dlosses, targets, B, T, V, P,
                    shared_val_acc_ct1
                        .get_multi_ptr<sycl::access::decorated::no>()
                        .get());
            });
    });
    /*
    DPCT1010:33: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    cudaCheck(0);
}

void fused_classifier(int kernel_num, float* dlogits, float* losses,
                      const float* logits, const float* dlosses, const int* targets,
                      int B, int T, int V, int P, int block_size) {
    switch (kernel_num) {
        case 1:
            fused_classifier1(dlogits, losses, logits, dlosses, targets, B, T, V, P, block_size);
            break;
        case 2:
            fused_classifier2(dlogits, losses, logits, dlosses, targets, B, T, V, P, block_size);
            break;
        case 3:
            fused_classifier3(dlogits, losses, logits, dlosses, targets, B, T, V, P, block_size);
            break;
        case 4:
            fused_classifier4(dlogits, losses, logits, dlosses, targets, B, T, V, P, block_size);
            break;
        case 5:
            fused_classifier5(dlogits, losses, logits, dlosses, targets, B, T, V, P, block_size);
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
        fprintf(stderr, "usage: %s <input> <output> B T V\n", argv[0]);
        return 2;
    }
    const int B = atoi(argv[3]);
    const int T = atoi(argv[4]);
    const int V = atoi(argv[5]);
    FILE* fin = fopen(argv[1], "rb");
    if (!fin) { perror(argv[1]); return 2; }
    float* logits = (float*)malloc(sizeof(float) * (size_t)(B*T*V));
    if (fread(logits, sizeof(float), (size_t)(B*T*V), fin) != (size_t)(B*T*V)) {
        fprintf(stderr, "input too short at logits\n"); return 2;
    }
    float* dlosses = (float*)malloc(sizeof(float) * (size_t)(B*T));
    if (fread(dlosses, sizeof(float), (size_t)(B*T), fin) != (size_t)(B*T)) {
        fprintf(stderr, "input too short at dlosses\n"); return 2;
    }
    int* targets = (int*)malloc(sizeof(int) * (size_t)(B*T));
    if (fread(targets, sizeof(int), (size_t)(B*T), fin) != (size_t)(B*T)) {
        fprintf(stderr, "input too short at targets\n"); return 2;
    }
    fclose(fin);
    FILE* fout = fopen(argv[2], "wb");
    if (!fout) { perror(argv[2]); return 2; }
    float* probs = (float*)malloc((size_t)B * T * V * sizeof(float)); for (int j = 0; j < B * T; ++j) { for (int k = 0; k < 3; ++k) { int idx = rand() % V; logits[j * V + idx] *= 20.0f; } }
    float* d_logits;
    cudaCheck(
        DPCT_CHECK_ERROR(d_logits = sycl::malloc_device<float>(
                             (size_t)(B * T * V), dpct::get_in_order_queue())));
    cudaCheck(DPCT_CHECK_ERROR(
        dpct::get_in_order_queue()
            .memcpy(d_logits, logits, sizeof(float) * (size_t)(B * T * V))
            .wait()));
    float* d_dlosses;
    cudaCheck(
        DPCT_CHECK_ERROR(d_dlosses = sycl::malloc_device<float>(
                             (size_t)(B * T), dpct::get_in_order_queue())));
    cudaCheck(DPCT_CHECK_ERROR(
        dpct::get_in_order_queue()
            .memcpy(d_dlosses, dlosses, sizeof(float) * (size_t)(B * T))
            .wait()));
    int* d_targets;
    cudaCheck(
        DPCT_CHECK_ERROR(d_targets = sycl::malloc_device<int>(
                             (size_t)(B * T), dpct::get_in_order_queue())));
    cudaCheck(DPCT_CHECK_ERROR(
        dpct::get_in_order_queue()
            .memcpy(d_targets, targets, sizeof(int) * (size_t)(B * T))
            .wait()));
    float* losses = (float*)malloc(sizeof(float) * (size_t)(B*T));
    float* d_losses;
    cudaCheck(
        DPCT_CHECK_ERROR(d_losses = sycl::malloc_device<float>(
                             (size_t)(B * T), dpct::get_in_order_queue())));
    float* dlogits = (float*)malloc(sizeof(float) * (size_t)(B*T*V));
    float* d_dlogits;
    cudaCheck(
        DPCT_CHECK_ERROR(d_dlogits = sycl::malloc_device<float>(
                             (size_t)(B * T * V), dpct::get_in_order_queue())));
    const int variants[] = {1, 2, 3, 4, 5};
    for (int vi = 0; vi < 5; vi++) {
        const int k = variants[vi];
        cudaCheck(DPCT_CHECK_ERROR(
            dpct::get_in_order_queue()
                .memset(d_losses, 0, sizeof(float) * (size_t)(B * T))
                .wait()));
        cudaCheck(DPCT_CHECK_ERROR(
            dpct::get_in_order_queue()
                .memset(d_dlogits, 0, sizeof(float) * (size_t)(B * T * V))
                .wait()));
        fused_classifier(k, d_dlogits, d_losses, d_logits, d_dlosses, d_targets, B, T, V, V, 256);
        /*
        DPCT1010:34: SYCL uses exceptions to report errors and does not use the
        error codes. The cudaGetLastError function call was replaced with 0. You
        need to rewrite this code.
        */
        cudaCheck(0);
        cudaCheck(DPCT_CHECK_ERROR(
            dpct::get_current_device().queues_wait_and_throw()));
        cudaCheck(DPCT_CHECK_ERROR(
            dpct::get_in_order_queue()
                .memcpy(losses, d_losses, sizeof(float) * (size_t)(B * T))
                .wait()));
        fwrite(losses, sizeof(float), (size_t)(B*T), fout);
        cudaCheck(DPCT_CHECK_ERROR(
            dpct::get_in_order_queue()
                .memcpy(dlogits, d_dlogits, sizeof(float) * (size_t)(B * T * V))
                .wait()));
        fwrite(dlogits, sizeof(float), (size_t)(B*T*V), fout);
    }
    fclose(fout);
    return 0;
}
