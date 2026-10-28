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

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
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
__device__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

// ----------------------------------------------------------------------------
// GPU kernels

struct SoftmaxParams {
    float Scale;
    float Offset;
};
namespace cg = cooperative_groups;
__device__ SoftmaxParams prepare_softmax(cg::thread_block_tile<32>& warp,
                                         int64_t idx, const float* inp, int V, int P) {
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
    for (int i = warp.thread_rank(); i < V; i += warp.size()) {
        float v = x[i];
        float old_maxval = maxval;
        // online softmax recurrence from "Online normalizer calculation for softmax" paper
        maxval = fmaxf(maxval, v);
        sumval *= expf((old_maxval - maxval));
        sumval += expf(v - maxval);
    }
    // warp-level reduction to get the maxval across the 32 threads
    float global_maxval = cg::reduce(warp, maxval, cg::greater<float>{});
    // all 32 threads do a final shift of the sum considering the global max in this row
    sumval *= expf((maxval - global_maxval));
    // warp-level reduction to get the sumval across the 32 threads
    float global_sumval = cg::reduce(warp, sumval, cg::plus<float>{});
    // the final normalization factor
    float norm = 1.0f / global_sumval;
    return SoftmaxParams{norm, global_maxval};
}

__global__ void fused_classifier_kernel1(float* dlogits, float* losses,
                             const float* logits, const float* dlosses, const int* targets,
                             int B, int T, int V, int P) {
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    // example: B = 4, T = 1024, block_size = 128 => we'd have grid_size = 1024
    // each block of 4 warps is in charge of 4 rows of the input, one warp per row
    // meta_group_size is the number of warps per block (e.g. 4)
    // meta_group_rank is the index of the warp in the block (e.g. 0, 1, 2, 3)
    int64_t idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    if (idx >= B * T) { // there are B * T rows in the input
        return;
    }
    int b = idx / T;
    int t = idx % T;

    // calculate the offset (maxval) and scale (sumval) for the softmax
    SoftmaxParams sp = prepare_softmax(warp, idx, logits, V, P);

    // in each row (handled by one warp), thread 0 calculates the loss
    // calculate the probability needed for the loss and update losses
    if(warp.thread_rank() == 0) {
        int ix = targets[b * T + t];
        float prob = expf(logits[idx * P + ix] - sp.Offset) * sp.Scale;
        losses[b * T + t] = -logf(prob);
    }

    // finally all threads calculate the gradients
    // prob is only materialized here temporarily and in registers, never
    // as a full tensor that gets written to global memory
    for (int i = warp.thread_rank(); i < V; i += warp.size()) {
        float prob = expf(logits[idx * P + i] - sp.Offset) * sp.Scale;
        float* dlogits_bt = dlogits + b * T * P + t * P;
        float dloss = dlosses[b * T + t];
        int ix = targets[b * T + t];
        float indicator = i == ix ? 1.0f : 0.0f;
        dlogits_bt[i] = (prob - indicator) * dloss;
    }
}


__device__ float vec_at(const float4& vec, int index) {
    return reinterpret_cast<const float*>(&vec)[index];
}

__device__ SoftmaxParams prepare_softmax_blockwide(cg::thread_block_tile<32>& warp,
                                                   int64_t idx, const float* inp, int V, int P) {
    // one row of inp, i.e. inp[idx, :] of shape (V,)
    // float4 to get 128-bit loads and memory level parallelism
    const float4* x_vec4 = reinterpret_cast<const float4*>(inp + idx * P);

    float thread_maxval = -INFINITY;
    float thread_sumval = 0.0f;
    // do the loop in reverse to maximise probability of L2 cache hits
    // so even small L2s get some hits on the 2nd read of the same thread
    for (int i = ceil_div(V, 4) + threadIdx.x - blockDim.x; i >= 0; i -= blockDim.x) {
        float4 v4 = x_vec4[i];
        #pragma unroll
        for(int k = 0; k < 4; k++) {
            if (i*4+k >= V) {  // bounds checking against real V
                continue;
            }
            float old_maxval = thread_maxval;
            thread_maxval = fmaxf(thread_maxval, vec_at(v4, k));
            thread_sumval *= expf(old_maxval - thread_maxval);
            thread_sumval += expf(vec_at(v4, k) - thread_maxval);
        }
    }

    // two reductions of up to 1024 threads:
    // 1) inside warp (shuffle), 2) cross-warp (shared memory), 3) inside warp (shuffle)
    // this results in much cleaner assembly than a multi-warp cg::reduce
    __shared__ float shared_maxval[32];
    __shared__ float shared_sumval[32];
    int num_warps = blockDim.x / 32;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    // reduce maxval within each warp
    float warp_maxval = cg::reduce(warp, thread_maxval, cg::greater<float>{});
    // thread 0 in each warp writes to shared memory
    if (lane_id == 0) { shared_maxval[warp_id] = warp_maxval; }
    __syncthreads();
    // each thread now loads the maxval across previous warps
    // if the thread is "out of range" of data, use -FLT_MAX as the maxval
    warp_maxval = (lane_id < num_warps) ? shared_maxval[lane_id] : -FLT_MAX;
    // now reduce the maxval among the warp threads
    float block_maxval = cg::reduce(warp, warp_maxval, cg::greater<float>{});
    // each thread uses maxval to scale sumval to avoid numerical instability / overflow
    thread_sumval *= expf(thread_maxval - block_maxval);
    // (warp-level) reduce sumval, thread 0 in each warp saves result in shared memory
    float warp_sumval = cg::reduce(warp, thread_sumval, cg::plus<float>{});
    if (lane_id == 0) { shared_sumval[warp_id] = warp_sumval; }
    __syncthreads();
    // same strategy, now reduce sumval across warps
    warp_sumval = (lane_id < num_warps) ? shared_sumval[lane_id] : 0.0f;
    float block_sumval = cg::reduce(warp, warp_sumval, cg::plus<float>{});
    // return the softmax parameters
    return SoftmaxParams{1.f / block_sumval, block_maxval};
}

// Fused forward and backward pass for classifier including softmax, and logit gradients
// Writes to both probs (only for debugging) and dlogits (only for training) are optional
// N.B.: We may want to reuse the logits memory for dlogits, so they should *not* be __restrict__!
__global__ void fused_classifier_kernel2(float* dlogits, float* losses, float* probs,
                                         const float* logits, const float* dlosses, const int* targets,
                                         int B, int T, int V, int P) {
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int64_t idx = blockIdx.x;
    int ix = targets[idx];

    // softmax (reading B * T * V, same logits read again below, hopefully still in cache)
    SoftmaxParams sp = prepare_softmax_blockwide(warp, idx, logits, V, P);

    // calculate the probability needed for the loss and update (single-threaded)
    if(threadIdx.x == 0) {
        float prob = expf(logits[idx * P + ix] - sp.Offset) * sp.Scale;
        losses[idx] = -logf(prob);
    }

    // very sensible default for dlosses is 1/(B*T), which is the uniform loss
    float dloss = dlosses != NULL ? dlosses[idx] : 1.0f / (B*T);
    // calculate the gradients directly, saves bandwidth from probs during training
    // but also supports writing probs for inference-only and debugging
    const float4* logits_vec4 = reinterpret_cast<const float4*>(logits + idx * P);
    for (int i = threadIdx.x; i < ceil_div(V, 4); i += blockDim.x) {
        // this is the 2nd read of logits after the one in prepare_softmax2
        // this data will never be needed again, so we reduce cache persistence
        float4 v4 = __ldcs(&logits_vec4[i]);

        #pragma unroll
        for(int k = 0; k < 4; ++k) {
            int element = i*4 + k;
            float prob = expf(vec_at(v4, k) - sp.Offset) * sp.Scale;
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

__device__ SoftmaxParams prepare_softmax_blockwide_nofloat4(cg::thread_block_tile<32>& warp,
                                                            int64_t idx, const float* inp, int V, int P) {
    // same but not float4
    // one row of inp, i.e. inp[idx, :] of shape (V,)

    const float* x = inp + idx * P;
    float thread_maxval = -INFINITY;
    float thread_sumval = 0.0f;
    // do the loop in reverse to maximise probability of L2 cache hits
    // so even small L2s get some hits on the 2nd read of the same thread
    for (int i = V + threadIdx.x - blockDim.x; i >= 0; i -= blockDim.x) {
        float v = x[i];
        float old_maxval = thread_maxval;
        thread_maxval = fmaxf(thread_maxval, v);
        thread_sumval *= expf(old_maxval - thread_maxval);
        thread_sumval += expf(v - thread_maxval);
    }

    // two reductions of up to 1024 threads:
    // 1) inside warp (shuffle), 2) cross-warp (shared memory), 3) inside warp (shuffle)
    // this results in much cleaner assembly than a multi-warp cg::reduce
    __shared__ float shared_maxval[32];
    __shared__ float shared_sumval[32];
    int num_warps = blockDim.x / 32;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    // reduce maxval within each warp
    float warp_maxval = cg::reduce(warp, thread_maxval, cg::greater<float>{});
    // thread 0 in each warp writes to shared memory
    if (lane_id == 0) { shared_maxval[warp_id] = warp_maxval; }
    __syncthreads();
    // each thread now loads the maxval across previous warps
    // if the thread is "out of range" of data, use -FLT_MAX as the maxval
    warp_maxval = (lane_id < num_warps) ? shared_maxval[lane_id] : -FLT_MAX;
    // now reduce the maxval among the warp threads
    float block_maxval = cg::reduce(warp, warp_maxval, cg::greater<float>{});
    // each thread uses maxval to scale sumval to avoid numerical instability / overflow
    thread_sumval *= expf(thread_maxval - block_maxval);
    // (warp-level) reduce sumval, thread 0 in each warp saves result in shared memory
    float warp_sumval = cg::reduce(warp, thread_sumval, cg::plus<float>{});
    if (lane_id == 0) { shared_sumval[warp_id] = warp_sumval; }
    __syncthreads();
    // same strategy, now reduce sumval across warps
    warp_sumval = (lane_id < num_warps) ? shared_sumval[lane_id] : 0.0f;
    float block_sumval = cg::reduce(warp, warp_sumval, cg::plus<float>{});
    // return the softmax parameters
    return SoftmaxParams{1.f / block_sumval, block_maxval};
}

// same as 2 but not using float4
__global__ void fused_classifier_kernel3(float* dlogits, float* losses, float* probs,
                                         const float* logits, const float* dlosses, const int* targets,
                                         int B, int T, int V, int P) {
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int64_t idx = blockIdx.x;
    int ix = targets[idx];

    // softmax (reading B * T * V, same logits read again below, hopefully still in cache)
    SoftmaxParams sp = prepare_softmax_blockwide_nofloat4(warp, idx, logits, V, P);

    // calculate the probability needed for the loss and update (single-threaded)
    if(threadIdx.x == 0) {
        float prob = expf(logits[idx * P + ix] - sp.Offset) * sp.Scale;
        losses[idx] = -logf(prob);
    }

    // very sensible default for dlosses is 1/(B*T), which is the uniform loss
    float dloss = dlosses != NULL ? dlosses[idx] : 1.0f / (B*T);
    // calculate the gradients directly, saves bandwidth from probs during training
    // but also supports writing probs for inference-only and debugging
    const float* logits_vec = logits + idx * P;
    for (int i = threadIdx.x; i < V; i += blockDim.x) {
        // this is the 2nd read of logits after the one in prepare_softmax2
        // this data will never be needed again, so we reduce cache persistence
        float v = __ldcs(&logits_vec[i]);
        float prob = expf(v - sp.Offset) * sp.Scale;
        if (probs != NULL) {
            probs[idx * P + i] = prob;
        }
        if (dlogits != NULL) {
            float indicator = (i == ix) ? 1.0f : 0.0f;
            dlogits[idx * P + i] = (prob - indicator) * dloss;
        }
    }
}

__device__ SoftmaxParams prepare_softmax_blockwide2(int64_t idx, const floatX* inp, int V, int P) {
    // one row of inp, i.e. inp[idx, :] of shape (V,)

    const floatX* x = inp + idx * P;
    float thread_maxval = -INFINITY;
    float thread_sumval = 0.0f;
    // do the loop in reverse to maximise probability of L2 cache hits
    // so even small L2s get some hits on the 2nd read of the same thread
    for (int i = ceil_div(V, x128::size) + threadIdx.x - blockDim.x; i >= 0; i -= blockDim.x) {
        x128 packed_x = load128cs(x + i * x128::size); // load and do not keep in cache
        for(int k = 0; k < packed_x.size; ++k) {
            if (i*x128::size+k >= V) {  // bounds checking against real V
                continue;
            }
            float v = (float)packed_x[k];
            float old_maxval = thread_maxval;
            thread_maxval = fmaxf(thread_maxval, v);
            thread_sumval *= expf(old_maxval - thread_maxval);
            thread_sumval += expf(v - thread_maxval);
        }
    }
    // two reductions of up to 1024 threads:
    // 1) inside warp (shuffle), 2) cross-warp (shared memory), 3) inside warp (shuffle)
    // this results in much cleaner assembly than a multi-warp cg::reduce
    __shared__ float shared_maxval[32];
    __shared__ float shared_sumval[32];
    int num_warps = blockDim.x / 32;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    // reduce maxval within each warp
    float warp_maxval = warpReduceMax(thread_maxval);
    // thread 0 in each warp writes to shared memory
    if (lane_id == 0) { shared_maxval[warp_id] = warp_maxval; }
    __syncthreads();
    // each thread now loads the maxval across previous warps
    // if the thread is "out of range" of data, use -FLT_MAX as the maxval
    warp_maxval = (lane_id < num_warps) ? shared_maxval[lane_id] : -FLT_MAX;
    // now reduce the maxval among the warp threads
    float block_maxval = warpReduceMax(warp_maxval);
    // each thread uses maxval to scale sumval to avoid numerical instability / overflow
    thread_sumval *= expf(thread_maxval - block_maxval);
    // (warp-level) reduce sumval, thread 0 in each warp saves result in shared memory
    float warp_sumval = warpReduceSum(thread_sumval); //cg::reduce(warp, thread_sumval, cg::plus<float>{});

    if (lane_id == 0) { shared_sumval[warp_id] = warp_sumval; }
    __syncthreads();
    // same strategy, now reduce sumval across warps
    warp_sumval = (lane_id < num_warps) ? shared_sumval[lane_id] : 0.0f;
    float block_sumval = warpReduceSum(warp_sumval); //cg::reduce(warp, thread_sumval, cg::plus<float>{});
    // return the softmax parameters
    return SoftmaxParams{1.f / block_sumval, block_maxval};
}

// same as 2 but using x128
__global__ void fused_classifier_kernel4(floatX* dlogits, floatX* losses, floatX* probs,
                                         const floatX* logits, const floatX* dlosses, const int* targets,
                                         int B, int T, int V, int P) {
    int64_t idx = blockIdx.x;
    int ix = targets[idx];

    // softmax (reading B * T * V, same logits read again below, hopefully still in cache)
    SoftmaxParams sp = prepare_softmax_blockwide2(idx, logits, V, P);

    // calculate the probability needed for the loss and update (single-threaded)
    if(threadIdx.x == 0) {
        float prob = expf((float)logits[idx * P + ix] - sp.Offset) * sp.Scale;
        losses[idx] = -logf(prob);
    }

    // very sensible default for dlosses is 1/(B*T), which is the uniform loss
    float dloss = dlosses != NULL ? (float)dlosses[idx] : 1.0f / (B*T);
    // calculate the gradients directly, saves bandwidth from probs during training
    // but also supports writing probs for inference-only and debugging
    const floatX* logits_vec = logits + idx * P;
    for (int i = threadIdx.x; i < ceil_div(V , x128::size); i += blockDim.x) {
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
            float prob = expf(v - sp.Offset) * sp.Scale;
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

__device__ SoftmaxParams prepare_softmax_blockwide3(int64_t idx, const floatX* inp, int V, int P) {
    // same but not float4
    // one row of inp, i.e. inp[idx, :] of shape (V,)

    const floatX* x = inp + idx * P;
    float thread_maxval = -INFINITY;
    float thread_sumval = 0.0f;
    int i = (V+x128::size-1)/x128::size + threadIdx.x - blockDim.x;

    // special-case loop to handle the unaligned elements at the end of the array
    // this lets us skip the bounds check in the main loop below, which improves performance
    while ((i+1)*x128::size > V) {
        for(int k = 0; k < x128::size; ++k) {
            if (i*x128::size+k >= V) {
                break; // bounds checking against real V (rather than padded P)
            }
            float v = (float)x[i*x128::size+k];
            float old_maxval = thread_maxval;
            thread_maxval = fmaxf(thread_maxval, v);
            thread_sumval *= expf((old_maxval - thread_maxval));
            thread_sumval += expf(v - thread_maxval);
        }
        i -= blockDim.x;
    }

    // main loop for the bulk of the iterations (no bounds checking required!)
    for (; i >= 0; i -= blockDim.x) {
        x128 packed_x = load128(x + i * x128::size); // load and keep in cache until fused_classifier loop
        for(int k = 0; k < x128::size; ++k) {
            float v = (float)packed_x[k];
            float old_maxval = thread_maxval;
            thread_maxval = fmaxf(thread_maxval, v);
            thread_sumval *= expf((old_maxval - thread_maxval));
            thread_sumval += expf(v - thread_maxval);
        }
    }

    // Block Max Reduction -> Maths -> Block Sum Reduction
    float block_maxval = blockReduce<warpReduceMax>(thread_maxval, false, -FLT_MAX);
    thread_sumval *= expf(thread_maxval - block_maxval);
    float block_sumval = blockReduce<warpReduceSum>(thread_sumval);

    // return the softmax parameters
    return SoftmaxParams{1.f / block_sumval, block_maxval};
}

// will _update_ logits to logit gradients
// uses template to decide whether to write logits and probs
// split both loops in "multiple-of-x128-size" and "bounds-checked remainder" parts
template <bool WriteLogits = true, bool WriteProbs = false>
__global__ void __launch_bounds__(1024, MAX_1024_THREADS_BLOCKS)
                fused_classifier_kernel5(floatX* dlogits, floatX* losses, floatX* probs,
                                         const floatX* logits, const floatX* dlosses, const int* targets,
                                         int B, int T, int V, int P) {
    int64_t idx = blockIdx.x;
    int ix = targets[idx];

    // softmax (reading B * T * V, same logits read again below, hopefully still in cache)
    SoftmaxParams sp = prepare_softmax_blockwide3(idx, logits, V, P);

    // calculate the probability needed for the loss and update (single-threaded)
    if(threadIdx.x == 0) {
        float prob = expf((float)logits[idx * P + ix] - sp.Offset) * sp.Scale;
        losses[idx] = (floatX)(-logf(prob));
    }

    // very sensible default for dlosses is 1/(B*T), which is the uniform loss
    float dloss = (dlosses != NULL) ? (float)dlosses[idx] : 1.0f / (B*T);
    // calculate the gradients directly, saves bandwidth from probs during training
    // but also supports writing probs for inference-only and debugging
    const floatX* logits_vec = logits + idx * P;
    for (int i = threadIdx.x; i < V/x128::size; i += blockDim.x) {
        // this is the 2nd read of logits after the one in prepare_softmax2
        // it will be overwritten by the logits gradients which is when we reduce cache persistence
        x128 packed_logits_vec = load128(logits_vec + i * x128::size); // rely on cs of store128cs
        x128 packed_probs;
        for(int k = 0; k < x128::size; ++k) {
            int element = i*x128::size + k;
            float prob = expf((float)packed_logits_vec[k] - sp.Offset) * sp.Scale;
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
    for (int i = threadIdx.x + unaligned_start; i < V; i++) {
        float prob = expf((float)logits_vec[i] - sp.Offset) * sp.Scale;
        float indicator = (i == ix) ? 1.0f : 0.0f;
        float dlogit = (prob - indicator) * dloss;
        if (WriteLogits){
            __stcs(dlogits + idx * P + i, (floatX)dlogit);
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
    fused_classifier_kernel1<<<grid_size, block_size>>>(dlogits, losses, logits, dlosses, targets, B, T, V, P);
    cudaCheck(cudaGetLastError());
}

void fused_classifier2(float* dlogits, float* losses,
                      const float* logits, const float* dlosses, const int* targets,
                      int B, int T, int V, int P, int block_size) {
    const int N = B * T;
    const int grid_size = N;
    fused_classifier_kernel2<<<grid_size, block_size>>>(dlogits, losses, NULL, logits, dlosses, targets, B, T, V, P);
    cudaCheck(cudaGetLastError());
}

void fused_classifier3(float* dlogits, float* losses,
                      const float* logits, const float* dlosses, const int* targets,
                      int B, int T, int V, int P, int block_size) {
    const int N = B * T;
    const int grid_size = N;
    fused_classifier_kernel3<<<grid_size, block_size>>>(dlogits, losses, NULL, logits, dlosses, targets, B, T, V, P);
    cudaCheck(cudaGetLastError());
}

void fused_classifier4(float* dlogits, float* losses,
                      const float* logits, const float* dlosses, const int* targets,
                      int B, int T, int V, int P, int block_size) {
    const int N = B * T;
    const int grid_size = N;
    fused_classifier_kernel4<<<grid_size, block_size>>>((floatX*)dlogits, (floatX*)losses, NULL, (floatX*)logits, (floatX*)dlosses, targets, B, T, V, P);
    cudaCheck(cudaGetLastError());
}

void fused_classifier5(float* dlogits, float* losses,
                      const float* logits, const float* dlosses, const int* targets,
                      int B, int T, int V, int P, int block_size) {
    const int N = B * T;
    const int grid_size = N;
    fused_classifier_kernel5<true,false><<<grid_size, block_size>>>((floatX*)dlogits, (floatX*)losses, NULL, (floatX*)logits, (floatX*)dlosses, targets, B, T, V, P);
    cudaCheck(cudaGetLastError());
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
    float* probs = (float*)malloc((size_t)B * T * V * sizeof(float)); for (size_t j = 0; j < (size_t)B * T; ++j) { for (int k = 0; k < 3; ++k) { size_t idx = (j * 1315423911u + (size_t)k * 2654435761u + 12345u) % (size_t)V; logits[j * (size_t)V + idx] *= 20.0f; } }
    float* d_logits;
    cudaCheck(cudaMalloc(&d_logits, sizeof(float) * (size_t)(B*T*V)));
    cudaCheck(cudaMemcpy(d_logits, logits, sizeof(float) * (size_t)(B*T*V), cudaMemcpyHostToDevice));
    float* d_dlosses;
    cudaCheck(cudaMalloc(&d_dlosses, sizeof(float) * (size_t)(B*T)));
    cudaCheck(cudaMemcpy(d_dlosses, dlosses, sizeof(float) * (size_t)(B*T), cudaMemcpyHostToDevice));
    int* d_targets;
    cudaCheck(cudaMalloc(&d_targets, sizeof(int) * (size_t)(B*T)));
    cudaCheck(cudaMemcpy(d_targets, targets, sizeof(int) * (size_t)(B*T), cudaMemcpyHostToDevice));
    float* losses = (float*)malloc(sizeof(float) * (size_t)(B*T));
    float* d_losses;
    cudaCheck(cudaMalloc(&d_losses, sizeof(float) * (size_t)(B*T)));
    float* dlogits = (float*)malloc(sizeof(float) * (size_t)(B*T*V));
    float* d_dlogits;
    cudaCheck(cudaMalloc(&d_dlogits, sizeof(float) * (size_t)(B*T*V)));
    const int variants[] = {1, 2, 3, 4, 5};
    for (int vi = 0; vi < 5; vi++) {
        const int k = variants[vi];
        cudaCheck(cudaMemset(d_losses, 0, sizeof(float) * (size_t)(B*T)));
        cudaCheck(cudaMemset(d_dlogits, 0, sizeof(float) * (size_t)(B*T*V)));
        fused_classifier(k, d_dlogits, d_losses, d_logits, d_dlosses, d_targets, B, T, V, V, 256);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaDeviceSynchronize());
        cudaCheck(cudaMemcpy(losses, d_losses, sizeof(float) * (size_t)(B*T), cudaMemcpyDeviceToHost));
        fwrite(losses, sizeof(float), (size_t)(B*T), fout);
        cudaCheck(cudaMemcpy(dlogits, d_dlogits, sizeof(float) * (size_t)(B*T*V), cudaMemcpyDeviceToHost));
        fwrite(dlogits, sizeof(float), (size_t)(B*T*V), fout);
    }
    fclose(fout);
    return 0;
}
