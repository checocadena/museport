/*
Fused Classifier:
- Forwards the Cross Entropy Loss
- Never materializes the full normalized logits, only at the target label
- (fusion) Also kicks off the backward pass, because everything is already loaded
SYCL 2020 port of llmc/fused_classifier.cuh.
*/
#ifndef FUSED_CLASSIFIER_H
#define FUSED_CLASSIFIER_H

// llmc internal imports
#include "sycl_common.h"
#include "sycl_utils.h"

// ----------------------------------------------------------------------------
// SYCL kernels

struct SoftmaxParams {
    float Scale;
    float Offset;
};

// replaces logits with logit gradients; template decides whether to write
// dlogits (training, WriteDLogits=true) or just accumulate losses (False).
template <bool WriteDLogits>
inline void fused_classifier_kernel(floatX* logits, float* losses,
                                    const float dloss, const int* targets,
                                    size_t B, size_t T, size_t V, size_t P) {
    size_t N = B * T;
    if (N == 0) { return; }
    constexpr size_t WG = 256;
    sycl_queue.parallel_for(sycl::nd_range<1>(sycl::range<1>(N * WG), sycl::range<1>(WG)),
        [=](sycl::nd_item<1> it) {
            size_t idx = it.get_group_linear_id();
            auto grp = it.get_group();
            size_t lid = it.get_local_linear_id();
            size_t lsz = it.get_local_range(0);
            size_t row = N - 1 - idx; // reverse order like CUDA (cache behaviour only)
            int ix = targets[row];
            const floatX* x = logits + row * P;
            // online softmax max/sum over V (padded P ignored), deterministic:
            // scalar pass in lane 0 for exact ordering
            float maxval = -INFINITY, sumval = 0.0f;
            if (lid == 0) {
                for (size_t i = 0; i < V; i++) {
                    float v = (float)x[i];
                    float old = maxval;
                    if (v > maxval) { maxval = v; }
                    sumval *= sycl::exp(old - maxval);
                    sumval += sycl::exp(v - maxval);
                }
            }
            maxval = sycl::group_broadcast(grp, maxval, 0);
            sumval = sycl::group_broadcast(grp, sumval, 0);
            float scale = 1.0f / sumval;
            if (lid == 0) {
                float prob = sycl::exp((float)x[ix] - maxval) * scale;
                losses[row] -= sycl::log(prob);
            }
            it.barrier(sycl::access::fence_space::global_space);
            if (WriteDLogits) {
                floatX* lr = logits + row * P;
                for (size_t i = lid; i < V; i += lsz) {
                    float prob = sycl::exp((float)x[i] - maxval) * scale;
                    float indicator = ((int)i == ix) ? 1.0f : 0.0f;
                    lr[i] = (floatX)((prob - indicator) * dloss);
                }
                // zero the padding (CUDA leaves it; we keep exact V semantics, pad untouched
                // except ensure deterministic zeros for the grads region beyond V)
                for (size_t i = lid + V; i < P; i += lsz) {
                    lr[i] = (floatX)(0.0f * dloss);
                }
            }
        }).wait();
}

// ----------------------------------------------------------------------------
// kernel launchers

// replaces logits with logit gradients
template <typename Type, bool WriteDLogits>
inline void fused_classifier(Type* logits, float* losses,
                             const float dloss, const int* targets,
                             size_t B, size_t T, size_t V, size_t P,
                             std::bool_constant<WriteDLogits> write_dlogits) {
    NVTX_RANGE_FN();
    (void)write_dlogits;
    fused_classifier_kernel<WriteDLogits>(logits, losses, dloss, targets, B, T, V, P);
}

#endif
