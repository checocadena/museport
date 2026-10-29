/*
Fused Classifier:
- Forwards the Cross Entropy Loss
- Never materializes the full normalized logits, only at the target label
- (fusion) Also kicks off the backward pass, because everything is already loaded
SYCL 2020 port of fused_classifier.cuh.
Fire-and-forget launch on the in-order queue (no per-kernel wait).
*/
#ifndef SYCL_FUSED_CLASSIFIER_H
#define SYCL_FUSED_CLASSIFIER_H

// llmc internal imports
#include "sycl_common.h"
#include "sycl_utils.hpp"

// ----------------------------------------------------------------------------
// SYCL kernels

struct SoftmaxParams {
    float Scale;
    float Offset;
};

// Compute per-row softmax normalisation constants with online softmax.
// Deterministic: sequential accumulation over the row in index order.
inline SoftmaxParams prepare_softmax_row(size_t idx, const floatX* inp, int V, int P) {
    const floatX* x = inp + idx * (size_t)P;
    float thread_maxval = -INFINITY;
    float thread_sumval = 0.0f;
    for (int i = 0; i < V; i++) {
        float v = (float)x[i];
        float old_maxval = thread_maxval;
        thread_maxval = fmaxf(thread_maxval, v);
        thread_sumval *= sycl::exp(old_maxval - thread_maxval);
        thread_sumval += sycl::exp(v - thread_maxval);
    }
    return SoftmaxParams{1.f / thread_sumval, thread_maxval};
}

// will _update_ logits to logit gradients
template <bool WriteDLogits = true>
inline void fused_classifier_kernel(floatX* logits, float* losses,
                                const float dloss, const int* targets,
                                int B, int T, int V, int P, std::bool_constant<WriteDLogits>, sycl::queue& q) {
    size_t N = (size_t)B * T;
    // Reverse row order, like the CUDA kernel (cache behaviour only; results identical).
    q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> id) {
        size_t fwd = id[0];
        size_t idx = N - 1 - fwd;
        int ix = targets[idx];
        SoftmaxParams sp = prepare_softmax_row(idx, logits, V, P);
        {
            float prob = sycl::exp((float)logits[idx * (size_t)P + ix] - sp.Offset) * sp.Scale;
            losses[idx] -= sycl::log(prob);
        }
        const floatX* logits_vec = logits + idx * (size_t)P;
        floatX* dlogits_vec = logits + idx * (size_t)P;
        int vec = (int)x128::size;
        int nfull = V / vec;
        for (int i = 0; i < nfull; i++) {
            for (int k = 0; k < vec; ++k) {
                int element = i * vec + k;
                float prob = sycl::exp((float)logits_vec[element] - sp.Offset) * sp.Scale;
                float indicator = (element == ix) ? 1.0f : 0.0f;
                if (WriteDLogits) {
                    dlogits_vec[element] = (floatX)((prob - indicator) * dloss);
                }
            }
        }
        int unaligned_start = V & ~(vec - 1);
        for (int i = unaligned_start; i < V; i++) {
            float prob = sycl::exp((float)logits_vec[i] - sp.Offset) * sp.Scale;
            float indicator = (i == ix) ? 1.0f : 0.0f;
            float dlogit = (prob - indicator) * dloss;
            if (WriteDLogits) {
                dlogits_vec[i] = (floatX)dlogit;
            }
        }
    });
}

// ----------------------------------------------------------------------------
// kernel launchers

// replaces logits with logit gradients
template <typename Type, bool WriteDLogits>
inline void fused_classifier(Type* logits, float* losses,
                      const float dloss, const int* targets,
                      int B, int T, int V, int P, std::bool_constant<WriteDLogits> write_dlogits, sycl::queue& q) {
    NVTX_RANGE_FN();
    fused_classifier_kernel(logits, losses, dloss, targets, B, T, V, P, write_dlogits, q);
}

#endif
