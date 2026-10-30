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
// One work-group (256) per row: lanes cooperate on an online softmax over the
// vocabulary (strided x128 loads), group-reduce to the row constants, then
// cooperatively rewrite the row with logit gradients. The max reduction is
// exact; the sum reduction order differs from a single-thread loop in the last
// ulp only (losses move by ~1e-8, dlogits flip a scattered bf16 ulp at most).
template <bool WriteDLogits = true>
inline void fused_classifier_kernel(floatX* logits, float* losses,
                                const float dloss, const int* targets,
                                int B, int T, int V, int P, std::bool_constant<WriteDLogits>, sycl::queue& q) {
    size_t N = (size_t)B * T;
    if (N == 0) { return; }
    constexpr size_t WG = 256;
    constexpr int VEC = (int)x128::size; // 8 packed bf16 per 128-bit load
    q.parallel_for(sycl::nd_range<1>(N * WG, WG), [=](sycl::nd_item<1> it) {
        size_t grp = it.get_group(0);
        size_t lid = it.get_local_id(0);
        sycl::group<1> g = it.get_group();
        // Reverse row order, like the CUDA kernel (cache behaviour only; results identical).
        size_t idx = N - 1 - grp;
        int ix = targets[idx];
        const floatX* x = logits + idx * (size_t)P;
        // pass 1: online max+sum over full vecs, strided by work-group size
        int nfull = V / VEC;
        float maxval = -INFINITY;
        float sumval = 0.0f;
        for (int i = (int)lid; i < nfull; i += (int)WG) {
            x128 packed = load128cs(x + (size_t)i * VEC);
            for (int k = 0; k < VEC; ++k) {
                float v = (float)packed[k];
                float old = maxval;
                maxval = fmaxf(maxval, v);
                sumval *= sycl::exp(old - maxval);
                sumval += sycl::exp(v - maxval);
            }
        }
        // remainder elements past the last full vec
        int unaligned_start = V & ~(VEC - 1);
        for (int i = unaligned_start + (int)lid; i < V; i += (int)WG) {
            float v = (float)x[i];
            float old = maxval;
            maxval = fmaxf(maxval, v);
            sumval *= sycl::exp(old - maxval);
            sumval += sycl::exp(v - maxval);
        }
        float gmax = sycl::reduce_over_group(g, maxval, sycl::maximum<float>());
        sumval *= sycl::exp(maxval - gmax);
        float gsum = sycl::reduce_over_group(g, sumval, sycl::plus<float>());
        float scale = 1.f / gsum;
        float offset = gmax;
        if (lid == 0) {
            float prob = sycl::exp((float)x[ix] - offset) * scale;
            losses[idx] -= sycl::log(prob);
        }
        sycl::group_barrier(g); // dlogits below overwrite the logits read above
        floatX* dlogits_vec = logits + idx * (size_t)P;
        for (int i = (int)lid; i < nfull; i += (int)WG) {
            x128 packed = load128cs(dlogits_vec + (size_t)i * VEC);
            x128 outv;
            for (int k = 0; k < VEC; ++k) {
                int element = i * VEC + k;
                float prob = sycl::exp((float)packed[k] - offset) * scale;
                float indicator = (element == ix) ? 1.0f : 0.0f;
                float d = (prob - indicator) * dloss;
                outv[k] = WriteDLogits ? (floatX)d : packed[k];
            }
            if (WriteDLogits) {
                store128(dlogits_vec + (size_t)i * VEC, outv);
            }
        }
        for (int i = unaligned_start + (int)lid; i < V; i += (int)WG) {
            float prob = sycl::exp((float)dlogits_vec[i] - offset) * scale;
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
