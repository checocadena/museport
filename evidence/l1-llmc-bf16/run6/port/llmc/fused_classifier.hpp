/*
Fused Classifier (ported from llmc/fused_classifier.cuh to SYCL 2020).
Forwards cross-entropy loss; optionally replaces logits with dlogits.
One work-group per row; SYCL group algorithms for max/sum reductions.
*/
#ifndef SYCL_FUSED_CLASSIFIER_HPP
#define SYCL_FUSED_CLASSIFIER_HPP

// llmc internal imports
#include "sycl_common.h"
#include "sycl_utils.hpp"

// ----------------------------------------------------------------------------

struct SoftmaxParams {
    float Scale;
    float Offset;
};

// replaces logits with logit gradients if WriteDLogits
template <bool WriteDLogits>
inline void fused_classifier_kernel(floatX* logits, float* losses,
                                const float dloss, const int* targets,
                                int B, int T, int V, int P, sycl::queue& q_) {
    size_t N = size_t(B) * T;
    if (N == 0) return;
    constexpr size_t LS = 128;
    floatX* L = logits; float* Ls = losses; const int* Tg = targets;
    float dl = dloss; int Vv = V, Pp = P;
    q_.parallel_for(sycl::nd_range<1>(N * LS, LS), [=](sycl::nd_item<1> it) {
        size_t idx = it.get_group(0);
        size_t lid = it.get_local_id(0);
        auto grp = it.get_group();
        int ix = Tg[idx];
        const floatX* xr = L + idx * (size_t)Pp;
        // online softmax max/sum over V (padded P layout, only first V count)
        float tmax = -INFINITY, tsum = 0.f;
        for (int i = int(lid); i < Vv; i += int(LS)) {
            float v = (float)xr[i];
            float old = tmax;
            tmax = sycl::fmax(tmax, v);
            tsum *= sycl::exp(old - tmax);
            tsum += sycl::exp(v - tmax);
        }
        float bmax = sycl::reduce_over_group(grp, tmax, sycl::maximum<float>());
        tsum *= sycl::exp(tmax - bmax);
        float bsum = sycl::reduce_over_group(grp, tsum, sycl::plus<float>());
        float scale = 1.f / bsum, off = bmax;
        if (lid == 0) {
            float prob = sycl::exp((float)xr[ix] - off) * scale;
            Ls[idx] -= sycl::log(prob);
        }
        it.barrier(sycl::access::fence_space::local_space);
        // gradients (must wait for loss read of logits row)
        for (int i = int(lid); i < Vv; i += int(LS)) {
            float prob = sycl::exp((float)xr[i] - off) * scale;
            float indicator = (i == ix) ? 1.0f : 0.0f;
            if (WriteDLogits) {
                L[idx * (size_t)Pp + i] = (floatX)((prob - indicator) * dl);
            }
        }
    }).wait();
}

// ----------------------------------------------------------------------------

// replaces logits with logit gradients
template <typename Type, bool WriteDLogits>
inline void fused_classifier(Type* logits, float* losses,
                      const float dloss, const int* targets,
                      int B, int T, int V, int P, std::bool_constant<WriteDLogits> write_dlogits, sycl::queue& q_) {
    (void)write_dlogits;
    NVTX_RANGE_FN();
    fused_classifier_kernel<WriteDLogits>((floatX*)logits, losses, dloss, targets, B, T, V, P, q_);
}

#endif
