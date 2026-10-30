/*
Fused Classifier (SYCL port of fused_classifier.cuh):
- Forwards the Cross Entropy Loss
- Never materializes the full normalized logits
- Kicks off the backward pass (logits -> dlogits) when WriteDLogits=true
*/
#ifndef FUSED_CLASSIFIER_SYCL_H
#define FUSED_CLASSIFIER_SYCL_H

#include <sycl/sycl.hpp>
// llmc internal imports
#include "sycl_common.h"
#include "sycl_utils.h"

// replaces logits with logit gradients when WriteDLogits=true
template <typename Type, bool WriteDLogits>
inline void fused_classifier(Type* logits, float* losses,
                             const float dloss, const int* targets,
                             int B, int T, int V, int P, std::bool_constant<WriteDLogits>, sycl::queue& q_) {
    NVTX_RANGE_FN();
    size_t N = (size_t)B * T;
    // one work-group per row
    constexpr size_t WG = 32;
    q_.parallel_for(sycl::nd_range<1>(sycl::range<1>(N * WG), sycl::range<1>(WG)),
        [=](sycl::nd_item<1> item) {
        // reverse order not needed for correctness; keep natural order
        size_t idx = item.get_group(0);
        if (idx >= N) { return; }
        size_t lid = item.get_local_id(0);
        auto grp = item.get_group();
        int ix = targets[idx];
        const Type* row = logits + idx * (size_t)P;
        // online softmax max/sum over V (padded P, ignore padding)
        const float NEG_INF = -3.402823466e+38f;
        float tmax = NEG_INF, tsum = 0.f;
        for (int i = (int)lid; i < V; i += (int)WG) {
            float v = (float)row[i];
            float old = tmax;
            tmax = sycl::fmax(tmax, v);
            tsum *= sycl::exp(old - tmax);
            tsum += sycl::exp(v - tmax);
        }
        float bmax = sycl::reduce_over_group(grp, tmax, sycl::maximum<float>());
        tsum *= sycl::exp(tmax - bmax);
        float bsum = sycl::reduce_over_group(grp, tsum, sycl::plus<float>());
        float scale = 1.f / bsum;
        float offset = bmax;
        if (lid == 0) {
            float prob = sycl::exp((float)row[ix] - offset) * scale;
            losses[idx] -= sycl::log(prob);
        }
        sycl::group_barrier(item.get_group());
        Type* wrow = logits + idx * (size_t)P;
        for (int i = (int)lid; i < V; i += (int)WG) {
            float prob = sycl::exp((float)row[i] - offset) * scale;
            float indicator = (i == ix) ? 1.0f : 0.0f;
            float dlogit = (prob - indicator) * dloss;
            if constexpr (WriteDLogits) {
                wrow[i] = (Type)dlogit;
            }
        }
        // zero the padding explicitly when writing dlogits (keeps buffer clean)
        if constexpr (WriteDLogits) {
            for (int i = V + (int)lid; i < P; i += (int)WG) {
                wrow[i] = (Type)0.f;
            }
        }
    }).wait();
}

#endif
