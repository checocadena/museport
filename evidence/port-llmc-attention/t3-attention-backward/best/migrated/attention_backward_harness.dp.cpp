/*
Kernels for attention backward pass.

Compile example:
nvcc -O3 --use_fast_math -lcublas -lcublasLt attention_backward.cu -o attention_backward

version 1 is a naive first version
OMP_NUM_THREADS=32 ./attention_backward 1

version 2 much ensures better load-balancing by having independent threads for each batch and attention head
OMP_NUM_THREADS=32 ./attention_backward 2

version 3 uses a full warp to calculate each result (instead of a thread), which enables coalesced memory access
OMP_NUM_THREADS=32 ./attention_backward 3

version 4 improves data reuse in registers by doing 8 values of t3 in one warp.
OMP_NUM_THREADS=32 ./attention_backward 4

version 5 reduces the amount of non-fp32 instructions needed by avoiding ifs
OMP_NUM_THREADS=32 ./attention_backward 5
*/

#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <float.h>
#include <dpct/blas_utils.hpp>

#include "common.h"
#include <cmath>

// ----------------------------------------------------------------------------
// CPU code reference

/*
NOTE:
This version of attention_forward is modified to be consistent with the
attention_forward GPU kernel in the following way small but important way:
- preatt is only QUERY @ KEY, without the scale
- the scale instead moved and fused into the softmax
- the full preatt matrix is materialized, even the parts that get masked out
    - this doesn't actually change anything due to masking, but it lets us
      easily compare to the GPU version, which also does the full, dense sgemm
In this way we'll be able to make sure that preatt and att agree CPU vs GPU
*/
void attention_forward_cpu(float* out, float* preatt, float* att,
                            float* inp,
                            int B, int T, int C, int NH) {
    // input is (B, T, 3C) holding the query, key, value (Q, K, V) vectors
    // preatt, att are (B, NH, T, T). NH = number of heads, T = sequence length
    // that holds the pre-attention and post-attention scores (used in backward)
    // output is (B, T, C)
    // attention is the only layer that mixes information across time
    // every other operation is applied at every (b,t) position independently
    // (and of course, no layer mixes information across batch)
    int C3 = C*3;
    int hs = C / NH; // head size
    float scale = 1.0 / sqrtf(hs);

    #pragma omp parallel for collapse(3)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < NH; h++) {
                float* query_t = inp + b * T * C3 + t * C3 + h * hs;
                float* preatt_bth = preatt + b*NH*T*T + h*T*T + t*T;
                float* att_bth = att + b*NH*T*T + h*T*T + t*T;

                // pass 1: calculate query dot key and maxval
                float maxval = -FLT_MAX;
                for (int t2 = 0; t2 < T; t2++) { // used to be t2 <= t
                    float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key

                    // (query_t) dot (key_t2)
                    float val = 0.0f;
                    for (int i = 0; i < hs; i++) {
                        val += query_t[i] * key_t2[i];
                    }
                    if (val > maxval) {
                        maxval = val;
                    }

                    preatt_bth[t2] = val;
                }

                // pass 2: calculate the exp and keep track of sum
                // maxval is being calculated and subtracted only for numerical stability
                float expsum = 0.0f;
                for (int t2 = 0; t2 <= t; t2++) {
                    float expv = expf(scale * (preatt_bth[t2] - maxval));
                    expsum += expv;
                    att_bth[t2] = expv;
                }
                float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

                // pass 3: normalize to get the softmax
                for (int t2 = 0; t2 < T; t2++) {
                    if (t2 <= t) {
                        att_bth[t2] *= expsum_inv;
                    } else {
                        // causal attention mask. not strictly necessary to set to zero here
                        // only doing this explicitly for debugging and checking to PyTorch
                        att_bth[t2] = 0.0f;
                    }
                }

                // pass 4: accumulate weighted values into the output of attention
                float* out_bth = out + b * T * C + t * C + h * hs;
                for (int i = 0; i < hs; i++) { out_bth[i] = 0.0f; }
                for (int t2 = 0; t2 <= t; t2++) {
                    float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
                    float att_btht2 = att_bth[t2];
                    for (int i = 0; i < hs; i++) {
                        out_bth[i] += att_btht2 * value_t2[i];
                    }
                }
            }
        }
    }
}

// NOTE: Also contains the re-shuffling of the exact position of "scale"
// and when it is applied (after preatt, not "during" preatt)
// also, full matrices are materialized, even the parts that get masked out
// museport: CPU reference attention_backward_cpu() removed (it is the hidden check)

// ----------------------------------------------------------------------------
// GPU kernels
// the forward pass that is the sequence [permute, sgemm, softmax, sgemm, unpermute]

void permute_kernel(float* q, float* k, float* v,
                               const float* inp,
                               int B, int N, int NH, int d) {
    // okay so now, this kernel wants Q,K,V to all be of shape (B, NH, N, d)
    // but instead, we have a single tensor QKV (inp) of shape (B, N, 3, NH, d)
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
              item_ct1.get_local_id(2);

    // Q[b][nh_][n][d_] = inp[b][n][0][nh_][d_]
    if (idx < B * NH * N * d) {
        int b = idx / (NH * N * d);
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);
        rest = rest % (N * d);
        int n = rest / d;
        int d_ = rest % d;

        int inp_idx = (b * N * 3 * NH * d) + (n * 3 * NH * d) + (0 * NH * d) + (nh_ * d) + d_;
        q[idx] = inp[inp_idx];
        k[idx] = inp[inp_idx + NH * d];
        v[idx] = inp[inp_idx + 2 * (NH * d)];
    }
}

void permute_kernel_backward(float* dinp,
                                        const float* dq, const float* dk, const float* dv,
                                        int B, int N, int NH, int d) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
              item_ct1.get_local_id(2);
    if (idx < B * NH * N * d) {
        int b = idx / (NH * N * d);
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);
        rest = rest % (N * d);
        int n = rest / d;
        int d_ = rest % d;

        int inp_idx = (b * N * 3 * NH * d) + (n * 3 * NH * d) + (0 * NH * d) + (nh_ * d) + d_;
        dinp[inp_idx] += dq[idx];
        dinp[inp_idx + NH * d] += dk[idx];
        dinp[inp_idx + 2 * (NH * d)] += dv[idx];
    }
}

void unpermute_kernel(const float* inp, float *out, int B, int N, int NH, int d) {
   // out has shape (B, nh, N, d) but we need to unpermute it to (B, N, nh, d)
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
              item_ct1.get_local_id(2);

    // out[b][n][nh_][d_] <- inp[b][nh_][n][d_]
    if (idx < B * NH * N * d) {
        int b = idx / (NH * N * d);
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);
        rest = rest % (N * d);
        int n = rest / d;
        int d_ = rest % d;

        int other_idx = (b * NH * N * d) + (n * NH * d) + (nh_ * d) + d_;
        out[other_idx] = inp[idx];
    }
}

void unpermute_kernel_backward(float* dinp, const float *dout, int B, int N, int NH, int d) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
              item_ct1.get_local_id(2);
    if (idx < B * NH * N * d) {
        int b = idx / (NH * N * d);
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);
        rest = rest % (N * d);
        int n = rest / d;
        int d_ = rest % d;

        int other_idx = (b * NH * N * d) + (n * NH * d) + (nh_ * d) + d_;
        dinp[idx] += dout[other_idx];
    }
}

float &vec_at(sycl::float4 &vec, int index) {
    return reinterpret_cast<float*>(&vec)[index];
}

float vec_at(const sycl::float4 &vec, int index) {
    return reinterpret_cast<const float*>(&vec)[index];
}

/*
DPCT1110:1: The total declared local variable size in device function
softmax_forward_kernel5 exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
void softmax_forward_kernel5(float *out, float inv_temperature,
                             const float *inp, int N, int T) {
    // inp, out shape: (N, T, T), where N = B * NH
    // fuses the multiplication by scale inside attention
    // directly autoregressive, so we only compute the lower triangular part
    // uses the online softmax algorithm
    assert(T % 4  == 0);

    sycl::group<3> block =
        sycl::ext::oneapi::this_work_item::get_work_group<3>();
    sycl::sub_group warp = sycl::ext::oneapi::this_work_item::get_sub_group();
    int idx = sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_group(2) *
                  sycl::ext::oneapi::this_work_item::get_sub_group()
                      .get_group_linear_range() +
              sycl::ext::oneapi::this_work_item::get_sub_group()
                  .get_group_linear_id();
    if(idx >= N * T) {
        return;
    }
    int own_pos = idx % T;
    int pos_by_4 = own_pos / 4;

    // one row of inp, i.e. inp[idx, :] of shape (T,)
    const float* x = inp + idx * T;

    // not INF, so we don't get NaNs accidentally when subtracting two values.
    float maxval = -FLT_MAX;
    float sumval = 0.0f;

    const sycl::float4 *x_vec = reinterpret_cast<const sycl::float4 *>(x);
    for (int i = sycl::ext::oneapi::this_work_item::get_sub_group()
                     .get_local_linear_id();
         i < pos_by_4; i += sycl::ext::oneapi::this_work_item::get_sub_group()
                                .get_local_linear_range()) {
        sycl::float4 v = x_vec[i];
        float old_maxval = maxval;
        for(int k = 0; k < 4; ++k) {
            maxval = sycl::fmax(maxval, vec_at(v, k));
        }
        sumval *= sycl::native::exp(inv_temperature * (old_maxval - maxval));
        for(int k = 0; k < 4; ++k) {
            sumval +=
                sycl::native::exp(inv_temperature * (vec_at(v, k) - maxval));
        }
    }

    if (4 * pos_by_4 + sycl::ext::oneapi::this_work_item::get_sub_group()
                           .get_local_linear_id() <=
        own_pos) {
        float old_maxval = maxval;
        maxval = sycl::fmax(
            maxval,
            x[4 * pos_by_4 + sycl::ext::oneapi::this_work_item::get_sub_group()
                                 .get_local_linear_id()]);
        sumval *= sycl::native::exp(inv_temperature * (old_maxval - maxval));
        sumval += sycl::native::exp(
            inv_temperature *
            (x[4 * pos_by_4 + sycl::ext::oneapi::this_work_item::get_sub_group()
                                  .get_local_linear_id()] -
             maxval));
    }

    float global_maxval = sycl::reduce_over_group(
        sycl::ext::oneapi::this_work_item::get_sub_group(), maxval,
        sycl::maximum<float>{});
    sumval *= sycl::native::exp(inv_temperature * (maxval - global_maxval));

    float sum = sycl::reduce_over_group(
        sycl::ext::oneapi::this_work_item::get_sub_group(), sumval,
        sycl::plus<float>{});
    float norm = 1.f / sum;

    // divide the whole row by the sum
    for (int i = sycl::ext::oneapi::this_work_item::get_sub_group()
                     .get_local_linear_id();
         i <= own_pos; i += sycl::ext::oneapi::this_work_item::get_sub_group()
                                .get_local_linear_range()) {
        // recalculation is faster than doing the round-trip through memory.
        /*
        DPCT1098:36: The '*' expression is used instead of the __ldcs call.
        These two expressions do not provide the exact same functionality. Check
        the generated code for potential precision and/or performance issues.
        */
        float ev =
            sycl::native::exp(inv_temperature * (*(x + i) - global_maxval));
        /*
        DPCT1098:35: The '=' expression is used instead of the __stcs call.
        These two expressions do not provide the exact same functionality. Check
        the generated code for potential precision and/or performance issues.
        */
        *(out + idx * T + i) = ev * norm;
    }
}

// naive kernel to backward through an autoregressive softmax, just to get correctness
void softmax_autoregressive_backward_kernel1(float* dpreatt, const float* datt, const float* att,
                                                     int B, int T, int C, int NH) {
    // dpreatt, datt, att are all (B, NH, T, T)
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int t3 = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
             item_ct1.get_local_id(2);
    if (t3 < T) {
        int hs = C / NH; // head size
        float scale = 1.0f / sycl::sqrt((float)hs);
        for (int b = 0; b < B; b++) {
            for (int h = 0; h < NH; h++) {
                for (int t = t3; t < T; t++) {
                    const float* att_bth = att + b*NH*T*T + h*T*T + t*T;
                    const float* datt_bth = datt + b*NH*T*T + h*T*T + t*T;
                    float* dpreatt_bth = dpreatt + b*NH*T*T + h*T*T + t*T;
                    float accum = 0.0f;
                    for (int t2 = 0; t2 <= t; t2++) {
                        float indicator = t2 == t3 ? 1.0f : 0.0f;
                        float local_derivative = att_bth[t2] * (indicator - att_bth[t3]);
                        accum +=  scale * local_derivative * datt_bth[t2];
                    }
                    dpreatt_bth[t3] = accum;
                }
            }
        }
    }
}

// parallelize across t,b,h
void softmax_autoregressive_backward_kernel2(float* dpreatt, const float* datt, const float* att,
                                                     int B, int T, int C, int NH) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int t3 = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
             item_ct1.get_local_id(2);
    int idx = item_ct1.get_group(1) * T * T;
    if (t3 >= T) { return; }

    int hs = C / NH; // head size
    float scale = 1.0f / sycl::sqrt((float)hs);
    for (int t = t3; t < T; t++) {
        float result = 0.0;
        const float* att_bth = att + idx + t*T;
        const float* datt_bth = datt + idx + t*T;
        float* dpreatt_bth = dpreatt + idx + t*T;

        for (int t2 = 0; t2 <= t; t2++) {
            float indicator = t2 == t3 ? 1.0f : 0.0f;
            float local_derivative = att_bth[t2] * (indicator - att_bth[t3]);
            result += scale * local_derivative * datt_bth[t2];
        }

        dpreatt_bth[t3] = result;
    }
}

// parallelize across t,b,h
void softmax_autoregressive_backward_kernel3(float* dpreatt, const float* datt, const float* att,
                                                     int B, int T, int C, int NH) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    sycl::group<3> block =
        sycl::ext::oneapi::this_work_item::get_work_group<3>();
    sycl::sub_group warp = sycl::ext::oneapi::this_work_item::get_sub_group();
    int t3 = item_ct1.get_group(2) *
                 sycl::ext::oneapi::this_work_item::get_sub_group()
                     .get_group_linear_range() +
             sycl::ext::oneapi::this_work_item::get_sub_group()
                 .get_group_linear_id();

    int idx = item_ct1.get_group(1) * T * T;
    if (t3 >= T) { return; }

    int hs = C / NH; // head size
    float scale = 1.0f / sycl::sqrt((float)hs);
    for (int t = t3; t < T; t++) {
        float result = 0.0;
        const float* att_bth = att + idx + t*T;
        const float* datt_bth = datt + idx + t*T;
        float* dpreatt_bth = dpreatt + idx + t*T;
        const float att_at_t3 = att_bth[t3];

        for (int t2 = sycl::ext::oneapi::this_work_item::get_sub_group()
                          .get_local_linear_id();
             t2 <= t; t2 += sycl::ext::oneapi::this_work_item::get_sub_group()
                                .get_local_linear_range()) {
            float indicator = t2 == t3 ? 1.0f : 0.0f;
            float local_derivative = att_bth[t2] * (indicator - att_at_t3);
            result += local_derivative * datt_bth[t2];
        }

        result = sycl::reduce_over_group(
            sycl::ext::oneapi::this_work_item::get_sub_group(), result,
            sycl::plus<float>());
        if (sycl::ext::oneapi::this_work_item::get_sub_group()
                .get_local_linear_id() == 0) {
            dpreatt_bth[t3] = scale * result;
        }
    }
}
/*
DPCT1110:2: The total declared local variable size in device function
softmax_autoregressive_backward_kernel4 exceeds 128 bytes and may cause high
register pressure. Consult with your hardware vendor to find the total register
size available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
void softmax_autoregressive_backward_kernel4(float *__restrict__ dpreatt,
                                             const float *__restrict__ datt,
                                             const float *__restrict__ att,
                                             int B, int T, int C, int NH) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int UNROLL = 8;

    sycl::group<3> block =
        sycl::ext::oneapi::this_work_item::get_work_group<3>();
    sycl::sub_group warp = sycl::ext::oneapi::this_work_item::get_sub_group();
    int t3 = UNROLL * (item_ct1.get_group(2) *
                           sycl::ext::oneapi::this_work_item::get_sub_group()
                               .get_group_linear_range() +
                       sycl::ext::oneapi::this_work_item::get_sub_group()
                           .get_group_linear_id());

    int idx = item_ct1.get_group(1) * T * T;
    if (t3 >= T) { return; }

    int hs = C / NH; // head size
    float scale = 1.0f / sycl::sqrt((float)hs);

    // the innermost loop combines different values of t2 with different values of t.
    // by handling [t3, t3 + UNROLL) in one thread, we get much better memory reuse:
    // any t3/t-dependent value can be loaded once before the t2 loop.
    // within the t2 loop, we can combine each loaded value with each of the UNROLL
    // pre-loaded values, thus cutting memory ready by a factor of ~UNROLL.

    // one iteration of this loop has to handle the cases
    // this may lead to some invalid indices; therefore, we have several
    // early-outs in the iteration over k below.
    for (int t = t3; t < T; t++) {
        float result[UNROLL] = {};
        const float* att_bth = att + idx + t * T;
        const float* datt_bth = datt + idx + t * T;
        float* dpreatt_bth = dpreatt + idx + t * T;

        float att_at_t3[UNROLL];
        for(int k = 0; k < UNROLL; ++k) {
            if (t < t3 + k) continue;
            att_at_t3[k] = att_bth[t3 + k];
        }

        for (int t2 = sycl::ext::oneapi::this_work_item::get_sub_group()
                          .get_local_linear_id();
             t2 <= t; t2 += sycl::ext::oneapi::this_work_item::get_sub_group()
                                .get_local_linear_range()) {
            float att_t2 = att_bth[t2];
            float datt_t2 = datt_bth[t2];
            for(int k = 0; k < UNROLL; ++k) {
                if (t < t3 + k) continue;
                float indicator = t2 == (t3 + k) ? 1.0f : 0.0f;
                float local_derivative = att_t2 * (indicator - att_at_t3[k]);
                result[k] += local_derivative * datt_t2;
            }
        }

        for(int k = 0; k < UNROLL; ++k) {
            result[k] = sycl::reduce_over_group(
                sycl::ext::oneapi::this_work_item::get_sub_group(), result[k],
                sycl::plus<float>());
        }
        if (sycl::ext::oneapi::this_work_item::get_sub_group()
                .get_local_linear_id() < UNROLL) {
            dpreatt_bth[t3 + sycl::ext::oneapi::this_work_item::get_sub_group()
                                 .get_local_linear_id()] =
                scale *
                result[sycl::ext::oneapi::this_work_item::get_sub_group()
                           .get_local_linear_id()];
        }
    }
}

/*
DPCT1110:3: The total declared local variable size in device function
softmax_autoregressive_backward_kernel5 exceeds 128 bytes and may cause high
register pressure. Consult with your hardware vendor to find the total register
size available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
void softmax_autoregressive_backward_kernel5(float *__restrict__ dpreatt,
                                             const float *__restrict__ datt,
                                             const float *__restrict__ att,
                                             int B, int T, int C, int NH) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int UNROLL = 8;

    sycl::group<3> block =
        sycl::ext::oneapi::this_work_item::get_work_group<3>();
    sycl::sub_group warp = sycl::ext::oneapi::this_work_item::get_sub_group();
    int t3 = UNROLL * (item_ct1.get_group(2) *
                           sycl::ext::oneapi::this_work_item::get_sub_group()
                               .get_group_linear_range() +
                       sycl::ext::oneapi::this_work_item::get_sub_group()
                           .get_group_linear_id());

    int idx = item_ct1.get_group(1) * T * T;
    if (t3 >= T) { return; }

    int hs = C / NH; // head size
    float scale = 1.0f / sycl::sqrt((float)hs);
    for (int t = t3; t < T; t++) {
        float result[UNROLL] = {};
        const float* att_bth = att + idx + t * T;
        const float* datt_bth = datt + idx + t * T;
        float* dpreatt_bth = dpreatt + idx + t * T;

        float att_at_t3[UNROLL];
        for(int k = 0; k < UNROLL; ++k) {
            // if t < t3+k, we're out of bounds.
            // in that case, we don't care what we read, because later on,
            // we won't write the corresponding result. So just clip to
            // make sure this is a valid (in-bounds) memory access.
            att_at_t3[k] = att_bth[sycl::min(t, t3 + k)];
        }

        // the code below is actually just a for loop; except,
        // we have to do something special in one iteration in
        // the middle, and an if turned out to have significant
        // performance impact.
        // so we split the loop in three parts. Ugly, but effective.

        // the beginning/end loop does the same thing, so we write the code
        // just once in a lambda. In this step, we're guaranteed that
        // indicator == 0
        auto loop_step = [&](int t2){
            float p = att_bth[t2] * datt_bth[t2];
            for (int k = 0; k < UNROLL; ++k) {
                result[k] -= p * att_at_t3[k];
            }
        };

        // Now the actual loop.
        {
            // declare the loop iterator. Needs to be kept across the
            // three different parts, so it's not a local variable in
            // the for loop.
            int t2 = sycl::ext::oneapi::this_work_item::get_sub_group()
                         .get_local_linear_id();

            // first part, as long as t2 < t3, indicator == 0
            for (; t2 < t3;
                 t2 += sycl::ext::oneapi::this_work_item::get_sub_group()
                           .get_local_linear_range()) {
                loop_step(t2);
            }

            // because k <= warp.size() (==32), the event that t3+k == t2
            // has to happen at this particular step.
            static_assert(UNROLL <= 32, "UNROLL is too large, this won't produce correct results.");
            if (t2 <= t) {
                float att_t2 = att_bth[t2];
                float datt_t2 = datt_bth[t2];
                float p = att_t2 * datt_t2;
                for (int k = 0; k < UNROLL; ++k) {
                    float indicator = t2 == (t3 + k) ? 1.0f : 0.0f;
                    result[k] += p * (indicator - att_at_t3[k]);
                }
                t2 += sycl::ext::oneapi::this_work_item::get_sub_group()
                          .get_local_linear_range();
            }

            // rest of the loop, indicator == 0 again
            for (; t2 <= t;
                 t2 += sycl::ext::oneapi::this_work_item::get_sub_group()
                           .get_local_linear_range()) {
                loop_step(t2);
            }
        }

        for(int k = 0; k < UNROLL; ++k) {
            result[k] = sycl::reduce_over_group(
                sycl::ext::oneapi::this_work_item::get_sub_group(), result[k],
                sycl::plus<float>());
        }

        // when storing, we need to check that this is actually a valid result.
        // here, warp.thread_rank() corresponds to `k` in the previous loops.
        if (sycl::ext::oneapi::this_work_item::get_sub_group()
                    .get_local_linear_id() < UNROLL &&
            t >= t3 + sycl::ext::oneapi::this_work_item::get_sub_group()
                          .get_local_linear_id()) {
            dpreatt_bth[t3 + sycl::ext::oneapi::this_work_item::get_sub_group()
                                 .get_local_linear_id()] =
                scale *
                result[sycl::ext::oneapi::this_work_item::get_sub_group()
                           .get_local_linear_id()];
        }
    }
}


// I want `BlockSize` to be statically known to the compiler, thus we get a template here.
// This kernel takes a step back, and looks at the original CPU code again. We have some simple outer loops
// That are independent, (b, t, h), and then the inner loops over (t2, t3) where we're combining elements -- this is
// where we can reuse data and be more efficient
// => handle b, t, h  through block indices; each block does all the work for the (t2, t3) loop cooperatively.
// Now we have two nested loops, and in the inner instruction, we combine indexing from both => this calls for
// loop tiling, and lifting some of the memory ops out of the loop.
// We're in luck here;  if we tile so that t3 is the outer loop, we can get a sinlge write op per result, AND also cache
// the t2-indexed part of the computation, which is the problematic one because it contains a multiplication that now we
// do not have to repeat over and over.
// => do an outer t3 loop where each thread gets one t3 index. Then, do an outer t2 loop in steps of BlockSize, and
// prepare BlockSize many elements for the inner loop. Here, each thread calculates one element and stores it in shmem.
// Then, in the inner t2 loop, each thread reads *all* the elements previously stored and does its computations.
// This way, we do 3*BlockSize loads, but BlockSize^2 computation steps => This kernel is now entirely compute bound.
// To fix up the compute issues, as above, we replace ifs in memory reading with min, and also split the inner loop
// into a large region where we don't have to calculate the indicator, and a small, costly region where we do.
template<int BlockSize>
void softmax_autoregressive_backward_kernel6(float* dpreatt, const float* datt, const float* att,
                                                        int B, int T, int C, int NH,
                                                        float *att_bth_s) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    sycl::group<3> block =
        sycl::ext::oneapi::this_work_item::get_work_group<3>();

    int idx = item_ct1.get_group(1);
    int t = item_ct1.get_group(2);

    att += idx * T * T;
    datt += idx * T * T;
    dpreatt += idx * T * T;

    int hs = C / NH; // head size
    float scale = 1.0f / sycl::sqrt((float)hs);
    const float* att_bth = att + t * T;
    const float* datt_bth = datt + t * T;
    float* dpreatt_bth = dpreatt + t * T;

    int block_steps = ceil_div(t+1, BlockSize);
    // very important: This loop condition needs to be the same for all threads.
    // even if a thread later on is not going to do any work, it needs to participate in the
    // data loading process!
    for (int t3f = 0; t3f < block_steps; ++t3f) {
        int t3 = t3f * BlockSize + item_ct1.get_local_linear_id();
        float acc = 0.f;
        float at3 = att_bth[t3];
        for (int t2b = 0; t2b <= t; t2b += BlockSize) {
            int end = sycl::min(t + 1 - t2b, BlockSize);
            /*
            DPCT1065:4: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
            {
                int t2i = item_ct1.get_local_linear_id();
                int t2 = sycl::min(t, t2b + t2i);
                att_bth_s[t2i] = att_bth[t2] * datt_bth[t2];
            }

            /*
            DPCT1065:5: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
            if(t3f * BlockSize == t2b) {
                for (int t2i = 0; t2i < end; t2i++) {
                    int t2 = t2b + t2i;
                    float indicator = t2 == t3 ? 1.0f : 0.0f;
                    acc += att_bth_s[t2i] * (indicator - at3);
                }
            } else {
                for (int t2i = 0; t2i < end; t2i++) {
                    acc +=  att_bth_s[t2i] * (0.f - at3);
                }
            }
        }
        dpreatt_bth[t3] = scale * acc;
    }
}

// Actually disentangling the loops and simplifying the resulting math gives us this pretty nice kernel.
template<int BlockSize>
void softmax_autoregressive_backward_kernel7(float* dpreatt, const float* datt, const float* att,
                                                        int B, int T, int C, float scale,
                                                        float *block_acc) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    sycl::group<3> block =
        sycl::ext::oneapi::this_work_item::get_work_group<3>();
    sycl::sub_group warp = sycl::ext::oneapi::this_work_item::get_sub_group();

    int idx = item_ct1.get_group(1);
    int t = item_ct1.get_group(2);

    att += idx * T * T;
    datt += idx * T * T;
    dpreatt += idx * T * T;

    const float* att_bth = att + t * T;
    const float* datt_bth = datt + t * T;
    float* dpreatt_bth = dpreatt + t * T;

    if (sycl::ext::oneapi::this_work_item::get_sub_group()
            .get_group_linear_id() == 0) {
        block_acc[sycl::ext::oneapi::this_work_item::get_sub_group()
                      .get_local_linear_id()] = 0;
    }

    float local_sum = 0;
    for (int t2 = item_ct1.get_local_linear_id(); t2 <= t; t2 += BlockSize) {
        local_sum += att_bth[t2] * datt_bth[t2];
    }

    block_acc[sycl::ext::oneapi::this_work_item::get_sub_group()
                  .get_group_linear_id()] =
        sycl::reduce_over_group(
            sycl::ext::oneapi::this_work_item::get_sub_group(), local_sum,
            sycl::plus<float>{});
    /*
    DPCT1065:6: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    local_sum = sycl::reduce_over_group(
        sycl::ext::oneapi::this_work_item::get_sub_group(),
        block_acc[sycl::ext::oneapi::this_work_item::get_sub_group()
                      .get_local_linear_id()],
        sycl::plus<float>{});

    for (int t3 = item_ct1.get_local_linear_id(); t3 <= t; t3 += BlockSize) {
        float acc = att_bth[t3] * (datt_bth[t3] - local_sum);
        dpreatt_bth[t3] = scale * acc;
    }
}

// The slightly less pretty version of kernel 7. Adding in all the dirty tricks that can give us a few more percent
//  - streaming memory access instructions
//  - reordering blocks to prevent tail effect
//  - multiple values of T per block
template<int BlockSize>
void softmax_autoregressive_backward_kernel8(float* dpreatt, const float* datt, const float* att,
                                                        int B, int T, int C, float scale,
                                                        float *block_acc) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    constexpr int T_per_block = 4;
    sycl::group<3> block =
        sycl::ext::oneapi::this_work_item::get_work_group<3>();
    sycl::sub_group warp = sycl::ext::oneapi::this_work_item::get_sub_group();

    int idx = item_ct1.get_group(1);
    // go through blocks in reverse order, so the slowest block starts first
    int t0 = T - 1 - T_per_block * item_ct1.get_group(2);

    att += idx * T * T;
    datt += idx * T * T;
    dpreatt += idx * T * T;

    if (sycl::ext::oneapi::this_work_item::get_sub_group()
            .get_group_linear_id() == 0) {
        block_acc[sycl::ext::oneapi::this_work_item::get_sub_group()
                      .get_local_linear_id()] = 0;
    }

    for(int to = 0; to < T_per_block; ++to) {
        int t = t0 - to;
        if(t < 0) return;
        const float* att_bth = att + t * T;
        const float* datt_bth = datt + t * T;
        float* dpreatt_bth = dpreatt + t * T;

        float local_sum = 0;
        for (int t2 = item_ct1.get_local_linear_id(); t2 <= t;
             t2 += BlockSize) {
            local_sum += att_bth[t2] * datt_bth[t2];
        }

        block_acc[sycl::ext::oneapi::this_work_item::get_sub_group()
                      .get_group_linear_id()] =
            sycl::reduce_over_group(
                sycl::ext::oneapi::this_work_item::get_sub_group(), local_sum,
                sycl::plus<float>{});
        /*
        DPCT1065:7: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        local_sum = sycl::reduce_over_group(
            sycl::ext::oneapi::this_work_item::get_sub_group(),
            block_acc[sycl::ext::oneapi::this_work_item::get_sub_group()
                          .get_local_linear_id()],
            sycl::plus<float>{});

        for (int t3 = item_ct1.get_local_linear_id(); t3 <= t;
             t3 += BlockSize) {
            // don't touch the cache. Some parts will still be here from the previous loop, and
            // we want to exploit those.
            /*
            DPCT1098:38: The '*' expression is used instead of the __ldcs call.
            These two expressions do not provide the exact same functionality.
            Check the generated code for potential precision and/or performance
            issues.
            */
            float acc = *(att_bth + t3) * (*(datt_bth + t3) - local_sum);
            /*
            DPCT1098:37: The '=' expression is used instead of the __stcs call.
            These two expressions do not provide the exact same functionality.
            Check the generated code for potential precision and/or performance
            issues.
            */
            *(dpreatt_bth + t3) = scale * acc;
        }
    }
}


// ----------------------------------------------------------------------------
// kernel launchers

// attention forward pass kernel
void attention_forward(float* out, float* vaccum, float* qkvr, float* preatt, float* att,
                       const float* inp,
                       int B, int T, int C, int NH,
                       const int block_size) {
    // inp is (B, T, 3C) QKV
    // preatt, att are (B, NH, T, T)
    // output is (B, T, C)
    int HS = C / NH; // head size

    // permute and separate inp from (B, T, 3, NH, HS) to 3X (B, NH, T, HS)
    float *q, *k, *v;
    q = qkvr + 0 * B * T * C;
    k = qkvr + 1 * B * T * C;
    v = qkvr + 2 * B * T * C;
    int total_threads = B * NH * T * HS;
    int num_blocks = ceil_div(total_threads, block_size);
    /*
    DPCT1049:8: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, block_size),
                          sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) {
            permute_kernel(q, k, v, inp, B, T, NH, HS);
        });

    // batched matrix multiply with cuBLAS
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasCheck(DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm_batch(
        cublas_handle->get_queue(), oneapi::mkl::transpose::trans,
        oneapi::mkl::transpose::nontrans, T, T, HS, alpha, k, HS, T * HS, q, HS,
        T * HS, beta, preatt, T, T * T, B * NH)));

    // multiply all elements of preatt elementwise by scale
    float scale = 1.0 / sqrtf(HS);
    int softmax_block_size = 256;
    int grid_size = ceil_div(B * NH * T * 32, softmax_block_size);
    /*
    DPCT1049:9: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
        auto B_NH_ct3 = B * NH;

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                                  sycl::range<3>(1, 1, softmax_block_size),
                              sycl::range<3>(1, 1, softmax_block_size)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                softmax_forward_kernel5(att, scale, preatt, B_NH_ct3, T);
            });
    });

    // new approach: first cuBLAS another batched matmul
    // vaccum = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
    cublasCheck(DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm_batch(
        cublas_handle->get_queue(), oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::nontrans, HS, T, T, alpha, v, HS, T * HS, att,
        T, T * T, beta, vaccum, HS, T * HS, B * NH)));

    // now unpermute
    // y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
    num_blocks = ceil_div(B * T * C, block_size);
    /*
    DPCT1049:10: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, block_size),
                          sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) {
            unpermute_kernel(vaccum, out, B, T, NH, HS);
        });
}

void launch_softmax_1(float* dpreatt, float* datt, const float* att, int B, int T, int C, int NH, int block_size) {
    int num_blocks = ceil_div(T, block_size);
    /*
    DPCT1049:11: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, B * NH, num_blocks) *
                              sycl::range<3>(1, 1, block_size),
                          sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) {
            softmax_autoregressive_backward_kernel1(dpreatt, datt, att, B, T, C,
                                                    NH);
        });
}

void launch_softmax_2(float* dpreatt, float* datt, const float* att, int B, int T, int C, int NH, int block_size) {
    int num_blocks = ceil_div(T, block_size);
    /*
    DPCT1049:12: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, B * NH, num_blocks) *
                              sycl::range<3>(1, 1, block_size),
                          sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) {
            softmax_autoregressive_backward_kernel2(dpreatt, datt, att, B, T, C,
                                                    NH);
        });
}

void launch_softmax_3(float* dpreatt, float* datt, const float* att, int B, int T, int C, int NH, int block_size) {
    int num_blocks = ceil_div(32*T, block_size);
    /*
    DPCT1049:13: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, B * NH, num_blocks) *
                              sycl::range<3>(1, 1, block_size),
                          sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
            softmax_autoregressive_backward_kernel3(dpreatt, datt, att, B, T, C,
                                                    NH);
        });
}

void launch_softmax_4(float* dpreatt, float* datt, const float* att, int B, int T, int C, int NH, int block_size) {
    int num_blocks = ceil_div(32/8*T, block_size);
    /*
    DPCT1049:14: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, B * NH, num_blocks) *
                              sycl::range<3>(1, 1, block_size),
                          sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
            softmax_autoregressive_backward_kernel4(dpreatt, datt, att, B, T, C,
                                                    NH);
        });
}

void launch_softmax_5(float* dpreatt, float* datt, const float* att, int B, int T, int C, int NH, int block_size) {
    int num_blocks = ceil_div(32/8*T, block_size);
    /*
    DPCT1049:15: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, B * NH, num_blocks) *
                              sycl::range<3>(1, 1, block_size),
                          sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
            softmax_autoregressive_backward_kernel5(dpreatt, datt, att, B, T, C,
                                                    NH);
        });
}

template<class Launcher>
void dispatch_launch(Launcher&& launch, int block_size) {
    switch(block_size) {
        case 32:
            return launch(std::integral_constant<int, 32>{});
        case 64:
            return launch(std::integral_constant<int, 64>{});
        case 128:
            return launch(std::integral_constant<int, 128>{});
        case 256:
            return launch(std::integral_constant<int, 256>{});
        case 512:
            return launch(std::integral_constant<int, 512>{});
        case 1024:
            return launch(std::integral_constant<int, 1024>{});
        default:
            assert(false && "Invalid block size");
    }
}

void launch_softmax_6(float* dpreatt, float* datt, const float* att, int B, int T, int C, int NH, int block_size) {
    auto launch = [&](auto int_const) {
        /*
        DPCT1049:16: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
            sycl::local_accessor<float, 1> att_bth_s_acc_ct1(
                sycl::range<1>(int_const.value), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, B * NH, T) *
                                      sycl::range<3>(1, 1, int_const.value),
                                  sycl::range<3>(1, 1, int_const.value)),
                [=](sycl::nd_item<3> item_ct1) {
                    softmax_autoregressive_backward_kernel6<int_const.value>(
                        dpreatt, datt, att, B, T, C, NH,
                        att_bth_s_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                });
        });
    };
    dispatch_launch(launch, block_size);
}

void launch_softmax_7(float* dpreatt, float* datt, const float* att, int B, int T, int C, int NH, int block_size) {
    int hs = C / NH; // head size
    float scale = 1.0f / sqrtf(hs);
    auto launch = [&](auto int_const) {
        constexpr int block_size = int_const.value;
        /*
        DPCT1049:17: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
            sycl::local_accessor<float, 1> block_acc_acc_ct1(sycl::range<1>(32),
                                                             cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, B * NH, T) *
                                      sycl::range<3>(1, 1, block_size),
                                  sycl::range<3>(1, 1, block_size)),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        softmax_autoregressive_backward_kernel7<block_size>(
                            dpreatt, datt, att, B, T, C, scale,
                            block_acc_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                    });
        });
    };
    dispatch_launch(launch, block_size);
}

void launch_softmax_8(float* dpreatt, float* datt, const float* att, int B, int T, int C, int NH, int block_size) {
    int hs = C / NH; // head size
    float scale = 1.0f / sqrtf(hs);
    auto launch = [&](auto int_const) {
        constexpr int block_size = int_const.value;
        /*
        DPCT1049:18: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
            sycl::local_accessor<float, 1> block_acc_acc_ct1(sycl::range<1>(32),
                                                             cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, B * NH, T / 4) *
                                      sycl::range<3>(1, 1, block_size),
                                  sycl::range<3>(1, 1, block_size)),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        softmax_autoregressive_backward_kernel8<block_size>(
                            dpreatt, datt, att, B, T, C, scale,
                            block_acc_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                    });
        });
    };
    dispatch_launch(launch, block_size);
}

// the sequence of transformations in this compound op is:
// inp (B,T,3C) -> qkvr (B,T,3C) -> preatt (B,NH,T,T) -> att (B,NH,T,T) -> vaccum (B,T,C) -> out (B,T,C)
template<class SoftmaxKernel>
void attention_backward1(float* dinp, float* dqkvr, float* dpreatt, float* datt, float* dvaccum,
                        const float* dout,
                        const float* inp, const float* qkvr, const float* preatt, const float* att, const float* vaccum,
                        int B, int T, int C, int NH,
                        SoftmaxKernel softmax_autoregressive_backward,
                        const int block_size) {
    int HS = C / NH; // head size
    const float alpha = 1.0f;
    const float beta = 1.0f; // note beta = 1.0f so that we accumulate gradients (+=)
    // unpack convenience pointers into q, k, v
    const float *q, *k, *v;
    q = qkvr + 0 * B * T * C;
    k = qkvr + 1 * B * T * C;
    v = qkvr + 2 * B * T * C;
    float *dq, *dk, *dv;
    dq = dqkvr + 0 * B * T * C;
    dk = dqkvr + 1 * B * T * C;
    dv = dqkvr + 2 * B * T * C;

    // backward through the unpermute operation
    int num_blocks = ceil_div(B * T * C, block_size);
    /*
    DPCT1049:19: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, block_size),
                          sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) {
            unpermute_kernel_backward(dvaccum, dout, B, T, NH, HS);
        });
    /*
    DPCT1010:39: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    cudaCheck(0);

    // backward into datt
    cublasCheck(DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm_batch(
        cublas_handle->get_queue(), oneapi::mkl::transpose::trans,
        oneapi::mkl::transpose::nontrans, T, T, HS, alpha, v, HS, T * HS,
        dvaccum, HS, T * HS, beta, datt, T, T * T, B * NH)));

    // backward into dv
    cublasCheck(DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm_batch(
        cublas_handle->get_queue(), oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::trans, HS, T, T, alpha, dvaccum, HS, T * HS,
        att, T, T * T, beta, dv, HS, T * HS, B * NH)));

    // backward into preatt
    softmax_autoregressive_backward(dpreatt, datt, att, B, T, C, NH, block_size);
    /*
    DPCT1010:40: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    cudaCheck(0);

    // backward into q
    cublasCheck(DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm_batch(
        cublas_handle->get_queue(), oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::nontrans, HS, T, T, alpha, k, HS, T * HS,
        dpreatt, T, T * T, beta, dq, HS, T * HS, B * NH)));
    // backward into k
    cublasCheck(DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm_batch(
        cublas_handle->get_queue(), oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::trans, HS, T, T, alpha, q, HS, T * HS, dpreatt,
        T, T * T, beta, dk, HS, T * HS, B * NH)));

    // backward into inp
    num_blocks = ceil_div(B * NH * T * HS, block_size);
    /*
    DPCT1049:20: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, block_size),
                          sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) {
            permute_kernel_backward(dinp, dq, dk, dv, B, T, NH, HS);
        });
    /*
    DPCT1010:41: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    cudaCheck(0);
}

// kernel version dispatch
void attention_backward(int kernel_num,
                        float* dinp, float* dqkvr, float* dpreatt, float* datt, float* dvaccum,
                        const float* dout,
                        const float* inp, const float* qkvr, const float* preatt, const float* att, const float* vaccum,
                        int B, int T, int C, int NH,
                        const int block_size) {
    switch (kernel_num) {
        case 1:
            attention_backward1(dinp, dqkvr, dpreatt, datt, dvaccum, dout, inp, qkvr, preatt, att, vaccum, B, T, C, NH,
                                launch_softmax_1, block_size);
            break;
        case 2:
            attention_backward1(dinp, dqkvr, dpreatt, datt, dvaccum, dout, inp, qkvr, preatt, att, vaccum, B, T, C, NH,
                                launch_softmax_2, block_size);
            break;
        case 3:
            attention_backward1(dinp, dqkvr, dpreatt, datt, dvaccum, dout, inp, qkvr, preatt, att, vaccum, B, T, C, NH,
                                launch_softmax_3, block_size);
            break;
        case 4:
            attention_backward1(dinp, dqkvr, dpreatt, datt, dvaccum, dout, inp, qkvr, preatt, att, vaccum, B, T, C, NH,
                                launch_softmax_4, block_size);
            break;
        case 5:
            attention_backward1(dinp, dqkvr, dpreatt, datt, dvaccum, dout, inp, qkvr, preatt, att, vaccum, B, T, C, NH,
                                launch_softmax_5, block_size);
            break;
        case 6:
            attention_backward1(dinp, dqkvr, dpreatt, datt, dvaccum, dout, inp, qkvr, preatt, att, vaccum, B, T, C, NH,
                                launch_softmax_6, block_size);
            break;
        case 7:
            attention_backward1(dinp, dqkvr, dpreatt, datt, dvaccum, dout, inp, qkvr, preatt, att, vaccum, B, T, C, NH,
                                launch_softmax_7, block_size);
            break;
        case 8:
            attention_backward1(dinp, dqkvr, dpreatt, datt, dvaccum, dout, inp, qkvr, preatt, att, vaccum, B, T, C, NH,
                                launch_softmax_8, block_size);
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
    if (argc != 7) {
        fprintf(stderr, "usage: %s <input> <output> B T C NH\n", argv[0]);
        return 2;
    }
    const int B = atoi(argv[3]);
    const int T = atoi(argv[4]);
    const int C = atoi(argv[5]);
    const int NH = atoi(argv[6]);
    FILE* fin = fopen(argv[1], "rb");
    if (!fin) { perror(argv[1]); return 2; }
    float* inp = (float*)malloc(sizeof(float) * (size_t)(B*T*3*C));
    if (fread(inp, sizeof(float), (size_t)(B*T*3*C), fin) != (size_t)(B*T*3*C)) {
        fprintf(stderr, "input too short at inp\n"); return 2;
    }
    float* dout = (float*)malloc(sizeof(float) * (size_t)(B*T*C));
    if (fread(dout, sizeof(float), (size_t)(B*T*C), fin) != (size_t)(B*T*C)) {
        fprintf(stderr, "input too short at dout\n"); return 2;
    }
    fclose(fin);
    FILE* fout = fopen(argv[2], "wb");
    if (!fout) { perror(argv[2]); return 2; }
    float* fwd_out = (float*)malloc(sizeof(float) * (size_t)B * T * C); float* preatt = (float*)malloc(sizeof(float) * (size_t)B * NH * T * T); float* att = (float*)malloc(sizeof(float) * (size_t)B * NH * T * T); attention_forward_cpu(fwd_out, preatt, att, inp, B, T, C, NH); float* dpreatt = (float*)calloc((size_t)B * NH * T * T, sizeof(float)); float* datt = (float*)calloc((size_t)B * NH * T * T, sizeof(float));
    float* d_inp;
    cudaCheck(DPCT_CHECK_ERROR(
        d_inp = sycl::malloc_device<float>((size_t)(B * T * 3 * C),
                                           dpct::get_in_order_queue())));
    cudaCheck(DPCT_CHECK_ERROR(
        dpct::get_in_order_queue()
            .memcpy(d_inp, inp, sizeof(float) * (size_t)(B * T * 3 * C))
            .wait()));
    float* d_dout;
    cudaCheck(
        DPCT_CHECK_ERROR(d_dout = sycl::malloc_device<float>(
                             (size_t)(B * T * C), dpct::get_in_order_queue())));
    cudaCheck(DPCT_CHECK_ERROR(
        dpct::get_in_order_queue()
            .memcpy(d_dout, dout, sizeof(float) * (size_t)(B * T * C))
            .wait()));
    float* dinp = (float*)malloc(sizeof(float) * (size_t)(B*T*3*C));
    float* d_dinp;
    cudaCheck(DPCT_CHECK_ERROR(
        d_dinp = sycl::malloc_device<float>((size_t)(B * T * 3 * C),
                                            dpct::get_in_order_queue())));
    float *d_qkvr, *d_preatt, *d_att, *d_vaccum, *d_fwd_out, *d_dqkvr,
        *d_dpreatt, *d_datt, *d_dvaccum;
        cudaCheck(DPCT_CHECK_ERROR(d_qkvr = (float *)sycl::malloc_device(
                                       sizeof(float) * (size_t)B * T * 3 * C,
                                       dpct::get_in_order_queue())));
        cudaCheck(DPCT_CHECK_ERROR(d_preatt = (float *)sycl::malloc_device(
                                       sizeof(float) * (size_t)B * NH * T * T,
                                       dpct::get_in_order_queue())));
        cudaCheck(DPCT_CHECK_ERROR(d_att = (float *)sycl::malloc_device(
                                       sizeof(float) * (size_t)B * NH * T * T,
                                       dpct::get_in_order_queue())));
        cudaCheck(DPCT_CHECK_ERROR(d_vaccum = (float *)sycl::malloc_device(
                                       sizeof(float) * (size_t)B * T * C,
                                       dpct::get_in_order_queue())));
        cudaCheck(DPCT_CHECK_ERROR(d_fwd_out = (float *)sycl::malloc_device(
                                       sizeof(float) * (size_t)B * T * C,
                                       dpct::get_in_order_queue())));
        cudaCheck(DPCT_CHECK_ERROR(d_dqkvr = (float *)sycl::malloc_device(
                                       sizeof(float) * (size_t)B * T * 3 * C,
                                       dpct::get_in_order_queue())));
        cudaCheck(DPCT_CHECK_ERROR(d_dpreatt = (float *)sycl::malloc_device(
                                       sizeof(float) * (size_t)B * NH * T * T,
                                       dpct::get_in_order_queue())));
        cudaCheck(DPCT_CHECK_ERROR(d_datt = (float *)sycl::malloc_device(
                                       sizeof(float) * (size_t)B * NH * T * T,
                                       dpct::get_in_order_queue())));
        cudaCheck(DPCT_CHECK_ERROR(d_dvaccum = (float *)sycl::malloc_device(
                                       sizeof(float) * (size_t)B * T * C,
                                       dpct::get_in_order_queue())));
        cudaCheck(DPCT_CHECK_ERROR(
            dpct::get_in_order_queue()
                .memset(d_att, 0, sizeof(float) * (size_t)B * NH * T * T)
                .wait()));
        cublasCheck(DPCT_CHECK_ERROR(
            cublas_handle->set_math_mode(dpct::blas::math_mode::mm_default)));
        attention_forward(d_fwd_out, d_vaccum, d_qkvr, d_preatt, d_att, d_inp,
                          B, T, C, NH, 256);
        cudaCheck(DPCT_CHECK_ERROR(
            dpct::get_current_device().queues_wait_and_throw()));
    const int variants[] = {1, 2, 3, 4, 5, 6, 7, 8};
    for (int vi = 0; vi < 8; vi++) {
        const int k = variants[vi];
        cudaCheck(DPCT_CHECK_ERROR(
            dpct::get_in_order_queue()
                .memset(d_dinp, 0, sizeof(float) * (size_t)(B * T * 3 * C))
                .wait()));
        cudaCheck(DPCT_CHECK_ERROR(
            dpct::get_in_order_queue()
                .memset(d_dqkvr, 0, sizeof(float) * (size_t)B * T * 3 * C)
                .wait()));
            cudaCheck(DPCT_CHECK_ERROR(
                dpct::get_in_order_queue()
                    .memset(d_dpreatt, 0,
                            sizeof(float) * (size_t)B * NH * T * T)
                    .wait()));
            cudaCheck(DPCT_CHECK_ERROR(
                dpct::get_in_order_queue()
                    .memset(d_datt, 0, sizeof(float) * (size_t)B * NH * T * T)
                    .wait()));
            cudaCheck(DPCT_CHECK_ERROR(
                dpct::get_in_order_queue()
                    .memset(d_dvaccum, 0, sizeof(float) * (size_t)B * T * C)
                    .wait()));
            attention_backward(k, d_dinp, d_dqkvr, d_dpreatt, d_datt, d_dvaccum,
                               d_dout, d_inp, d_qkvr, d_preatt, d_att, d_vaccum,
                               B, T, C, NH, 256);
        /*
        DPCT1010:42: SYCL uses exceptions to report errors and does not use the
        error codes. The cudaGetLastError function call was replaced with 0. You
        need to rewrite this code.
        */
        cudaCheck(0);
        cudaCheck(DPCT_CHECK_ERROR(
            dpct::get_current_device().queues_wait_and_throw()));
        cudaCheck(DPCT_CHECK_ERROR(
            dpct::get_in_order_queue()
                .memcpy(dinp, d_dinp, sizeof(float) * (size_t)(B * T * 3 * C))
                .wait()));
        fwrite(dinp, sizeof(float), (size_t)(B*T*3*C), fout);
    }
    fclose(fout);
    return 0;
}
