// GPT-2 fp32 training (llm.c train_gpt2_fp32.cu) ported to idiomatic SYCL 2020.
// All computation runs on the SYCL device: 1-D kernels, work-group
// reductions, oneMath column_major gemm / gemm_batch for matmuls,
// precise sycl::exp / sycl::log / sycl::tanh math.

#include <sycl/sycl.hpp>
#include <oneapi/math.hpp>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <float.h>
#include <string.h>
#include <unistd.h>

// llm.c host utilities (plain C, on the include path)
#include "llmc/utils.h"
#include "llmc/tokenizer.h"
#include "llmc/dataloader.h"

#ifndef CEIL_DIV
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
#endif

// ----------------------------------------------------------------------------
// device queue shared by every launcher

static sycl::queue *g_q = nullptr;
static inline sycl::queue &qq() { return *g_q; }

static inline void atomic_add_float(float *addr, float v) {
    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space>
        ar(*addr);
    ar.fetch_add(v);
}

// launch a 1-D elementwise kernel with n work-items
template <typename F> static inline void launch_1d(size_t n, F f) {
    if (n == 0) return;
    qq().parallel_for(sycl::range<1>(n), f).wait_and_throw();
}

// launch a 1-D nd_range kernel: rows work-groups of ls work-items
template <typename F> static inline void launch_groups(size_t rows, size_t ls, F f) {
    if (rows == 0) return;
    qq().parallel_for(sycl::nd_range<1>(rows * ls, ls), f).wait_and_throw();
}

// ----------------------------------------------------------------------------
// encoder

static void encoder_forward(float *out, const int *inp, const float *wte,
                            const float *wpe, int B, int T, int C) {
    size_t N = (size_t)B * T * (size_t)C;
    launch_1d(N, [=](sycl::id<1> id) {
        size_t i = id[0];
        size_t c = i % (size_t)C;
        size_t bt = i / (size_t)C;
        size_t t = bt % (size_t)T;
        size_t b = bt / (size_t)T;
        int ix = inp[b * T + t];
        out[i] = wte[(size_t)ix * C + c] + wpe[t * (size_t)C + c];
    });
}

static void encoder_backward(float *dwte, float *dwpe, const float *dout,
                             const int *inp, int B, int T, int C) {
    size_t N = (size_t)B * T * (size_t)C;
    launch_1d(N, [=](sycl::id<1> id) {
        size_t i = id[0];
        size_t c = i % (size_t)C;
        size_t bt = i / (size_t)C;
        size_t t = bt % (size_t)T;
        size_t b = bt / (size_t)T;
        int ix = inp[b * T + t];
        float g = dout[i];
        atomic_add_float(dwte + (size_t)ix * C + c, g);
        atomic_add_float(dwpe + t * (size_t)C + c, g);
    });
}

// ----------------------------------------------------------------------------
// layernorm forward: one work-group per row

static void layernorm_forward(float *out, float *mean, float *rstd,
                              const float *inp, const float *weight,
                              const float *bias, int N, int C) {
    constexpr size_t LS = 256;
    launch_groups((size_t)N, LS, [=](sycl::nd_item<1> it) {
        size_t row = it.get_group(0);
        size_t lid = it.get_local_id(0);
        size_t ls = it.get_local_range(0);
        sycl::group<1> g = it.get_group();
        const float *x = inp + row * (size_t)C;
        float sum = 0.0f;
        for (int i = (int)lid; i < C; i += (int)ls) sum += x[i];
        sum = sycl::reduce_over_group(g, sum, sycl::plus<float>());
        float m = sum / (float)C;
        if (lid == 0 && mean != nullptr) mean[row] = m;
        float sq = 0.0f;
        for (int i = (int)lid; i < C; i += (int)ls) {
            float d = x[i] - m;
            sq += d * d;
        }
        sq = sycl::reduce_over_group(g, sq, sycl::plus<float>());
        float s = 1.0f / sycl::sqrt(sq / (float)C + 1e-5f);
        if (lid == 0 && rstd != nullptr) rstd[row] = s;
        float *o = out + row * (size_t)C;
        for (int c = (int)lid; c < C; c += (int)ls) {
            float nrm = s * (x[c] - m);
            o[c] = nrm * weight[c] + bias[c];
        }
    });
}

// ----------------------------------------------------------------------------
// permute helpers (attention head reshapes)

static void permute_qkv(float *q, float *k, float *v, const float *inp,
                        int B, int T, int NH, int HS) {
    size_t N = (size_t)B * NH * T * (size_t)HS;
    launch_1d(N, [=](sycl::id<1> id) {
        size_t i = id[0];
        size_t d = i % (size_t)HS;
        size_t r = i / (size_t)HS;
        size_t n = r % (size_t)T;
        r /= (size_t)T;
        size_t nh = r % (size_t)NH;
        size_t b = r / (size_t)NH;
        size_t base = (((b * (size_t)T + n) * 3 + 0) * (size_t)NH + nh) * (size_t)HS + d;
        size_t stride = (size_t)NH * HS;
        q[i] = inp[base];
        k[i] = inp[base + stride];
        v[i] = inp[base + 2 * stride];
    });
}

static void permute_qkv_backward(float *dinp, const float *dq, const float *dk,
                                 const float *dv, int B, int T, int NH, int HS) {
    size_t N = (size_t)B * NH * T * (size_t)HS;
    launch_1d(N, [=](sycl::id<1> id) {
        size_t i = id[0];
        size_t d = i % (size_t)HS;
        size_t r = i / (size_t)HS;
        size_t n = r % (size_t)T;
        r /= (size_t)T;
        size_t nh = r % (size_t)NH;
        size_t b = r / (size_t)NH;
        size_t base = (((b * (size_t)T + n) * 3 + 0) * (size_t)NH + nh) * (size_t)HS + d;
        size_t stride = (size_t)NH * HS;
        dinp[base] = dq[i];
        dinp[base + stride] = dk[i];
        dinp[base + 2 * stride] = dv[i];
    });
}

static void unpermute_out(float *out, const float *inp, int B, int T, int NH, int HS) {
    size_t N = (size_t)B * NH * T * (size_t)HS;
    launch_1d(N, [=](sycl::id<1> id) {
        size_t i = id[0];
        size_t d = i % (size_t)HS;
        size_t r = i / (size_t)HS;
        size_t n = r % (size_t)T;
        r /= (size_t)T;
        size_t nh = r % (size_t)NH;
        size_t b = r / (size_t)NH;
        size_t o = ((b * (size_t)T + n) * (size_t)NH + nh) * (size_t)HS + d;
        out[o] = inp[i];
    });
}

static void unpermute_backward(float *dinp, const float *dout,
                               int B, int T, int NH, int HS) {
    size_t N = (size_t)B * NH * T * (size_t)HS;
    launch_1d(N, [=](sycl::id<1> id) {
        size_t i = id[0];
        size_t d = i % (size_t)HS;
        size_t r = i / (size_t)HS;
        size_t n = r % (size_t)T;
        r /= (size_t)T;
        size_t nh = r % (size_t)NH;
        size_t b = r / (size_t)NH;
        size_t o = ((b * (size_t)T + n) * (size_t)NH + nh) * (size_t)HS + d;
        dinp[i] = dout[o];
    });
}

// ----------------------------------------------------------------------------
// causal softmax forward (fused scale): one work-group per row

static void softmax_forward(float *out, float scale, const float *inp, int rows, int T) {
    constexpr size_t LS = 128;
    launch_groups((size_t)rows, LS, [=](sycl::nd_item<1> it) {
        size_t row = it.get_group(0);
        size_t lid = it.get_local_id(0);
        size_t ls = it.get_local_range(0);
        sycl::group<1> g = it.get_group();
        int own = (int)(row % (size_t)T);
        const float *x = inp + row * (size_t)T;
        float tmax = -FLT_MAX;
        for (int i = (int)lid; i <= own; i += (int)ls) tmax = sycl::fmax(tmax, x[i]);
        float bmax = sycl::reduce_over_group(g, tmax, sycl::maximum<float>());
        float tsum = 0.0f;
        for (int i = (int)lid; i <= own; i += (int)ls)
            tsum += sycl::exp(scale * (x[i] - bmax));
        float bsum = sycl::reduce_over_group(g, tsum, sycl::plus<float>());
        float norm = 1.0f / bsum;
        for (int i = (int)lid; i <= own; i += (int)ls)
            out[row * (size_t)T + i] = sycl::exp(scale * (x[i] - bmax)) * norm;
    });
}

// ----------------------------------------------------------------------------
// elementwise helpers

static void residual_forward(float *out, const float *a, const float *b, size_t N) {
    launch_1d(N, [=](sycl::id<1> id) { out[id[0]] = a[id[0]] + b[id[0]]; });
}

#define GELU_SCALING_FACTOR 0.7978845608028654f // sqrt(2/pi)

static void gelu_forward(float *out, const float *inp, size_t N) {
    launch_1d(N, [=](sycl::id<1> id) {
        size_t i = id[0];
        float xi = inp[i];
        float cube = 0.044715f * xi * xi * xi;
        out[i] = 0.5f * xi * (1.0f + sycl::tanh(GELU_SCALING_FACTOR * (xi + cube)));
    });
}

static void gelu_backward(float *dinp, const float *inp, const float *dout, size_t N) {
    launch_1d(N, [=](sycl::id<1> id) {
        size_t i = id[0];
        float x = inp[i];
        float cube = 0.044715f * x * x * x;
        float ta = GELU_SCALING_FACTOR * (x + cube);
        float to = sycl::tanh(ta);
        float co = sycl::cosh(ta);
        float se = 1.0f / (co * co);
        float lg = 0.5f * (1.0f + to) +
                   x * 0.5f * se * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
        dinp[i] = lg * dout[i];
    });
}

static void bias_add(float *out, const float *bias, int M, int OC) {
    launch_1d((size_t)M * (size_t)OC, [=](sycl::id<1> id) {
        out[id[0]] += bias[id[0] % (size_t)OC];
    });
}

// dbias[j] += sum_rows dout[row, j]
static void bias_row_sum(float *dbias, const float *dout, int BT, int OC) {
    launch_1d((size_t)OC, [=](sycl::id<1> id) {
        size_t j = id[0];
        float s = 0.0f;
        for (int r = 0; r < BT; r++) s += dout[(size_t)r * OC + j];
        atomic_add_float(dbias + j, s);
    });
}

// ----------------------------------------------------------------------------
// layernorm backward: one work-group per row, atomics into params

static void layernorm_backward(float *dinp, float *dweight, float *dbias,
                               const float *dout, const float *inp,
                               const float *weight, const float *mean,
                               const float *rstd, int B, int T, int C) {
    constexpr size_t LS = 256;
    launch_groups((size_t)B * (size_t)T, LS, [=](sycl::nd_item<1> it) {
        size_t row = it.get_group(0);
        size_t lid = it.get_local_id(0);
        size_t ls = it.get_local_range(0);
        sycl::group<1> g = it.get_group();
        const float *db = dout + row * (size_t)C;
        const float *xb = inp + row * (size_t)C;
        float *gb = dinp + row * (size_t)C;
        float mb = mean[row];
        float rb = rstd[row];
        float p1 = 0.0f, p2 = 0.0f;
        for (int i = (int)lid; i < C; i += (int)ls) {
            float nrm = (xb[i] - mb) * rb;
            float dn = weight[i] * db[i];
            p1 += dn;
            p2 += dn * nrm;
        }
        float dmean = sycl::reduce_over_group(g, p1, sycl::plus<float>()) / (float)C;
        float dnrm = sycl::reduce_over_group(g, p2, sycl::plus<float>()) / (float)C;
        for (int i = (int)lid; i < C; i += (int)ls) {
            float nrm = (xb[i] - mb) * rb;
            float dn = weight[i] * db[i];
            atomic_add_float(dbias + i, db[i]);
            atomic_add_float(dweight + i, nrm * db[i]);
            atomic_add_float(gb + i, (dn - dmean - nrm * dnrm) * rb);
        }
    });
}

// ----------------------------------------------------------------------------
// causal softmax backward: one work-group per (head-batch, t) row

static void softmax_backward(float *dpreatt, const float *datt, const float *att,
                             int rows, int T, float scale) {
    constexpr size_t LS = 128;
    launch_groups((size_t)rows * (size_t)T, LS, [=](sycl::nd_item<1> it) {
        size_t row = it.get_group(0);
        size_t lid = it.get_local_id(0);
        size_t ls = it.get_local_range(0);
        sycl::group<1> g = it.get_group();
        int t = (int)(row % (size_t)T);
        size_t base = row * (size_t)T;
        float psum = 0.0f;
        for (int i = (int)lid; i <= t; i += (int)ls) psum += att[base + i] * datt[base + i];
        float lsum = sycl::reduce_over_group(g, psum, sycl::plus<float>());
        for (int i = (int)lid; i <= t; i += (int)ls)
            dpreatt[base + i] = scale * att[base + i] * (datt[base + i] - lsum);
    });
}

// ----------------------------------------------------------------------------
// fused classifier: softmax + cross-entropy + logit grads, one group per row

static void fused_classifier(float *logits, float *losses, const int *targets,
                             int B, int T, int V, int P) {
    constexpr size_t LS = 256;
    launch_groups((size_t)B * (size_t)T, LS, [=](sycl::nd_item<1> it) {
        size_t row = it.get_group(0);
        size_t lid = it.get_local_id(0);
        size_t ls = it.get_local_range(0);
        sycl::group<1> g = it.get_group();
        int ix = targets[row];
        const float *lv = logits + row * (size_t)P;
        float tmax = -FLT_MAX;
        for (int i = (int)lid; i < V; i += (int)ls) tmax = sycl::fmax(tmax, lv[i]);
        float bmax = sycl::reduce_over_group(g, tmax, sycl::maximum<float>());
        float tsum = 0.0f;
        for (int i = (int)lid; i < V; i += (int)ls) tsum += sycl::exp(lv[i] - bmax);
        float bsum = sycl::reduce_over_group(g, tsum, sycl::plus<float>());
        float sc = 1.0f / bsum;
        if (lid == 0) {
            float prob = sycl::exp(lv[ix] - bmax) * sc;
            losses[row] = -sycl::log(prob);
        }
        sycl::group_barrier(g);
        float dloss = 1.0f / ((float)B * (float)T);
        float *gw = logits + row * (size_t)P;
        for (int i = (int)lid; i < V; i += (int)ls) {
            float prob = sycl::exp(gw[i] - bmax) * sc;
            float ind = (i == ix) ? 1.0f : 0.0f;
            gw[i] = (prob - ind) * dloss;
        }
    });
}

// ----------------------------------------------------------------------------
// AdamW update

static void adamw_update(float *params, float *grads, float *m, float *v,
                         size_t n, float lr, float b1, float b2,
                         float b1c, float b2c, float eps, float wd) {
    launch_1d(n, [=](sycl::id<1> id) {
        size_t i = id[0];
        float g = grads[i];
        float mi = m[i];
        float vi = v[i];
        mi = sycl::fma(b1, mi - g, g); // lerp(g, mi, b1)
        m[i] = mi;
        vi = sycl::fma(b2, vi - g * g, g * g); // lerp(g*g, vi, b2)
        v[i] = vi;
        float mh = mi / b1c;
        float vh = vi / b2c;
        params[i] -= lr * (mh / (sycl::sqrt(vh) + eps) + wd * params[i]);
    });
}

// ----------------------------------------------------------------------------
// matmuls via oneMath column_major.
// Row-major C = A * B is expressed through the transpose identity:
// a row-major [R,C] buffer is the transpose of a column-major [C,R] view.

namespace bm = oneapi::math::blas::column_major;

static void matmul_forward(float *out, const float *inp, const float *weight,
                           const float *bias, int B, int T, int C, int OC) {
    // row-major: out[M,OC] = inp[M,C] * weight[OC,C]^T (+ bias)
    // transposed: out^T = weight * inp^T
    int64_t M = (int64_t)B * T;
    bm::gemm(qq(), oneapi::math::transpose::trans, oneapi::math::transpose::nontrans,
             (int64_t)OC, M, (int64_t)C, 1.0f, weight, (int64_t)C,
             inp, (int64_t)C, 0.0f, out, (int64_t)OC)
        .wait_and_throw();
    if (bias != nullptr) bias_add(out, bias, (int)M, OC);
}

static void matmul_backward(float *dinp, float *dweight, float *dbias,
                            float *dout, float *inp, float *weight,
                            int B, int T, int C, int OC) {
    // row-major: dinp[M,C] = dout[M,OC] * weight[OC,C]
    // transposed: dinp^T = weight^T * dout^T
    int64_t M = (int64_t)B * T;
    bm::gemm(qq(), oneapi::math::transpose::nontrans, oneapi::math::transpose::nontrans,
             (int64_t)C, M, (int64_t)OC, 1.0f, weight, (int64_t)C,
             dout, (int64_t)OC, 0.0f, dinp, (int64_t)C)
        .wait_and_throw();
    // row-major: dweight[OC,C] += dout[M,OC]^T * inp[M,C]
    // transposed: dweight^T = inp^T * dout
    bm::gemm(qq(), oneapi::math::transpose::nontrans, oneapi::math::transpose::trans,
             (int64_t)C, (int64_t)OC, M, 1.0f, inp, (int64_t)C,
             dout, (int64_t)OC, 1.0f, dweight, (int64_t)C)
        .wait_and_throw();
    if (dbias != nullptr) bias_row_sum(dbias, dout, (int)M, OC);
}

// ----------------------------------------------------------------------------
// attention

static void attention_forward(float *out, float *qkvr, float *att, float *inp,
                              int B, int T, int C, int NH) {
    int HS = C / NH;
    float *q = qkvr;
    float *k = qkvr + (size_t)B * T * C;
    float *v = qkvr + 2 * (size_t)B * T * C;
    permute_qkv(q, k, v, inp, B, T, NH, HS);
    float alpha = 1.0f, beta = 0.0f;
    float *preatt = inp;
    int64_t batch = (int64_t)B * NH;
    int64_t st_qkv = (int64_t)T * HS, st_att = (int64_t)T * T;
    bm::gemm_batch(qq(), oneapi::math::transpose::trans, oneapi::math::transpose::nontrans,
                   (int64_t)T, (int64_t)T, (int64_t)HS, alpha,
                   k, (int64_t)HS, st_qkv, q, (int64_t)HS, st_qkv,
                   beta, preatt, (int64_t)T, st_att, batch)
        .wait_and_throw();
    float scale = 1.0f / sqrtf((float)HS);
    softmax_forward(att, scale, preatt, (int)batch * T, T);
    float *vaccum = inp;
    bm::gemm_batch(qq(), oneapi::math::transpose::nontrans, oneapi::math::transpose::nontrans,
                   (int64_t)HS, (int64_t)T, (int64_t)T, alpha,
                   v, (int64_t)HS, st_qkv, att, (int64_t)T, st_att,
                   beta, vaccum, (int64_t)HS, st_qkv, batch)
        .wait_and_throw();
    unpermute_out(out, vaccum, B, T, NH, HS);
}

static void attention_backward(float *dinp, float *dqkvr, float *dpreatt, float *datt,
                               float *scratch, const float *dout, const float *qkvr,
                               const float *att, int B, int T, int C, int NH) {
    int HS = C / NH;
    const float *q = qkvr;
    const float *k = qkvr + (size_t)B * T * C;
    const float *v = qkvr + 2 * (size_t)B * T * C;
    float *dq = dqkvr;
    float *dk = dqkvr + (size_t)B * T * C;
    float *dv = dqkvr + 2 * (size_t)B * T * C;
    unpermute_backward(scratch, dout, B, T, NH, HS);
    float one = 1.0f, zero = 0.0f;
    int64_t batch = (int64_t)B * NH;
    int64_t st_qkv = (int64_t)T * HS, st_att = (int64_t)T * T;
    bm::gemm_batch(qq(), oneapi::math::transpose::trans, oneapi::math::transpose::nontrans,
                   (int64_t)T, (int64_t)T, (int64_t)HS, one,
                   v, (int64_t)HS, st_qkv, scratch, (int64_t)HS, st_qkv,
                   zero, datt, (int64_t)T, st_att, batch)
        .wait_and_throw();
    bm::gemm_batch(qq(), oneapi::math::transpose::nontrans, oneapi::math::transpose::trans,
                   (int64_t)HS, (int64_t)T, (int64_t)T, one,
                   scratch, (int64_t)HS, st_qkv, att, (int64_t)T, st_att,
                   zero, dv, (int64_t)HS, st_qkv, batch)
        .wait_and_throw();
    float scale = 1.0f / sqrtf((float)HS);
    softmax_backward(dpreatt, datt, att, (int)batch, T, scale);
    bm::gemm_batch(qq(), oneapi::math::transpose::nontrans, oneapi::math::transpose::nontrans,
                   (int64_t)HS, (int64_t)T, (int64_t)T, one,
                   k, (int64_t)HS, st_qkv, dpreatt, (int64_t)T, st_att,
                   zero, dq, (int64_t)HS, st_qkv, batch)
        .wait_and_throw();
    bm::gemm_batch(qq(), oneapi::math::transpose::nontrans, oneapi::math::transpose::trans,
                   (int64_t)HS, (int64_t)T, (int64_t)T, one,
                   q, (int64_t)HS, st_qkv, dpreatt, (int64_t)T, st_att,
                   zero, dk, (int64_t)HS, st_qkv, batch)
        .wait_and_throw();
    permute_qkv_backward(dinp, dq, dk, dv, B, T, NH, HS);
}

// ----------------------------------------------------------------------------
// GPT-2 model

typedef struct {
    int max_seq_len;
    int vocab_size;
    int padded_vocab_size;
    int num_layers;
    int num_heads;
    int channels;
} GPT2Config;

#define NUM_PARAMETER_TENSORS 16
typedef struct {
    float *wte, *wpe, *ln1w, *ln1b, *qkvw, *qkvb, *attprojw, *attprojb;
    float *ln2w, *ln2b, *fcw, *fcb, *fcprojw, *fcprojb, *lnfw, *lnfb;
} ParameterTensors;

static void fill_in_parameter_sizes(size_t *s, GPT2Config c) {
    size_t Vp = (size_t)c.padded_vocab_size, C = (size_t)c.channels;
    size_t maxT = (size_t)c.max_seq_len, L = (size_t)c.num_layers;
    s[0] = Vp * C; s[1] = maxT * C; s[2] = L * C; s[3] = L * C;
    s[4] = L * 3 * C * C; s[5] = L * 3 * C; s[6] = L * C * C; s[7] = L * C;
    s[8] = L * C; s[9] = L * C; s[10] = L * 4 * C * C; s[11] = L * 4 * C;
    s[12] = L * C * 4 * C; s[13] = L * C; s[14] = C; s[15] = C;
}

static float *alloc_params(ParameterTensors *p, size_t *sizes, int on_device) {
    size_t n = 0;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) n += sizes[i];
    float *mem = on_device ? sycl::malloc_device<float>(n, qq())
                           : (float *)malloc(n * sizeof(float));
    if (mem == NULL) { fprintf(stderr, "param alloc failed\n"); exit(EXIT_FAILURE); }
    float **ptrs[] = { &p->wte, &p->wpe, &p->ln1w, &p->ln1b, &p->qkvw, &p->qkvb,
                       &p->attprojw, &p->attprojb, &p->ln2w, &p->ln2b, &p->fcw, &p->fcb,
                       &p->fcprojw, &p->fcprojb, &p->lnfw, &p->lnfb };
    float *it = mem;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) { *(ptrs[i]) = it; it += sizes[i]; }
    return mem;
}

#define NUM_ACTIVATION_TENSORS 21
typedef struct {
    float *encoded, *ln1, *ln1_mean, *ln1_rstd, *atty, *att, *attproj, *residual2;
    float *ln2, *ln2_mean, *ln2_rstd, *fch, *fch_gelu, *fcproj, *residual3;
    float *lnf, *lnf_mean, *lnf_rstd, *losses, *qkvr, *output;
} ActivationTensors;

static void fill_in_activation_sizes(size_t *s, int B, int T, GPT2Config c) {
    size_t Vp = (size_t)c.padded_vocab_size, L = (size_t)c.num_layers;
    size_t NH = (size_t)c.num_heads, C = (size_t)c.channels;
    size_t BT = (size_t)B * T;
    s[0] = BT * C; s[1] = L * BT * C; s[2] = L * BT; s[3] = L * BT;
    s[4] = L * BT * C; s[5] = L * (size_t)B * NH * T * T; s[6] = L * BT * C;
    s[7] = L * BT * C; s[8] = L * BT * C; s[9] = L * BT; s[10] = L * BT;
    s[11] = L * BT * 4 * C; s[12] = L * BT * 4 * C; s[13] = L * BT * C;
    s[14] = L * BT * C; s[15] = BT * C; s[16] = BT; s[17] = BT; s[18] = BT;
    s[19] = L * BT * 3 * C;
    size_t tail = 3 * C;
    size_t nha = NH * (size_t)T;
    if (nha > tail) tail = nha;
    if (Vp > tail) tail = Vp;
    s[20] = BT * tail;
}

#define NUM_BACKWARD_TENSORS 3
typedef struct {
    float *bt4c, *preatt, *residual3;
} GradActTensors;

static void fill_in_grad_act_sizes(size_t *s, int B, int T, GPT2Config c) {
    size_t NH = (size_t)c.num_heads, C = (size_t)c.channels;
    s[0] = (size_t)B * T * 4 * C;
    s[1] = (size_t)B * NH * T * T;
    s[2] = (size_t)B * T * C;
}

static float *alloc_point(float **targets[], const size_t *sizes, int n) {
    size_t total = 0;
    for (int i = 0; i < n; i++) total += sizes[i];
    float *mem = sycl::malloc_device<float>(total, qq());
    if (mem == NULL) { fprintf(stderr, "alloc failed\n"); exit(EXIT_FAILURE); }
    float *it = mem;
    for (int i = 0; i < n; i++) { *(targets[i]) = it; it += sizes[i]; }
    return mem;
}

static float *alloc_activations(ActivationTensors *a, const size_t *s) {
    float **ptrs[] = { &a->encoded, &a->ln1, &a->ln1_mean, &a->ln1_rstd, &a->atty,
                       &a->att, &a->attproj, &a->residual2, &a->ln2, &a->ln2_mean,
                       &a->ln2_rstd, &a->fch, &a->fch_gelu, &a->fcproj, &a->residual3,
                       &a->lnf, &a->lnf_mean, &a->lnf_rstd, &a->losses, &a->qkvr,
                       &a->output };
    return alloc_point(ptrs, s, NUM_ACTIVATION_TENSORS);
}

static float *alloc_backward(GradActTensors *a, const size_t *s) {
    float **ptrs[] = { &a->bt4c, &a->preatt, &a->residual3 };
    return alloc_point(ptrs, s, NUM_BACKWARD_TENSORS);
}

typedef struct {
    GPT2Config config;
    ParameterTensors params;
    size_t param_sizes[NUM_PARAMETER_TENSORS];
    float *params_memory;
    size_t num_parameters;
    ParameterTensors grads;
    float *grads_memory;
    float *m_memory;
    float *v_memory;
    ActivationTensors acts;
    size_t act_sizes[NUM_ACTIVATION_TENSORS];
    float *acts_memory;
    size_t num_activations;
    GradActTensors grads_acts;
    size_t num_grad_acts;
    float *grads_acts_memory;
    int batch_size;
    int seq_len;
    int *inputs;
    int *targets;
    float mean_loss;
    float *cpu_losses;
} GPT2;

static void gpt2_build_from_checkpoint(GPT2 *model, const char *path) {
    FILE *f = fopenCheck(path, "rb");
    int header[256];
    freadCheck(header, sizeof(int), 256, f);
    if (header[0] != 20240326 || header[1] != 3) {
        fprintf(stderr, "Bad model file\n");
        exit(EXIT_FAILURE);
    }
    model->config.max_seq_len = header[2];
    model->config.vocab_size = header[3];
    model->config.num_layers = header[4];
    model->config.num_heads = header[5];
    model->config.channels = header[6];
    model->config.padded_vocab_size = header[7];
    fill_in_parameter_sizes(model->param_sizes, model->config);
    size_t n = 0;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) n += model->param_sizes[i];
    model->num_parameters = n;
    model->params_memory = alloc_params(&model->params, model->param_sizes, 1);
    float *cpu = (float *)mallocCheck(n * sizeof(float));
    freadCheck(cpu, sizeof(float), n, f);
    fcloseCheck(f);
    qq().memcpy(model->params_memory, cpu, n * sizeof(float)).wait_and_throw();
    free(cpu);
    model->acts_memory = NULL;
    model->grads_memory = NULL;
    model->m_memory = NULL;
    model->v_memory = NULL;
    model->grads_acts_memory = NULL;
    model->inputs = NULL;
    model->targets = NULL;
    model->cpu_losses = NULL;
    model->batch_size = 0;
    model->seq_len = 0;
    model->mean_loss = -1.0f;
}

static void gpt2_forward(GPT2 *model, int *inputs, int *targets, int B, int T) {
    if (model->params_memory == NULL) { fprintf(stderr, "model not init\n"); exit(EXIT_FAILURE); }
    int V = model->config.vocab_size;
    int Vp = model->config.padded_vocab_size;
    int L = model->config.num_layers;
    int NH = model->config.num_heads;
    int C = model->config.channels;
    for (int i = 0; i < B * T; i++) {
        assert(0 <= inputs[i] && inputs[i] < V);
        if (targets != NULL) assert(0 <= targets[i] && targets[i] < V);
    }
    if (model->acts_memory == NULL) {
        model->batch_size = B;
        model->seq_len = T;
        fill_in_activation_sizes(model->act_sizes, B, T, model->config);
        size_t n = 0;
        for (int i = 0; i < NUM_ACTIVATION_TENSORS; i++) n += model->act_sizes[i];
        model->num_activations = n;
        model->acts_memory = alloc_activations(&model->acts, model->act_sizes);
        printf("allocated %zu MiB for activations\n", (n * sizeof(float)) >> 20);
        model->inputs = sycl::malloc_device<int>((size_t)B * T, qq());
        model->targets = sycl::malloc_device<int>((size_t)B * T, qq());
        model->cpu_losses = (float *)malloc((size_t)B * T * sizeof(float));
        if (!model->inputs || !model->targets || !model->cpu_losses) {
            fprintf(stderr, "act alloc failed\n");
            exit(EXIT_FAILURE);
        }
    } else if (B != model->batch_size || T != model->seq_len) {
        fprintf(stderr, "B/T mismatch\n");
        exit(EXIT_FAILURE);
    }
    qq().memcpy(model->inputs, inputs, (size_t)B * T * sizeof(int)).wait_and_throw();
    if (targets != NULL)
        qq().memcpy(model->targets, targets, (size_t)B * T * sizeof(int)).wait_and_throw();

    ParameterTensors params = model->params;
    ActivationTensors acts = model->acts;
    encoder_forward(acts.encoded, model->inputs, params.wte, params.wpe, B, T, C);
    for (int l = 0; l < L; l++) {
        float *residual = (l == 0) ? acts.encoded : acts.residual3 + (size_t)(l - 1) * B * T * C;
        float *l_ln1w = params.ln1w + (size_t)l * C;
        float *l_ln1b = params.ln1b + (size_t)l * C;
        float *l_qkvw = params.qkvw + (size_t)l * 3 * C * C;
        float *l_qkvb = params.qkvb + (size_t)l * 3 * C;
        float *l_attprojw = params.attprojw + (size_t)l * C * C;
        float *l_attprojb = params.attprojb + (size_t)l * C;
        float *l_ln2w = params.ln2w + (size_t)l * C;
        float *l_ln2b = params.ln2b + (size_t)l * C;
        float *l_fcw = params.fcw + (size_t)l * 4 * C * C;
        float *l_fcb = params.fcb + (size_t)l * 4 * C;
        float *l_fcprojw = params.fcprojw + (size_t)l * C * 4 * C;
        float *l_fcprojb = params.fcprojb + (size_t)l * C;
        float *l_ln1 = acts.ln1 + (size_t)l * B * T * C;
        float *l_ln1_mean = acts.ln1_mean + (size_t)l * B * T;
        float *l_ln1_rstd = acts.ln1_rstd + (size_t)l * B * T;
        float *l_qkvr = acts.qkvr + (size_t)l * B * T * 3 * C;
        float *l_atty = acts.atty + (size_t)l * B * T * C;
        float *l_att = acts.att + (size_t)l * B * NH * T * T;
        float *l_attproj = acts.attproj + (size_t)l * B * T * C;
        float *l_residual2 = acts.residual2 + (size_t)l * B * T * C;
        float *l_ln2 = acts.ln2 + (size_t)l * B * T * C;
        float *l_ln2_mean = acts.ln2_mean + (size_t)l * B * T;
        float *l_ln2_rstd = acts.ln2_rstd + (size_t)l * B * T;
        float *l_fch = acts.fch + (size_t)l * B * T * 4 * C;
        float *l_fch_gelu = acts.fch_gelu + (size_t)l * B * T * 4 * C;
        float *l_fcproj = acts.fcproj + (size_t)l * B * T * C;
        float *l_residual3 = acts.residual3 + (size_t)l * B * T * C;
        float *scratch = acts.output;
        layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B * T, C);
        matmul_forward(scratch, l_ln1, l_qkvw, l_qkvb, B, T, C, 3 * C);
        attention_forward(l_atty, l_qkvr, l_att, scratch, B, T, C, NH);
        matmul_forward(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C);
        residual_forward(l_residual2, residual, l_attproj, (size_t)B * T * C);
        layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B * T, C);
        matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4 * C);
        gelu_forward(l_fch_gelu, l_fch, (size_t)B * T * 4 * C);
        matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4 * C, C);
        residual_forward(l_residual3, l_residual2, l_fcproj, (size_t)B * T * C);
    }
    float *residual = acts.residual3 + (size_t)(L - 1) * B * T * C;
    layernorm_forward(acts.lnf, acts.lnf_mean, acts.lnf_rstd, residual,
                      params.lnfw, params.lnfb, B * T, C);
    matmul_forward(acts.output, acts.lnf, params.wte, NULL, B, T, C, Vp);
    if (targets != NULL) {
        fused_classifier(acts.output, acts.losses, model->targets, B, T, V, Vp);
        qq().memcpy(model->cpu_losses, acts.losses, (size_t)B * T * sizeof(float))
            .wait_and_throw();
        double acc = 0.0;
        for (int i = 0; i < B * T; i++) acc += model->cpu_losses[i];
        model->mean_loss = (float)(acc / (B * T));
    } else {
        model->mean_loss = -1.0f;
    }
}

static void gpt2_zero_grad(GPT2 *model) {
    if (model->grads_acts_memory != NULL)
        qq().fill(model->grads_acts_memory, 0.0f, model->num_grad_acts).wait_and_throw();
    if (model->grads_memory != NULL)
        qq().fill(model->grads_memory, 0.0f, model->num_parameters).wait_and_throw();
}

static void gpt2_backward(GPT2 *model) {
    if (model->mean_loss == -1.0f) { fprintf(stderr, "forward first\n"); exit(EXIT_FAILURE); }
    if (model->grads_memory == NULL) {
        model->grads_memory = alloc_params(&model->grads, model->param_sizes, 1);
        printf("allocated %zu MiB for parameter gradients\n", (model->num_parameters * sizeof(float)) >> 20);
        size_t bw[NUM_BACKWARD_TENSORS];
        GPT2Config cfg = model->config;
        cfg.num_layers = 1;
        fill_in_grad_act_sizes(bw, model->batch_size, model->seq_len, cfg);
        model->grads_acts_memory = alloc_backward(&model->grads_acts, bw);
        model->num_grad_acts = 0;
        for (int i = 0; i < NUM_BACKWARD_TENSORS; i++) model->num_grad_acts += bw[i];
        printf("allocated %zu MiB for activation gradients\n", (model->num_grad_acts * sizeof(float)) >> 20);
        gpt2_zero_grad(model);
    }
    int B = model->batch_size, T = model->seq_len;
    int Vp = model->config.padded_vocab_size;
    int L = model->config.num_layers;
    int NH = model->config.num_heads;
    int C = model->config.channels;
    ParameterTensors params = model->params;
    ParameterTensors grads = model->grads;
    ActivationTensors acts = model->acts;
    GradActTensors ga = model->grads_acts;
    matmul_backward(ga.bt4c, grads.wte, NULL, acts.output, acts.lnf, params.wte, B, T, C, Vp);
    float *residual = acts.residual3 + (size_t)(L - 1) * B * T * C;
    float *dresidual = ga.residual3;
    layernorm_backward(dresidual, grads.lnfw, grads.lnfb, ga.bt4c, residual,
                       params.lnfw, acts.lnf_mean, acts.lnf_rstd, B, T, C);
    for (int l = L - 1; l >= 0; l--) {
        float *res = (l == 0) ? acts.encoded : acts.residual3 + (size_t)(l - 1) * B * T * C;
        float *l_qkvw = params.qkvw + (size_t)l * 3 * C * C;
        float *l_attprojw = params.attprojw + (size_t)l * C * C;
        float *l_ln2w = params.ln2w + (size_t)l * C;
        float *l_fcw = params.fcw + (size_t)l * 4 * C * C;
        float *l_fcprojw = params.fcprojw + (size_t)l * C * 4 * C;
        float *l_ln1w = params.ln1w + (size_t)l * C;
        float *dl_ln1w = grads.ln1w + (size_t)l * C;
        float *dl_ln1b = grads.ln1b + (size_t)l * C;
        float *dl_qkvw = grads.qkvw + (size_t)l * 3 * C * C;
        float *dl_qkvb = grads.qkvb + (size_t)l * 3 * C;
        float *dl_attprojw = grads.attprojw + (size_t)l * C * C;
        float *dl_attprojb = grads.attprojb + (size_t)l * C;
        float *dl_ln2w = grads.ln2w + (size_t)l * C;
        float *dl_ln2b = grads.ln2b + (size_t)l * C;
        float *dl_fcw = grads.fcw + (size_t)l * 4 * C * C;
        float *dl_fcb = grads.fcb + (size_t)l * 4 * C;
        float *dl_fcprojw = grads.fcprojw + (size_t)l * C * 4 * C;
        float *dl_fcprojb = grads.fcprojb + (size_t)l * C;
        float *l_ln1 = acts.ln1 + (size_t)l * B * T * C;
        float *l_ln1_mean = acts.ln1_mean + (size_t)l * B * T;
        float *l_ln1_rstd = acts.ln1_rstd + (size_t)l * B * T;
        float *l_qkvr = acts.qkvr + (size_t)l * B * T * 3 * C;
        float *l_atty = acts.atty + (size_t)l * B * T * C;
        float *l_att = acts.att + (size_t)l * B * NH * T * T;
        float *l_residual2 = acts.residual2 + (size_t)l * B * T * C;
        float *l_ln2 = acts.ln2 + (size_t)l * B * T * C;
        float *l_ln2_mean = acts.ln2_mean + (size_t)l * B * T;
        float *l_ln2_rstd = acts.ln2_rstd + (size_t)l * B * T;
        float *l_fch = acts.fch + (size_t)l * B * T * 4 * C;
        float *l_fch_gelu = acts.fch_gelu + (size_t)l * B * T * 4 * C;
        float *dl_btc = acts.lnf;
        float *dl_bt4c = ga.bt4c;
        float *dl_preatt = ga.preatt;
        float *scratch = acts.output;
        matmul_backward(dl_bt4c, dl_fcprojw, dl_fcprojb, dresidual, l_fch_gelu, l_fcprojw,
                        B, T, 4 * C, C);
        gelu_backward(dl_bt4c, l_fch, dl_bt4c, (size_t)B * T * 4 * C);
        matmul_backward(dl_btc, dl_fcw, dl_fcb, dl_bt4c, l_ln2, l_fcw, B, T, C, 4 * C);
        layernorm_backward(dresidual, dl_ln2w, dl_ln2b, dl_btc, l_residual2, l_ln2w,
                           l_ln2_mean, l_ln2_rstd, B, T, C);
        matmul_backward(dl_btc, dl_attprojw, dl_attprojb, dresidual, l_atty, l_attprojw,
                        B, T, C, C);
        float *buffer_a = l_atty;
        float *buffer_b = l_fch;
        attention_backward(dl_bt4c, buffer_b, dl_preatt, scratch, buffer_a, dl_btc,
                           l_qkvr, l_att, B, T, C, NH);
        matmul_backward(dl_btc, dl_qkvw, dl_qkvb, dl_bt4c, l_ln1, l_qkvw, B, T, C, 3 * C);
        layernorm_backward(dresidual, dl_ln1w, dl_ln1b, dl_btc, res, l_ln1w,
                           l_ln1_mean, l_ln1_rstd, B, T, C);
    }
    encoder_backward(grads.wte, grads.wpe, dresidual, model->inputs, B, T, C);
}

static void gpt2_update(GPT2 *model, float lr, float b1, float b2, float eps,
                        float wd, int t) {
    if (model->m_memory == NULL) {
        model->m_memory = sycl::malloc_device<float>(model->num_parameters, qq());
        model->v_memory = sycl::malloc_device<float>(model->num_parameters, qq());
        if (!model->m_memory || !model->v_memory) { fprintf(stderr, "adam alloc\n"); exit(EXIT_FAILURE); }
        qq().fill(model->m_memory, 0.0f, model->num_parameters).wait_and_throw();
        qq().fill(model->v_memory, 0.0f, model->num_parameters).wait_and_throw();
        printf("allocated %zu MiB for AdamW optimizer state m\n", (model->num_parameters * sizeof(float)) >> 20);
        printf("allocated %zu MiB for AdamW optimizer state v\n", (model->num_parameters * sizeof(float)) >> 20);
    }
    float b1c = 1.0f - powf(b1, (float)t);
    float b2c = 1.0f - powf(b2, (float)t);
    adamw_update(model->params_memory, model->grads_memory, model->m_memory,
                 model->v_memory, model->num_parameters, lr, b1, b2, b1c, b2c, eps, wd);
}

static void gpt2_free(GPT2 *model) {
    if (model->params_memory != NULL) sycl::free(model->params_memory, qq());
    if (model->grads_memory != NULL) sycl::free(model->grads_memory, qq());
    if (model->m_memory != NULL) sycl::free(model->m_memory, qq());
    if (model->v_memory != NULL) sycl::free(model->v_memory, qq());
    if (model->acts_memory != NULL) sycl::free(model->acts_memory, qq());
    if (model->grads_acts_memory != NULL) sycl::free(model->grads_acts_memory, qq());
    if (model->inputs != NULL) sycl::free(model->inputs, qq());
    if (model->targets != NULL) sycl::free(model->targets, qq());
    if (model->cpu_losses != NULL) free(model->cpu_losses);
}

// ----------------------------------------------------------------------------
// sampler: takes probabilities and samples integers from them

#define GPT2_EOT 50256

static unsigned int random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
static float random_f32(unsigned long long *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

static int sample_softmax(const float *logits, int n, float coin) {
    // sample index from logits (converted to probabilities using softmax)
    // coin is a random number in [0, 1), usually from random_f32()
    double norm = 0;
    for (int i = 0; i < n; i++) {
        norm += expf(logits[i]);
    }
    // instead of dividing all exp(logits), we can just multiply coin.
    coin *= (float)norm;
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += expf(logits[i]);
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

// ----------------------------------------------------------------------------
// Logger lite

typedef struct {
    FILE *logfile;
    int flush_every;
} Logger;

static void logger_init(Logger *logger, const char *filename) {
    logger->flush_every = 20;
    logger->logfile = NULL;
    if (filename != NULL) { logger->logfile = fopenCheck(filename, "w"); }
}

static void logger_log_val(Logger *logger, int step, float val_loss) {
    if (logger->logfile != NULL) {
        fprintf(logger->logfile, "s:%d tel:%.4f\n", step, val_loss);
    }
}

static void logger_log_train(Logger *logger, int step, float train_loss) {
    if (logger->logfile != NULL) {
        fprintf(logger->logfile, "s:%d trl:%.4f\n", step, train_loss);
        if (step % 10 == 0) { fflush(logger->logfile); }
    }
}

static void logger_free(Logger *logger) {
    if (logger->logfile != NULL) { fclose(logger->logfile); }
}

// ----------------------------------------------------------------------------
// CLI

static void error_usage() {
    fprintf(stderr, "Usage:   ./train_gpt2fp32cu [options]\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -i <string> train data filename pattern (default = dev/data/tinyshakespeare/tiny_shakespeare_train.bin)\n");
    fprintf(stderr, "  -j <string> val data filename pattern (default = dev/data/tinyshakespeare/tiny_shakespeare_val.bin)\n");
    fprintf(stderr, "  -o <string> output log file (default = NULL)\n");
    fprintf(stderr, "  -b <int>    batch size B (default = 4)\n");
    fprintf(stderr, "  -t <int>    sequence length T (default = 1024)\n");
    fprintf(stderr, "  -l <float>  learning rate (default = 3e-4f)\n");
    fprintf(stderr, "  -v <int>    val_loss_every, how often we evaluate val loss (default = 20)\n");
    fprintf(stderr, "  -m <int>    val_max_steps, up to how many val batches to estimate val loss? (default = 20)\n");
    fprintf(stderr, "  -s <int>    sample_every, how often we inference the model (default = 20)\n");
    fprintf(stderr, "  -g <int>    genT, how many steps of inference we do (default = 64)\n");
    exit(EXIT_FAILURE);
}

// ----------------------------------------------------------------------------
// main training loop (llm.c real main, SYCL device for all model compute)
#ifndef TESTING
int main(int argc, char *argv[]) {
    // read in the (optional) command line arguments
    const char *train_data_pattern = "dev/data/tinyshakespeare/tiny_shakespeare_train.bin";
    const char *val_data_pattern = "dev/data/tinyshakespeare/tiny_shakespeare_val.bin";
    const char *output_log_file = NULL;
    int B = 4;
    int T = 1024;
    float learning_rate = 3e-4f;
    int val_loss_every = 20;
    int val_max_steps = 20;
    int sample_every = 20;
    int genT = 64;
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 >= argc) { error_usage(); }
        if (argv[i][0] != '-') { error_usage(); }
        if (strlen(argv[i]) != 2) { error_usage(); }
        if (argv[i][1] == 'i') { train_data_pattern = argv[i+1]; }
        else if (argv[i][1] == 'j') { val_data_pattern = argv[i+1]; }
        else if (argv[i][1] == 'o') { output_log_file = argv[i+1]; }
        else if (argv[i][1] == 'b') { B = atoi(argv[i+1]); }
        else if (argv[i][1] == 't') { T = atoi(argv[i+1]); }
        else if (argv[i][1] == 'l') { learning_rate = (float)atof(argv[i+1]); }
        else if (argv[i][1] == 'v') { val_loss_every = atoi(argv[i+1]); }
        else if (argv[i][1] == 'm') { val_max_steps = atoi(argv[i+1]); }
        else if (argv[i][1] == 's') { sample_every = atoi(argv[i+1]); }
        else if (argv[i][1] == 'g') { genT = atoi(argv[i+1]); }
        else { error_usage(); }
    }
    printf("+-----------------------+----------------------------------------------------+\n");
    printf("| Parameter             | Value                                              |\n");
    printf("+-----------------------+----------------------------------------------------+\n");
    printf("| train data pattern    | %-50s |\n", train_data_pattern);
    printf("| val data pattern      | %-50s |\n", val_data_pattern);
    printf("| output log file       | %-50s |\n", output_log_file == NULL ? "NULL" : output_log_file);
    printf("| batch size B          | %-50d |\n", B);
    printf("| sequence length T     | %-50d |\n", T);
    printf("| learning rate         | %-50f |\n", learning_rate);
    printf("| val_loss_every        | %-50d |\n", val_loss_every);
    printf("| val_max_steps         | %-50d |\n", val_max_steps);
    printf("| sample_every          | %-50d |\n", sample_every);
    printf("| genT                  | %-50d |\n", genT);
    printf("+-----------------------+----------------------------------------------------+\n");

    try {
        sycl::queue q{ sycl::property::queue::in_order{} };
        g_q = &q;
        std::string devname = q.get_device().get_info<sycl::info::device::name>();
        printf("| device                | %-50s |\n", devname.c_str());
        printf("+-----------------------+----------------------------------------------------+\n");

        // build the GPT-2 model from a checkpoint
        GPT2 model;
        gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");
        printf("| max_sequence_length T | %-50d |\n", model.config.max_seq_len);
        printf("| vocab_size V          | %-50d |\n", model.config.vocab_size);
        printf("| padded_vocab_size Vp  | %-50d |\n", model.config.padded_vocab_size);
        printf("| num_layers L          | %-50d |\n", model.config.num_layers);
        printf("| num_heads NH          | %-50d |\n", model.config.num_heads);
        printf("| channels C            | %-50d |\n", model.config.channels);
        printf("| num_parameters        | %-50zu |\n", model.num_parameters);
        printf("+-----------------------+----------------------------------------------------+\n");

        // build DataLoaders for both train and val
        DataLoader train_loader, val_loader;
        dataloader_init(&train_loader, train_data_pattern, (size_t)B, (size_t)T, 0, 1, 1);
        dataloader_init(&val_loader, val_data_pattern, (size_t)B, (size_t)T, 0, 1, 0);
        int train_num_batches = (int)(train_loader.num_tokens / ((size_t)B * (size_t)T));
        int val_num_batches = (int)(val_loader.num_tokens / ((size_t)B * (size_t)T));
        if (val_num_batches > val_max_steps) { val_num_batches = val_max_steps; }
        printf("| train_num_batches     | %-50d |\n", train_num_batches);
        printf("| val_num_batches       | %-50d |\n", val_num_batches);
        printf("+-----------------------+----------------------------------------------------+\n");

        printf("allocated %d MiB for model parameters\n", (int)round((double)model.num_parameters * sizeof(float) / (1024 * 1024)));

        // set up the Logger
        Logger logger;
        logger_init(&logger, output_log_file);

        // build the Tokenizer
        Tokenizer tokenizer;
        tokenizer_init(&tokenizer, "gpt2_tokenizer.bin");

        // some memory for generating samples from the model
        unsigned long long rng_state = 1337;
        int *gen_tokens = (int *)mallocCheck((size_t)B * T * sizeof(int));
        float *cpu_logits = (float *)mallocCheck((size_t)model.config.vocab_size * sizeof(float));

        // train
        struct timespec start, end;
        double total_sum_iteration_time_s = 0.0;
        for (int step = 0; step <= train_num_batches; step++) {
            int last_step = step == train_num_batches;

            // once in a while estimate the validation loss
            if (step % val_loss_every == 0 || last_step) {
                float val_loss = 0.0f;
                dataloader_reset(&val_loader);
                for (int i = 0; i < val_num_batches; i++) {
                    dataloader_next_batch(&val_loader);
                    gpt2_forward(&model, val_loader.inputs, val_loader.targets, B, T);
                    val_loss += model.mean_loss;
                }
                val_loss /= val_num_batches;
                printf("val loss %f\n", val_loss);
                logger_log_val(&logger, step, val_loss);
            }

            // once in a while do model inference to print generated text
            if ((step > 0 && step % sample_every == 0) || last_step) {
                for (int i = 0; i < B * T; ++i) {
                    gen_tokens[i] = GPT2_EOT;
                }
                printf("generating:\n---\n");
                for (int t = 1; t < genT; t++) {
                    gpt2_forward(&model, gen_tokens, NULL, B, T);
                    float *logits = model.acts.output + (size_t)(t - 1) * (size_t)model.config.padded_vocab_size;
                    qq().memcpy(cpu_logits, logits, (size_t)model.config.vocab_size * sizeof(float)).wait_and_throw();
                    float coin = random_f32(&rng_state);
                    int next_token = sample_softmax(cpu_logits, model.config.vocab_size, coin);
                    gen_tokens[t] = next_token;
                    if (tokenizer.init_ok) {
                        const char *token_str = tokenizer_decode(&tokenizer, (uint32_t)next_token);
                        safe_printf(token_str);
                    } else {
                        printf("%d ", next_token);
                    }
                    fflush(stdout);
                }
                printf("\n---\n");
            }

            if (last_step) { break; }

            // do a training step
            clock_gettime(CLOCK_MONOTONIC, &start);
            dataloader_next_batch(&train_loader);
            gpt2_forward(&model, train_loader.inputs, train_loader.targets, B, T);
            gpt2_zero_grad(&model);
            gpt2_backward(&model);
            gpt2_update(&model, learning_rate, 0.9f, 0.999f, 1e-8f, 0.0f, step + 1);
            qq().wait_and_throw();
            clock_gettime(CLOCK_MONOTONIC, &end);
            double time_elapsed_s = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec) / 1e9;
            total_sum_iteration_time_s += time_elapsed_s;
            int tokens_per_second = (int)(((size_t)B * (size_t)T) / time_elapsed_s);
            printf("step %4d/%d: train loss %f (%f ms, %d tok/s)\n", step + 1, train_num_batches, model.mean_loss, time_elapsed_s * 1000, tokens_per_second);
            logger_log_train(&logger, step, model.mean_loss);
        }
        printf("total average iteration time: %f ms\n", total_sum_iteration_time_s / train_num_batches * 1000);

        // free
        dataloader_free(&train_loader);
        dataloader_free(&val_loader);
        tokenizer_free(&tokenizer);
        gpt2_free(&model);
        free(cpu_logits);
        free(gen_tokens);
        logger_free(&logger);
    } catch (sycl::exception const &e) {
        fprintf(stderr, "SYCL error: %s\n", e.what());
        return 1;
    }

    return 0;
}
#endif
