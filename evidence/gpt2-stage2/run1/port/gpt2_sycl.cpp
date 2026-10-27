// GPT-2 fp32 training (forward + backward) ported to idiomatic SYCL 2020.
// Computes on the SYCL device; oneMKL row_major GEMM replaces cuBLAS.
// Contract main: <debug_state.bin> <output> <gpt2_124M.bin> writes all
// parameter gradients then mean loss as raw float32.
#include <sycl/sycl.hpp>
#include <oneapi/math.hpp>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <cstring>
#include <cassert>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

static sycl::queue *g_q = nullptr;
static sycl::queue &Q() { return *g_q; }

// ---------------------------------------------------------------- file utils
static FILE *xfopen(const char *path, const char *mode) {
  FILE *fp = std::fopen(path, mode);
  if (!fp) { std::fprintf(stderr, "fopen failed: %s\n", path); std::exit(EXIT_FAILURE); }
  return fp;
}
static void xfread(void *ptr, std::size_t sz, std::size_t n, FILE *s) {
  if (std::fread(ptr, sz, n, s) != n) { std::fprintf(stderr, "fread failed\n"); std::exit(EXIT_FAILURE); }
}

// ---------------------------------------------------------------- kernels
static void encoder_forward(float *out, const int *inp, const float *wte,
                            const float *wpe, int B, int T, int C) {
  std::size_t N = (std::size_t)B * T * C;
  Q().parallel_for(sycl::range<1>(N), [=](sycl::id<1> id) {
    std::size_t i = id[0];
    std::size_t c = i % (std::size_t)C;
    std::size_t bt = i / (std::size_t)C;
    std::size_t t = bt % (std::size_t)T;
    int ix = inp[bt];
    out[i] = wte[(std::size_t)ix * C + c] + wpe[t * (std::size_t)C + c];
  });
}

static void encoder_backward(float *dwte, float *dwpe, const float *dout,
                             const int *inp, int B, int T, int C) {
  std::size_t N = (std::size_t)B * T * C;
  Q().parallel_for(sycl::range<1>(N), [=](sycl::id<1> id) {
    std::size_t i = id[0];
    std::size_t c = i % (std::size_t)C;
    std::size_t bt = i / (std::size_t)C;
    std::size_t t = bt % (std::size_t)T;
    int ix = inp[bt];
    float v = dout[i];
    using Aref = sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                  sycl::memory_scope::device,
                                  sycl::access::address_space::global_space>;
    Aref a1(dwte[(std::size_t)ix * C + c]);
    a1.fetch_add(v);
    Aref a2(dwpe[t * (std::size_t)C + c]);
    a2.fetch_add(v);
  });
}

static void layernorm_forward(float *out, float *mean, float *rstd, const float *inp,
                              const float *weight, const float *bias, int B, int T, int C) {
  std::size_t N = (std::size_t)B * T;
  Q().parallel_for(sycl::range<1>(N), [=](sycl::id<1> id) {
    std::size_t r = id[0];
    const float *x = inp + r * (std::size_t)C;
    float *o = out + r * (std::size_t)C;
    double sum = 0.0;
    for (int c = 0; c < C; ++c) sum += (double)x[c];
    float m = (float)(sum / C);
    if (mean) mean[r] = m;
    double var = 0.0;
    for (int c = 0; c < C; ++c) { double d = (double)x[c] - (double)m; var += d * d; }
    float s = 1.0f / sycl::sqrt((float)(var / C) + 1e-5f);
    if (rstd) rstd[r] = s;
    for (int c = 0; c < C; ++c) {
      float nrm = s * (x[c] - m);
      o[c] = nrm * weight[c] + bias[c];
    }
  });
}

static void layernorm_backward(float *dinp, float *dweight, float *dbias, const float *dout,
                               const float *inp, const float *weight, const float *mean,
                               const float *rstd, int B, int T, int C) {
  std::size_t N = (std::size_t)B * T;
  Q().parallel_for(sycl::range<1>(N), [=](sycl::id<1> id) {
    std::size_t r = id[0];
    const float *dout_bt = dout + r * (std::size_t)C;
    const float *inp_bt = inp + r * (std::size_t)C;
    float *dinp_bt = dinp + r * (std::size_t)C;
    float m = mean[r], s = rstd[r];
    double dm = 0.0, dnm = 0.0;
    for (int c = 0; c < C; ++c) {
      float nrm = (inp_bt[c] - m) * s;
      float dn = weight[c] * dout_bt[c];
      dm += (double)dn;
      dnm += (double)dn * (double)nrm;
    }
    float fdm = (float)(dm / C), fdnm = (float)(dnm / C);
    using Aref = sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                  sycl::memory_scope::device,
                                  sycl::access::address_space::global_space>;
    for (int c = 0; c < C; ++c) {
      float nrm = (inp_bt[c] - m) * s;
      float dn = weight[c] * dout_bt[c];
      Aref ab(dbias[c]);
      ab.fetch_add(dout_bt[c]);
      Aref aw(dweight[c]);
      aw.fetch_add(nrm * dout_bt[c]);
      float dval = (dn - fdm - nrm * fdnm) * s;
      dinp_bt[c] += dval;
    }
  });
}

static void permute_f(float *q, float *k, float *v, const float *inp,
                      int B, int N, int NH, int d) {
  std::size_t tot = (std::size_t)B * NH * N * d;
  Q().parallel_for(sycl::range<1>(tot), [=](sycl::id<1> id) {
    std::size_t i = id[0];
    std::size_t d_ = i % (std::size_t)d;
    std::size_t rem = i / (std::size_t)d;
    std::size_t n = rem % (std::size_t)N;
    rem /= (std::size_t)N;
    std::size_t nh = rem % (std::size_t)NH;
    std::size_t b = rem / (std::size_t)NH;
    std::size_t base = ((b * (std::size_t)N + n) * 3 * NH + 0 * NH + nh) * (std::size_t)d + d_;
    std::size_t NHd = (std::size_t)NH * d;
    q[i] = inp[base];
    k[i] = inp[base + NHd];
    v[i] = inp[base + 2 * NHd];
  });
}

static void permute_b(float *dinp, const float *dq, const float *dk, const float *dv,
                      int B, int N, int NH, int d) {
  std::size_t tot = (std::size_t)B * NH * N * d;
  Q().parallel_for(sycl::range<1>(tot), [=](sycl::id<1> id) {
    std::size_t i = id[0];
    std::size_t d_ = i % (std::size_t)d;
    std::size_t rem = i / (std::size_t)d;
    std::size_t n = rem % (std::size_t)N;
    rem /= (std::size_t)N;
    std::size_t nh = rem % (std::size_t)NH;
    std::size_t b = rem / (std::size_t)NH;
    std::size_t base = ((b * (std::size_t)N + n) * 3 * NH + nh) * (std::size_t)d + d_;
    std::size_t NHd = (std::size_t)NH * d;
    dinp[base] = dq[i];
    dinp[base + NHd] = dk[i];
    dinp[base + 2 * NHd] = dv[i];
  });
}

static void unpermute_f(const float *inp, float *out, int B, int N, int NH, int d) {
  std::size_t tot = (std::size_t)B * NH * N * d;
  Q().parallel_for(sycl::range<1>(tot), [=](sycl::id<1> id) {
    std::size_t i = id[0];
    std::size_t d_ = i % (std::size_t)d;
    std::size_t rem = i / (std::size_t)d;
    std::size_t n = rem % (std::size_t)N;
    rem /= (std::size_t)N;
    std::size_t nh = rem % (std::size_t)NH;
    std::size_t b = rem / (std::size_t)NH;
    std::size_t o = ((b * (std::size_t)N + n) * NH + nh) * (std::size_t)d + d_;
    out[o] = inp[i];
  });
}

static void unpermute_b(float *dinp, const float *dout, int B, int N, int NH, int d) {
  std::size_t tot = (std::size_t)B * NH * N * d;
  Q().parallel_for(sycl::range<1>(tot), [=](sycl::id<1> id) {
    std::size_t i = id[0];
    std::size_t d_ = i % (std::size_t)d;
    std::size_t rem = i / (std::size_t)d;
    std::size_t n = rem % (std::size_t)N;
    rem /= (std::size_t)N;
    std::size_t nh = rem % (std::size_t)NH;
    std::size_t b = rem / (std::size_t)NH;
    std::size_t o = ((b * (std::size_t)N + n) * NH + nh) * (std::size_t)d + d_;
    dinp[i] = dout[o];
  });
}

static void softmax_forward(float *out, float scale, const float *inp, int N, int T) {
  std::size_t rows = (std::size_t)N * T;
  Q().parallel_for(sycl::range<1>(rows), [=](sycl::id<1> id) {
    std::size_t r = id[0];
    int own = (int)(r % (std::size_t)T);
    const float *x = inp + r * (std::size_t)T;
    float *o = out + r * (std::size_t)T;
    float mx = -FLT_MAX;
    for (int i = 0; i <= own; ++i) mx = sycl::fmax(mx, x[i]);
    float sum = 0.0f;
    for (int i = 0; i <= own; ++i) sum += sycl::exp(scale * (x[i] - mx));
    float inv = 1.0f / sum;
    for (int i = 0; i <= own; ++i) o[i] = sycl::exp(scale * (x[i] - mx)) * inv;
  });
}

static void softmax_backward(float *dpreatt, const float *datt, const float *att,
                             int B, int NH, int T, float scale) {
  std::size_t rows = (std::size_t)B * NH * T;
  Q().parallel_for(sycl::range<1>(rows), [=](sycl::id<1> id) {
    std::size_t r = id[0];
    int t = (int)(r % (std::size_t)T);
    const float *a = att + r * (std::size_t)T;
    const float *da = datt + r * (std::size_t)T;
    float *dp = dpreatt + r * (std::size_t)T;
    float s = 0.0f;
    for (int i = 0; i <= t; ++i) s += a[i] * da[i];
    for (int i = 0; i <= t; ++i) dp[i] = scale * a[i] * (da[i] - s);
  });
}

static void residual_forward(float *out, const float *a, const float *b, std::size_t N) {
  Q().parallel_for(sycl::range<1>(N), [=](sycl::id<1> id) {
    std::size_t i = id[0];
    out[i] = a[i] + b[i];
  });
}

static void gelu_forward(float *out, const float *inp, std::size_t N) {
  const float g = sycl::sqrt(2.0f / 3.14159265358979323846f);
  Q().parallel_for(sycl::range<1>(N), [=](sycl::id<1> id) {
    std::size_t i = id[0];
    float x = inp[i];
    float cube = 0.044715f * x * x * x;
    out[i] = 0.5f * x * (1.0f + sycl::tanh(g * (x + cube)));
  });
}

static void gelu_backward(float *dinp, const float *inp, const float *dout, std::size_t N) {
  const float g = sycl::sqrt(2.0f / 3.14159265358979323846f);
  Q().parallel_for(sycl::range<1>(N), [=](sycl::id<1> id) {
    std::size_t i = id[0];
    float x = inp[i];
    float ta = g * (x + 0.044715f * x * x * x);
    float th = sycl::tanh(ta);
    float se = 1.0f / (sycl::cosh(ta) * sycl::cosh(ta));
    float lg = 0.5f * (1.0f + th) +
               x * 0.5f * se * g * (1.0f + 3.0f * 0.044715f * x * x);
    dinp[i] = lg * dout[i];
  });
}

static void bias_backward(float *dbias, const float *dout, int B, int T, int OC) {
  std::size_t rows = (std::size_t)B * T;
  Q().parallel_for(sycl::range<1>((std::size_t)OC), [=](sycl::id<1> id) {
    std::size_t c = id[0];
    double s = 0.0;
    for (std::size_t r = 0; r < rows; ++r) s += (double)dout[r * OC + c];
    dbias[c] += (float)s;
  });
}

static void bias_add(float *out, const float *bias, int B, int T, int OC) {
  std::size_t N = (std::size_t)B * T * OC;
  Q().parallel_for(sycl::range<1>(N), [=](sycl::id<1> id) {
    std::size_t i = id[0];
    out[i] += bias[i % (std::size_t)OC];
  });
}

// fused classifier: in-place replaces logits with dlogits, writes losses
static void fused_classifier(float *logits, float *losses, const int *targets,
                             int B, int T, int V, int P) {
  std::size_t N = (std::size_t)B * T;
  float dloss = 1.0f / (float)N;
  Q().parallel_for(sycl::range<1>(N), [=](sycl::id<1> id) {
    std::size_t r = id[0];
    float *row = logits + r * (std::size_t)P;
    int ix = targets[r];
    float mx = -FLT_MAX;
    for (int i = 0; i < V; ++i) mx = sycl::fmax(mx, row[i]);
    float sum = 0.0f;
    for (int i = 0; i < V; ++i) sum += sycl::exp(row[i] - mx);
    float scale = 1.0f / sum;
    float prob = sycl::exp(row[ix] - mx) * scale;
    losses[r] = -sycl::log(prob);
    for (int i = 0; i < V; ++i) {
      float p = sycl::exp(row[i] - mx) * scale;
      row[i] = (p - (i == ix ? 1.0f : 0.0f)) * dloss;
    }
    for (int i = V; i < P; ++i) row[i] = 0.0f;
  });
}

// ---------------------------------------------------------------- GEMM wrappers (row-major oneMKL)
static void sgemm_nn(std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                     const float *a, std::int64_t lda, const float *b, std::int64_t ldb,
                     float beta, float *c, std::int64_t ldc) {
  oneapi::math::blas::row_major::gemm(Q(), oneapi::math::transpose::nontrans,
                                      oneapi::math::transpose::nontrans, m, n, k, alpha,
                                      a, lda, b, ldb, beta, c, ldc).wait();
}
static void sgemm_nt(std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                     const float *a, std::int64_t lda, const float *b, std::int64_t ldb,
                     float beta, float *c, std::int64_t ldc) {
  oneapi::math::blas::row_major::gemm(Q(), oneapi::math::transpose::nontrans,
                                      oneapi::math::transpose::trans, m, n, k, alpha,
                                      a, lda, b, ldb, beta, c, ldc).wait();
}
static void sgemm_tn(std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                     const float *a, std::int64_t lda, const float *b, std::int64_t ldb,
                     float beta, float *c, std::int64_t ldc) {
  oneapi::math::blas::row_major::gemm(Q(), oneapi::math::transpose::trans,
                                      oneapi::math::transpose::nontrans, m, n, k, alpha,
                                      a, lda, b, ldb, beta, c, ldc).wait();
}
static void sgemm_batch(oneapi::math::transpose ta, oneapi::math::transpose tb,
                        std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                        const float *a, std::int64_t lda, std::int64_t sa,
                        const float *b, std::int64_t ldb, std::int64_t sb, float beta,
                        float *c, std::int64_t ldc, std::int64_t sc, std::int64_t batch) {
  oneapi::math::blas::row_major::gemm_batch(Q(), ta, tb, m, n, k, alpha, a, lda, sa, b,
                                            ldb, sb, beta, c, ldc, sc, batch).wait();
}

static void matmul_forward(float *out, const float *inp, const float *weight,
                           const float *bias, int B, int T, int C, int OC) {
  std::int64_t bt = (std::int64_t)B * T;
  sgemm_nt(bt, OC, C, 1.0f, inp, C, weight, C, 0.0f, out, OC);
  if (bias) bias_add(out, bias, B, T, OC);
}

static void matmul_backward(float *dinp, float *dweight, float *dbias, float *dout,
                            float *inp, float *weight, int B, int T, int C, int OC) {
  std::int64_t bt = (std::int64_t)B * T;
  sgemm_nn(bt, C, OC, 1.0f, dout, OC, weight, C, 0.0f, dinp, C);
  sgemm_tn(OC, C, bt, 1.0f, dout, OC, inp, C, 1.0f, dweight, C);
  if (dbias) bias_backward(dbias, dout, B, T, OC);
}

static void attention_forward(float *out, float *qkvr, float *att, float *inp,
                              int B, int T, int C, int NH) {
  constexpr auto NT = oneapi::math::transpose::nontrans;
  constexpr auto TT = oneapi::math::transpose::trans;
  int HS = C / NH;
  float *q = qkvr, *k = qkvr + (std::size_t)B * T * C, *v = qkvr + 2 * (std::size_t)B * T * C;
  permute_f(q, k, v, inp, B, T, NH, HS);
  Q().wait();
  float *preatt = inp;
  std::int64_t b = (std::int64_t)B * NH, stride_qkv = (std::int64_t)T * HS,
               stride_a = (std::int64_t)T * T;
  sgemm_batch(NT, TT, T, T, HS, 1.0f, q, HS, stride_qkv, k, HS, stride_qkv, 0.0f, preatt,
              T, stride_a, b);
  float scale = 1.0f / sycl::sqrt((float)HS);
  softmax_forward(att, scale, preatt, B * NH, T);
  Q().wait();
  float *vaccum = inp;
  sgemm_batch(NT, NT, T, HS, T, 1.0f, att, T, stride_a, v, HS, stride_qkv, 0.0f, vaccum,
              HS, stride_qkv, b);
  unpermute_f(vaccum, out, B, T, NH, HS);
}

static void attention_backward(float *dinp, float *dqkvr, float *dpreatt, float *datt,
                               float *scratch, const float *dout, const float *qkvr,
                               const float *att, int B, int T, int C, int NH) {
  constexpr auto NT = oneapi::math::transpose::nontrans;
  constexpr auto TT = oneapi::math::transpose::trans;
  int HS = C / NH;
  const float *q = qkvr, *k = qkvr + (std::size_t)B * T * C,
              *v = qkvr + 2 * (std::size_t)B * T * C;
  float *dq = dqkvr, *dk = dqkvr + (std::size_t)B * T * C, *dv = dqkvr + 2 * (std::size_t)B * T * C;
  std::int64_t b = (std::int64_t)B * NH, sq = (std::int64_t)T * HS, sa = (std::int64_t)T * T;
  unpermute_b(scratch, dout, B, T, NH, HS);
  Q().wait();
  sgemm_batch(NT, TT, T, T, HS, 1.0f, scratch, HS, sq, v, HS, sq, 0.0f, datt, T, sa, b);
  sgemm_batch(TT, NT, T, HS, T, 1.0f, att, T, sa, scratch, HS, sq, 0.0f, dv, HS, sq, b);
  float scale = 1.0f / sycl::sqrt((float)HS);
  softmax_backward(dpreatt, datt, att, B, NH, T, scale);
  Q().wait();
  sgemm_batch(NT, NT, T, HS, T, 1.0f, dpreatt, T, sa, k, HS, sq, 0.0f, dq, HS, sq, b);
  sgemm_batch(TT, NT, T, HS, T, 1.0f, dpreatt, T, sa, q, HS, sq, 0.0f, dk, HS, sq, b);
  permute_b(dinp, dq, dk, dv, B, T, NH, HS);
}

// ---------------------------------------------------------------- model
typedef struct {
  int max_seq_len, vocab_size, padded_vocab_size, num_layers, num_heads, channels;
} GPT2Config;

#define NUM_PARAMETER_TENSORS 16
typedef struct {
  float *wte, *wpe, *ln1w, *ln1b, *qkvw, *qkvb, *attprojw, *attprojb, *ln2w, *ln2b,
      *fcw, *fcb, *fcprojw, *fcprojb, *lnfw, *lnfb;
} ParameterTensors;

static void fill_in_parameter_sizes(std::size_t *s, GPT2Config c) {
  int Vp = c.padded_vocab_size, C = c.channels, maxT = c.max_seq_len, L = c.num_layers;
  s[0] = (std::size_t)Vp * C;
  s[1] = (std::size_t)maxT * C;
  s[2] = s[3] = (std::size_t)L * C;
  s[4] = (std::size_t)L * 3 * C * C;
  s[5] = (std::size_t)L * 3 * C;
  s[6] = (std::size_t)L * C * C;
  s[7] = s[8] = s[9] = (std::size_t)L * C;
  s[10] = (std::size_t)L * 4 * C * C;
  s[11] = (std::size_t)L * 4 * C;
  s[12] = (std::size_t)L * C * 4 * C;
  s[13] = (std::size_t)L * C;
  s[14] = s[15] = (std::size_t)C;
}

static void point_params(ParameterTensors *p, const std::size_t *sz, float *mem) {
  float **q[] = {&p->wte, &p->wpe, &p->ln1w, &p->ln1b, &p->qkvw, &p->qkvb,
                 &p->attprojw, &p->attprojb, &p->ln2w, &p->ln2b, &p->fcw, &p->fcb,
                 &p->fcprojw, &p->fcprojb, &p->lnfw, &p->lnfb};
  for (int i = 0; i < NUM_PARAMETER_TENSORS; ++i) {
    *q[i] = mem;
    mem += sz[i];
  }
}

#define NUM_ACTIVATION_TENSORS 21
typedef struct {
  float *encoded, *ln1, *ln1_mean, *ln1_rstd, *atty, *att, *attproj, *residual2, *ln2,
      *ln2_mean, *ln2_rstd, *fch, *fch_gelu, *fcproj, *residual3, *lnf, *lnf_mean,
      *lnf_rstd, *losses, *qkvr, *output;
} ActivationTensors;

static void fill_in_activation_sizes(std::size_t *a, int B, int T, GPT2Config c) {
  std::size_t Vp = c.padded_vocab_size, L = c.num_layers, NH = c.num_heads, C = c.channels;
  a[0] = (std::size_t)B * T * C;
  a[1] = L * B * T * C;
  a[2] = a[3] = L * B * T;
  a[4] = L * B * T * C;
  a[5] = L * B * NH * T * T;
  a[6] = a[7] = a[8] = L * B * T * C;
  a[9] = a[10] = L * B * T;
  a[11] = a[12] = L * B * T * 4 * C;
  a[13] = a[14] = L * B * T * C;
  a[15] = (std::size_t)B * T * C;
  a[16] = a[17] = a[18] = (std::size_t)B * T;
  a[19] = L * B * T * 3 * C;
  std::size_t m1 = 3 * C, m2 = NH * (std::size_t)T;
  if (m2 > m1) m1 = m2;
  if (Vp > m1) m1 = Vp;
  a[20] = (std::size_t)B * T * m1;
}

typedef struct {
  float *bt4c, *preatt, *residual3;
} GradActTensors;

static void fill_in_grad_act_sizes(std::size_t *a, int B, int T, GPT2Config c) {
  a[0] = (std::size_t)B * T * 4 * c.channels;
  a[1] = (std::size_t)B * c.num_heads * T * T;
  a[2] = (std::size_t)B * T * c.channels;
}

typedef struct {
  GPT2Config config;
  ParameterTensors params;
  std::size_t param_sizes[NUM_PARAMETER_TENSORS];
  float *params_memory = nullptr;
  std::size_t num_parameters = 0;
  ParameterTensors grads;
  float *grads_memory = nullptr;
  ActivationTensors acts;
  std::size_t act_sizes[NUM_ACTIVATION_TENSORS];
  float *acts_memory = nullptr;
  std::size_t num_activations = 0;
  GradActTensors grads_acts;
  std::size_t num_grad_acts = 0;
  float *grads_acts_memory = nullptr;
  int batch_size = 0, seq_len = 0;
  int *inputs = nullptr, *targets = nullptr;
  float mean_loss = -1.0f;
  float *cpu_losses = nullptr;
} GPT2;

static float *dev_alloc_float(std::size_t n) {
  return sycl::malloc_device<float>(n, Q());
}

static void gpt2_build_from_checkpoint(GPT2 *model, const char *path) {
  FILE *f = xfopen(path, "rb");
  int hdr[256];
  xfread(hdr, sizeof(int), 256, f);
  if (hdr[0] != 20240326 || hdr[1] != 3) {
    std::fprintf(stderr, "bad checkpoint\n");
    std::exit(EXIT_FAILURE);
  }
  model->config.max_seq_len = hdr[2];
  model->config.vocab_size = hdr[3];
  model->config.num_layers = hdr[4];
  model->config.num_heads = hdr[5];
  model->config.channels = hdr[6];
  model->config.padded_vocab_size = hdr[7];
  fill_in_parameter_sizes(model->param_sizes, model->config);
  std::size_t np = 0;
  for (int i = 0; i < NUM_PARAMETER_TENSORS; ++i) np += model->param_sizes[i];
  model->num_parameters = np;
  float *cpu = (float *)std::malloc(np * sizeof(float));
  if (!cpu) std::exit(EXIT_FAILURE);
  xfread(cpu, sizeof(float), np, f);
  std::fclose(f);
  model->params_memory = dev_alloc_float(np);
  Q().memcpy(model->params_memory, cpu, np * sizeof(float)).wait();
  std::free(cpu);
  point_params(&model->params, model->param_sizes, model->params_memory);
  model->grads_memory = nullptr;
  model->acts_memory = nullptr;
  model->grads_acts_memory = nullptr;
  model->inputs = nullptr;
  model->targets = nullptr;
  model->cpu_losses = nullptr;
  model->batch_size = model->seq_len = 0;
  model->mean_loss = -1.0f;
}

static void gpt2_forward(GPT2 *model, int *h_inputs, int *h_targets, int B, int T) {
  int V = model->config.vocab_size, Vp = model->config.padded_vocab_size,
      L = model->config.num_layers, NH = model->config.num_heads, C = model->config.channels;
  if (!model->params_memory) {
    std::fprintf(stderr, "model not init\n");
    std::exit(EXIT_FAILURE);
  }
  if (!model->acts_memory) {
    model->batch_size = B;
    model->seq_len = T;
    fill_in_activation_sizes(model->act_sizes, B, T, model->config);
    std::size_t na = 0;
    for (int i = 0; i < NUM_ACTIVATION_TENSORS; ++i) na += model->act_sizes[i];
    model->num_activations = na;
    model->acts_memory = dev_alloc_float(na);
    float **q[] = {&model->acts.encoded, &model->acts.ln1, &model->acts.ln1_mean,
                   &model->acts.ln1_rstd, &model->acts.atty, &model->acts.att,
                   &model->acts.attproj, &model->acts.residual2, &model->acts.ln2,
                   &model->acts.ln2_mean, &model->acts.ln2_rstd, &model->acts.fch,
                   &model->acts.fch_gelu, &model->acts.fcproj, &model->acts.residual3,
                   &model->acts.lnf, &model->acts.lnf_mean, &model->acts.lnf_rstd,
                   &model->acts.losses, &model->acts.qkvr, &model->acts.output};
    float *it = model->acts_memory;
    for (int i = 0; i < NUM_ACTIVATION_TENSORS; ++i) {
      *q[i] = it;
      it += model->act_sizes[i];
    }
    model->inputs = sycl::malloc_device<int>((std::size_t)B * T, Q());
    model->targets = sycl::malloc_device<int>((std::size_t)B * T, Q());
    model->cpu_losses = (float *)std::malloc((std::size_t)B * T * sizeof(float));
  } else {
    if (B != model->batch_size || T != model->seq_len) {
      std::fprintf(stderr, "B/T mismatch\n");
      std::exit(EXIT_FAILURE);
    }
  }
  Q().memcpy(model->inputs, h_inputs, (std::size_t)B * T * sizeof(int)).wait();
  if (h_targets) Q().memcpy(model->targets, h_targets, (std::size_t)B * T * sizeof(int)).wait();

  ParameterTensors params = model->params;
  ActivationTensors acts = model->acts;
  encoder_forward(acts.encoded, model->inputs, params.wte, params.wpe, B, T, C);
  Q().wait();
  for (int l = 0; l < L; ++l) {
    float *residual = l == 0 ? acts.encoded : acts.residual3 + (std::size_t)(l - 1) * B * T * C;
    float *l_ln1w = params.ln1w + (std::size_t)l * C;
    float *l_ln1b = params.ln1b + (std::size_t)l * C;
    float *l_qkvw = params.qkvw + (std::size_t)l * 3 * C * C;
    float *l_qkvb = params.qkvb + (std::size_t)l * 3 * C;
    float *l_attprojw = params.attprojw + (std::size_t)l * C * C;
    float *l_attprojb = params.attprojb + (std::size_t)l * C;
    float *l_ln2w = params.ln2w + (std::size_t)l * C;
    float *l_ln2b = params.ln2b + (std::size_t)l * C;
    float *l_fcw = params.fcw + (std::size_t)l * 4 * C * C;
    float *l_fcb = params.fcb + (std::size_t)l * 4 * C;
    float *l_fcprojw = params.fcprojw + (std::size_t)l * C * 4 * C;
    float *l_fcprojb = params.fcprojb + (std::size_t)l * C;
    float *l_ln1 = acts.ln1 + (std::size_t)l * B * T * C;
    float *l_ln1_mean = acts.ln1_mean + (std::size_t)l * B * T;
    float *l_ln1_rstd = acts.ln1_rstd + (std::size_t)l * B * T;
    float *l_qkvr = acts.qkvr + (std::size_t)l * B * T * 3 * C;
    float *l_atty = acts.atty + (std::size_t)l * B * T * C;
    float *l_att = acts.att + (std::size_t)l * B * NH * T * T;
    float *l_attproj = acts.attproj + (std::size_t)l * B * T * C;
    float *l_residual2 = acts.residual2 + (std::size_t)l * B * T * C;
    float *l_ln2 = acts.ln2 + (std::size_t)l * B * T * C;
    float *l_ln2_mean = acts.ln2_mean + (std::size_t)l * B * T;
    float *l_ln2_rstd = acts.ln2_rstd + (std::size_t)l * B * T;
    float *l_fch = acts.fch + (std::size_t)l * B * T * 4 * C;
    float *l_fch_gelu = acts.fch_gelu + (std::size_t)l * B * T * 4 * C;
    float *l_fcproj = acts.fcproj + (std::size_t)l * B * T * C;
    float *l_residual3 = acts.residual3 + (std::size_t)l * B * T * C;
    float *scratch = acts.output;
    layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C);
    Q().wait();
    matmul_forward(scratch, l_ln1, l_qkvw, l_qkvb, B, T, C, 3 * C);
    attention_forward(l_atty, l_qkvr, l_att, scratch, B, T, C, NH);
    Q().wait();
    matmul_forward(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C);
    residual_forward(l_residual2, residual, l_attproj, (std::size_t)B * T * C);
    Q().wait();
    layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C);
    Q().wait();
    matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4 * C);
    gelu_forward(l_fch_gelu, l_fch, (std::size_t)B * T * 4 * C);
    Q().wait();
    matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4 * C, C);
    residual_forward(l_residual3, l_residual2, l_fcproj, (std::size_t)B * T * C);
    Q().wait();
  }
  float *residual = acts.residual3 + (std::size_t)(L - 1) * B * T * C;
  layernorm_forward(acts.lnf, acts.lnf_mean, acts.lnf_rstd, residual, params.lnfw, params.lnfb,
                    B, T, C);
  Q().wait();
  matmul_forward(acts.output, acts.lnf, params.wte, nullptr, B, T, C, Vp);
  if (h_targets) {
    fused_classifier(acts.output, acts.losses, model->targets, B, T, V, Vp);
    Q().wait();
    Q().memcpy(model->cpu_losses, acts.losses, (std::size_t)B * T * sizeof(float)).wait();
    double s = 0.0;
    for (int i = 0; i < B * T; ++i) s += (double)model->cpu_losses[i];
    model->mean_loss = (float)(s / (B * T));
  } else {
    model->mean_loss = -1.0f;
  }
}

static void gpt2_zero_grad(GPT2 *model) {
  if (model->grads_acts_memory)
    Q().memset(model->grads_acts_memory, 0, model->num_grad_acts * sizeof(float)).wait();
  if (model->grads_memory)
    Q().memset(model->grads_memory, 0, model->num_parameters * sizeof(float)).wait();
}

static void gpt2_backward(GPT2 *model) {
  if (model->mean_loss == -1.0f) {
    std::fprintf(stderr, "must forward with targets\n");
    std::exit(EXIT_FAILURE);
  }
  if (!model->grads_memory) {
    model->grads_memory = dev_alloc_float(model->num_parameters);
    point_params(&model->grads, model->param_sizes, model->grads_memory);
    GPT2Config cfg = model->config;
    cfg.num_layers = 1;
    std::size_t bw[3];
    fill_in_grad_act_sizes(bw, model->batch_size, model->seq_len, cfg);
    model->num_grad_acts = bw[0] + bw[1] + bw[2];
    model->grads_acts_memory = dev_alloc_float(model->num_grad_acts);
    model->grads_acts.bt4c = model->grads_acts_memory;
    model->grads_acts.preatt = model->grads_acts_memory + bw[0];
    model->grads_acts.residual3 = model->grads_acts_memory + bw[0] + bw[1];
    gpt2_zero_grad(model);
  }
  int B = model->batch_size, T = model->seq_len, Vp = model->config.padded_vocab_size,
      L = model->config.num_layers, NH = model->config.num_heads, C = model->config.channels;
  ParameterTensors params = model->params, grads = model->grads;
  ActivationTensors acts = model->acts;
  GradActTensors ga = model->grads_acts;
  matmul_backward(ga.bt4c, grads.wte, nullptr, acts.output, acts.lnf, params.wte, B, T, C, Vp);
  float *residual = acts.residual3 + (std::size_t)(L - 1) * B * T * C;
  float *dresidual = ga.residual3;
  layernorm_backward(dresidual, grads.lnfw, grads.lnfb, ga.bt4c, residual, params.lnfw,
                     acts.lnf_mean, acts.lnf_rstd, B, T, C);
  Q().wait();
  for (int l = L - 1; l >= 0; --l) {
    residual = l == 0 ? acts.encoded : acts.residual3 + (std::size_t)(l - 1) * B * T * C;
    float *l_qkvw = params.qkvw + (std::size_t)l * 3 * C * C;
    float *l_attprojw = params.attprojw + (std::size_t)l * C * C;
    float *l_ln2w = params.ln2w + (std::size_t)l * C;
    float *l_fcw = params.fcw + (std::size_t)l * 4 * C * C;
    float *l_fcprojw = params.fcprojw + (std::size_t)l * C * 4 * C;
    float *dl_ln1w = grads.ln1w + (std::size_t)l * C;
    float *dl_ln1b = grads.ln1b + (std::size_t)l * C;
    float *dl_qkvw = grads.qkvw + (std::size_t)l * 3 * C * C;
    float *dl_qkvb = grads.qkvb + (std::size_t)l * 3 * C;
    float *dl_attprojw = grads.attprojw + (std::size_t)l * C * C;
    float *dl_attprojb = grads.attprojb + (std::size_t)l * C;
    float *dl_ln2w = grads.ln2w + (std::size_t)l * C;
    float *dl_ln2b = grads.ln2b + (std::size_t)l * C;
    float *dl_fcw = grads.fcw + (std::size_t)l * 4 * C * C;
    float *dl_fcb = grads.fcb + (std::size_t)l * 4 * C;
    float *dl_fcprojw = grads.fcprojw + (std::size_t)l * C * 4 * C;
    float *dl_fcprojb = grads.fcprojb + (std::size_t)l * C;
    float *l_ln1 = acts.ln1 + (std::size_t)l * B * T * C;
    float *l_ln1_mean = acts.ln1_mean + (std::size_t)l * B * T;
    float *l_ln1_rstd = acts.ln1_rstd + (std::size_t)l * B * T;
    float *l_qkvr = acts.qkvr + (std::size_t)l * B * T * 3 * C;
    float *l_atty = acts.atty + (std::size_t)l * B * T * C;
    float *l_att = acts.att + (std::size_t)l * B * NH * T * T;
    float *l_residual2 = acts.residual2 + (std::size_t)l * B * T * C;
    float *l_ln2 = acts.ln2 + (std::size_t)l * B * T * C;
    float *l_ln2_mean = acts.ln2_mean + (std::size_t)l * B * T;
    float *l_ln2_rstd = acts.ln2_rstd + (std::size_t)l * B * T;
    float *l_fch = acts.fch + (std::size_t)l * B * T * 4 * C;
    float *l_fch_gelu = acts.fch_gelu + (std::size_t)l * B * T * 4 * C;
    float *dl_btc = acts.lnf;
    float *dl_bt4c = ga.bt4c;
    float *dl_preatt = ga.preatt;
    float *scratch = acts.output;
    float *l_ln1w = params.ln1w + (std::size_t)l * C;
    matmul_backward(dl_bt4c, dl_fcprojw, dl_fcprojb, dresidual, l_fch_gelu, l_fcprojw, B, T,
                    4 * C, C);
    gelu_backward(dl_bt4c, l_fch, dl_bt4c, (std::size_t)B * T * 4 * C);
    Q().wait();
    matmul_backward(dl_btc, dl_fcw, dl_fcb, dl_bt4c, l_ln2, l_fcw, B, T, C, 4 * C);
    layernorm_backward(dresidual, dl_ln2w, dl_ln2b, dl_btc, l_residual2, l_ln2w, l_ln2_mean,
                       l_ln2_rstd, B, T, C);
    Q().wait();
    matmul_backward(dl_btc, dl_attprojw, dl_attprojb, dresidual, l_atty, l_attprojw, B, T, C,
                    C);
    float *buffer_a = l_atty;
    float *buffer_b = l_fch;
    attention_backward(dl_bt4c, buffer_b, dl_preatt, scratch, buffer_a, dl_btc, l_qkvr, l_att,
                       B, T, C, NH);
    Q().wait();
    matmul_backward(dl_btc, dl_qkvw, dl_qkvb, dl_bt4c, l_ln1, l_qkvw, B, T, C, 3 * C);
    layernorm_backward(dresidual, dl_ln1w, dl_ln1b, dl_btc, residual, l_ln1w, l_ln1_mean,
                       l_ln1_rstd, B, T, C);
    Q().wait();
  }
  encoder_backward(grads.wte, grads.wpe, dresidual, model->inputs, B, T, C);
  Q().wait();
}

static void gpt2_free(GPT2 *model) {
  if (model->params_memory) sycl::free(model->params_memory, Q());
  if (model->grads_memory) sycl::free(model->grads_memory, Q());
  if (model->acts_memory) sycl::free(model->acts_memory, Q());
  if (model->grads_acts_memory) sycl::free(model->grads_acts_memory, Q());
  if (model->inputs) sycl::free(model->inputs, Q());
  if (model->targets) sycl::free(model->targets, Q());
  if (model->cpu_losses) std::free(model->cpu_losses);
}

int main(int argc, char **argv) {
  if (argc != 4) {
    std::fprintf(stderr, "usage: %s <debug_state.bin> <output> <gpt2_124M.bin>\n", argv[0]);
    return 1;
  }
  sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order{}};
  g_q = &q;
  GPT2 model{};
  gpt2_build_from_checkpoint(&model, argv[3]);
  FILE *s = xfopen(argv[1], "rb");
  int sh[256];
  xfread(sh, sizeof(int), 256, s);
  int B = sh[2], T = sh[3];
  int *x = (int *)std::malloc((std::size_t)B * T * sizeof(int));
  int *y = (int *)std::malloc((std::size_t)B * T * sizeof(int));
  xfread(x, sizeof(int), (std::size_t)B * T, s);
  xfread(y, sizeof(int), (std::size_t)B * T, s);
  std::fclose(s);
  gpt2_forward(&model, x, y, B, T);
  gpt2_zero_grad(&model);
  gpt2_backward(&model);
  std::size_t P = model.num_parameters;
  float *grads = (float *)std::malloc(P * sizeof(float));
  Q().memcpy(grads, model.grads_memory, P * sizeof(float)).wait();
  FILE *o = xfopen(argv[2], "wb");
  std::fwrite(grads, sizeof(float), P, o);
  std::fwrite(&model.mean_loss, sizeof(float), 1, o);
  std::fclose(o);
  std::free(x);
  std::free(y);
  std::free(grads);
  gpt2_free(&model);
  return 0;
}
