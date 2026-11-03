#include <sycl/sycl.hpp>
#include <oneapi/math.hpp>
#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <cmath>

template <typename T>
static T ceil_div(T dividend, T divisor) {
  return (dividend + divisor - 1) / divisor;
}

// ---------------------------------------------------------------- k1
static void attention_forward1(sycl::queue &q, float *out, float *preatt,
                               float *att, const float *inp, int B, int T,
                               int C, int NH) {
  {
    std::size_t total = (std::size_t)B * (std::size_t)NH * (std::size_t)T * (std::size_t)T;
    q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> id) {
       std::size_t idx = id[0];
       int t2 = (int)(idx % (std::size_t)T);
       int t = (int)((idx / (std::size_t)T) % (std::size_t)T);
       if (t2 > t) {
         preatt[idx] = -INFINITY;
         return;
       }
       int h = (int)((idx / ((std::size_t)T * (std::size_t)T)) % (std::size_t)NH);
       int b = (int)(idx / ((std::size_t)NH * (std::size_t)T * (std::size_t)T));
       int C3 = C * 3;
       int hs = C / NH;
       const float *query_t = inp + (std::size_t)b * T * C3 + (std::size_t)t * C3 + (std::size_t)h * hs;
       const float *key_t2 = inp + (std::size_t)b * T * C3 + (std::size_t)t2 * C3 + (std::size_t)h * hs + C;
       float val = 0.0f;
       for (int i = 0; i < hs; ++i) val += query_t[i] * key_t2[i];
       val *= 1.0f / sycl::sqrt((float)hs);
       preatt[idx] = val;
     }).wait();
  }
  {
    std::size_t total = (std::size_t)B * (std::size_t)T * (std::size_t)NH;
    q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> id) {
       std::size_t idx = id[0];
       int h = (int)(idx % (std::size_t)NH);
       int t = (int)((idx / (std::size_t)NH) % (std::size_t)T);
       int b = (int)(idx / ((std::size_t)NH * (std::size_t)T));
       const float *pre = preatt + (std::size_t)b * NH * T * T + (std::size_t)h * T * T + (std::size_t)t * T;
       float *at = att + (std::size_t)b * NH * T * T + (std::size_t)h * T * T + (std::size_t)t * T;
       float maxval = -FLT_MAX;
       for (int t2 = 0; t2 <= t; ++t2) maxval = sycl::fmax(maxval, pre[t2]);
       float expsum = 0.0f;
       for (int t2 = 0; t2 <= t; ++t2) {
         float e = sycl::exp(pre[t2] - maxval);
         expsum += e;
         at[t2] = e;
       }
       float inv = (expsum == 0.0f) ? 0.0f : 1.0f / expsum;
       for (int t2 = 0; t2 < T; ++t2) {
         if (t2 <= t) at[t2] *= inv;
         else at[t2] = 0.0f;
       }
     }).wait();
  }
  {
    std::size_t total = (std::size_t)B * (std::size_t)T * (std::size_t)NH;
    q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> id) {
       std::size_t idx = id[0];
       int h = (int)(idx % (std::size_t)NH);
       int t = (int)((idx / (std::size_t)NH) % (std::size_t)T);
       int b = (int)(idx / ((std::size_t)NH * (std::size_t)T));
       int C3 = C * 3;
       int hs = C / NH;
       float *out_bth = out + (std::size_t)b * T * C + (std::size_t)t * C + (std::size_t)h * hs;
       const float *att_bth = att + (std::size_t)b * NH * T * T + (std::size_t)h * T * T + (std::size_t)t * T;
       for (int i = 0; i < hs; ++i) out_bth[i] = 0.0f;
       for (int t2 = 0; t2 <= t; ++t2) {
         const float *v = inp + (std::size_t)b * T * C3 + (std::size_t)t2 * C3 + (std::size_t)h * hs + (std::size_t)C * 2;
         float a = att_bth[t2];
         for (int i = 0; i < hs; ++i) out_bth[i] += a * v[i];
       }
     }).wait();
  }
}

// ---------------------------------------------------------------- k2 flash
static void attention_forward2(sycl::queue &q, float *out, const float *inp,
                               int B, int T, int C, int NH) {
  const int N = T;
  const int nh = NH;
  const int d = C / NH;
  const int Bc = 32;
  const int Br = 32;
  const int Tc = (N + Bc - 1) / Bc;
  const int Tr = (N + Br - 1) / Br;
  const float softmax_scale = 1.0f / sycl::sqrt((float)d);
  std::size_t lm_n = (std::size_t)B * nh * N;
  std::size_t qkv_n = (std::size_t)B * T * C;
  float *l = sycl::malloc_device<float>(lm_n, q);
  float *m = sycl::malloc_device<float>(lm_n, q);
  float *qq = sycl::malloc_device<float>(qkv_n, q);
  float *kk = sycl::malloc_device<float>(qkv_n, q);
  float *vv = sycl::malloc_device<float>(qkv_n, q);
  float *tmpO = sycl::malloc_device<float>(qkv_n, q);
  q.fill(l, 0.0f, lm_n).wait();
  q.fill(m, -10000.0f, lm_n).wait();
  q.fill(tmpO, 0.0f, qkv_n).wait();
  // permute k2
  {
    std::size_t total = (std::size_t)B * nh * N * d;
    q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> id) {
       std::size_t idx = id[0];
       std::size_t denom = (std::size_t)nh * N * d;
       int b = (int)(idx / denom);
       std::size_t rest = idx % denom;
       int nh_ = (int)(rest / ((std::size_t)N * d));
       rest = rest % ((std::size_t)N * d);
       int n = (int)(rest / (std::size_t)d);
       int d_ = (int)(rest % (std::size_t)d);
       std::size_t inp_idx = (std::size_t)b * N * 3 * nh * d + (std::size_t)n * 3 * nh * d + (std::size_t)nh_ * d + (std::size_t)d_;
       qq[idx] = inp[inp_idx];
       kk[idx] = inp[inp_idx + (std::size_t)nh * d];
       vv[idx] = inp[inp_idx + (std::size_t)2 * nh * d];
     }).wait();
  }
  // flash kernel k2
  {
    int num_groups = B * nh;
    std::size_t sram_floats = (std::size_t)Br * d + (std::size_t)2 * Bc * d + (std::size_t)Bc * Br;
    q.submit([&](sycl::handler &h) {
       sycl::local_accessor<float, 1> sram(sycl::range<1>(sram_floats), h);
       h.parallel_for(sycl::nd_range<1>(sycl::range<1>((std::size_t)num_groups * Br),
                                        sycl::range<1>((std::size_t)Br)),
                      [=](sycl::nd_item<1> it) {
                        int tx = (int)it.get_local_id(0);
                        int g = (int)it.get_group(0);
                        int bx = g / nh;
                        int by = g % nh;
                        std::size_t qkv_offset = (std::size_t)bx * nh * N * d + (std::size_t)by * N * d;
                        std::size_t lm_offset = (std::size_t)bx * nh * N + (std::size_t)by * N;
                        auto mp = sram.get_multi_ptr<sycl::access::decorated::no>();
                        float *Qi = mp.get();
                        float *Kj = Qi + (std::size_t)Br * d;
                        float *Vj = Kj + (std::size_t)Bc * d;
                        float *S = Vj + (std::size_t)Bc * d;
                        for (int j = 0; j < Tc; ++j) {
                          for (int x = 0; x < d; ++x) {
                            Kj[tx * d + x] = kk[qkv_offset + (std::size_t)j * Bc * d + (std::size_t)tx * d + x];
                            Vj[tx * d + x] = vv[qkv_offset + (std::size_t)j * Bc * d + (std::size_t)tx * d + x];
                          }
                          sycl::group_barrier(it.get_group());
                          for (int i = 0; i < Tr; ++i) {
                            if (i * Br + tx >= N) break;
                            for (int x = 0; x < d; ++x)
                              Qi[tx * d + x] = qq[qkv_offset + (std::size_t)i * Br * d + (std::size_t)tx * d + x];
                            float row_m_prev = m[lm_offset + (std::size_t)i * Br + tx];
                            float row_l_prev = l[lm_offset + (std::size_t)i * Br + tx];
                            float row_m = -INFINITY;
                            for (int y = 0; y < Bc; ++y) {
                              if (j * Bc + y >= N) break;
                              float sum = 0.0f;
                              for (int x = 0; x < d; ++x) sum += Qi[tx * d + x] * Kj[y * d + x];
                              sum *= softmax_scale;
                              if (i * Br + tx < j * Bc + y) sum = -INFINITY;
                              S[tx * Bc + y] = sum;
                              if (sum > row_m) row_m = sum;
                            }
                            float row_l = 0.0f;
                            for (int y = 0; y < Bc; ++y) {
                              if (j * Bc + y >= N) break;
                              if (i * Br + tx < j * Bc + y) S[tx * Bc + y] = 0.0f;
                              else S[tx * Bc + y] = sycl::exp(S[tx * Bc + y] - row_m);
                              row_l += S[tx * Bc + y];
                            }
                            float row_m_new = (row_m_prev > row_m) ? row_m_prev : row_m;
                            float row_l_new = sycl::exp(row_m_prev - row_m_new) * row_l_prev +
                                              sycl::exp(row_m - row_m_new) * row_l;
                            for (int x = 0; x < d; ++x) {
                              float pv = 0.0f;
                              for (int y = 0; y < Bc; ++y) {
                                if (j * Bc + y >= N) break;
                                pv += S[tx * Bc + y] * Vj[y * d + x];
                              }
                              std::size_t o_idx = qkv_offset + (std::size_t)i * Br * d + (std::size_t)tx * d + x;
                              float o_old = tmpO[o_idx];
                              tmpO[o_idx] = (1.0f / row_l_new) *
                                            ((row_l_prev * sycl::exp(row_m_prev - row_m_new) * o_old) +
                                             (sycl::exp(row_m - row_m_new) * pv));
                            }
                            m[lm_offset + (std::size_t)i * Br + tx] = row_m_new;
                            l[lm_offset + (std::size_t)i * Br + tx] = row_l_new;
                          }
                          it.barrier(sycl::access::fence_space::local_space);
                        }
                      });
     }).wait();
  }
  // unpermute k2
  {
    std::size_t total = (std::size_t)B * nh * N * d;
    q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> id) {
       std::size_t idx = id[0];
       std::size_t denom = (std::size_t)nh * N * d;
       int b = (int)(idx / denom);
       std::size_t rest = idx % denom;
       int nh_ = (int)(rest / ((std::size_t)N * d));
       rest = rest % ((std::size_t)N * d);
       int n = (int)(rest / (std::size_t)d);
       int d_ = (int)(rest % (std::size_t)d);
       std::size_t other_idx = (std::size_t)b * nh * N * d + (std::size_t)n * nh * d + (std::size_t)nh_ * d + (std::size_t)d_;
       out[other_idx] = tmpO[idx];
     }).wait();
  }
  sycl::free(l, q);
  sycl::free(m, q);
  sycl::free(qq, q);
  sycl::free(kk, q);
  sycl::free(vv, q);
  sycl::free(tmpO, q);
}

// helpers for oneMath column-major sgemm strided batch
static void sgemm_colmajor_batch(sycl::queue &q, bool transA, bool transB,
                                 std::int64_t Mm, std::int64_t Nn, std::int64_t Kk,
                                 const float *A, std::int64_t lda, std::int64_t sA,
                                 const float *Bmat, std::int64_t ldb, std::int64_t sB,
                                 float *Cmat, std::int64_t ldc, std::int64_t sC,
                                 std::int64_t batch) {
  using namespace oneapi::math::blas;
  oneapi::math::transpose ta = transA ? oneapi::math::transpose::trans : oneapi::math::transpose::nontrans;
  oneapi::math::transpose tb = transB ? oneapi::math::transpose::trans : oneapi::math::transpose::nontrans;
  column_major::gemm_batch(q, ta, tb, Mm, Nn, Kk, 1.0f, A, lda, sA, Bmat, ldb, sB, 0.0f,
                           Cmat, ldc, sC, batch);
  q.wait();
}

// ---------------------------------------------------------------- k3
static void attention_forward3(sycl::queue &q, float *out, float *vaccum, float *qkvr,
                               float *preatt, float *att, const float *inp,
                               int B, int T, int C, int NH) {
  int HS = C / NH;
  float *qq = qkvr + (std::size_t)0 * B * T * C;
  float *kk = qkvr + (std::size_t)1 * B * T * C;
  float *vv = qkvr + (std::size_t)2 * B * T * C;
  {
    std::size_t total = (std::size_t)B * NH * T * HS;
    q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> id) {
       std::size_t idx = id[0];
       std::size_t denom = (std::size_t)NH * T * HS;
       int b = (int)(idx / denom);
       std::size_t rest = idx % denom;
       int nh_ = (int)(rest / ((std::size_t)T * HS));
       rest = rest % ((std::size_t)T * HS);
       int n = (int)(rest / (std::size_t)HS);
       int d_ = (int)(rest % (std::size_t)HS);
       std::size_t inp_idx = (std::size_t)b * T * 3 * NH * HS + (std::size_t)n * 3 * NH * HS +
                             (std::size_t)nh_ * HS + (std::size_t)d_;
       qq[idx] = inp[inp_idx];
       kk[idx] = inp[inp_idx + (std::size_t)NH * HS];
       vv[idx] = inp[inp_idx + (std::size_t)2 * NH * HS];
     }).wait();
  }
  sgemm_colmajor_batch(q, true, false, T, T, HS, kk, HS, (std::int64_t)T * HS, qq, HS,
                       (std::int64_t)T * HS, preatt, T, (std::int64_t)T * T, (std::int64_t)B * NH);
  {
    float scale = 1.0f / sycl::sqrt((float)HS);
    std::size_t total = (std::size_t)B * NH * T * T;
    q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> id) {
       std::size_t idx = id[0];
       std::size_t rest = idx % ((std::size_t)NH * T * T);
       rest = rest % ((std::size_t)T * T);
       int t2 = (int)(rest / (std::size_t)T);
       int t = (int)(rest % (std::size_t)T);
       if (t > t2) preatt[idx] = -INFINITY;
       else preatt[idx] *= scale;
     }).wait();
  }
  {
    std::int64_t rows = (std::int64_t)B * NH * T;
    std::int64_t cols = T;
    q.parallel_for(sycl::range<1>((std::size_t)rows), [=](sycl::id<1> id) {
       std::size_t r = id[0];
       const float *x = preatt + r * cols;
       float *y = att + r * cols;
       float maxv = -INFINITY;
       for (std::int64_t i = 0; i < cols; ++i) maxv = sycl::fmax(maxv, x[i]);
       float sum = 0.0f;
       for (std::int64_t i = 0; i < cols; ++i) {
         float e = sycl::exp(x[i] - maxv);
         y[i] = e;
         sum += e;
       }
       float inv = (sum == 0.0f) ? 0.0f : 1.0f / sum;
       for (std::int64_t i = 0; i < cols; ++i) y[i] *= inv;
     }).wait();
  }
  sgemm_colmajor_batch(q, false, false, HS, T, T, vv, HS, (std::int64_t)T * HS, att, T,
                       (std::int64_t)T * T, vaccum, HS, (std::int64_t)T * HS, (std::int64_t)B * NH);
  {
    std::size_t total = (std::size_t)B * T * C;
    q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> id) {
       std::size_t idx = id[0];
       (void)idx;
       // decode as (b, nh_, n, d_) linear over permuted layout then scatter
       // idx here is over permuted (B,NH,T,HS); map to output (B,T,NH,HS)
       std::size_t denom = (std::size_t)NH * T * HS;
       int b = (int)(idx / denom);
       std::size_t rest = idx % denom;
       int nh_ = (int)(rest / ((std::size_t)T * HS));
       rest = rest % ((std::size_t)T * HS);
       int n = (int)(rest / (std::size_t)HS);
       int d_ = (int)(rest % (std::size_t)HS);
       std::size_t other_idx =
           (std::size_t)b * NH * T * HS + (std::size_t)n * NH * HS + (std::size_t)nh_ * HS + d_;
       out[other_idx] = vaccum[idx];
     }).wait();
  }
}

// ---------------------------------------------------------------- k4 online softmax
static void attention_forward4(sycl::queue &q, float *out, float *vaccum, float *qkvr,
                               float *preatt, float *att, const float *inp,
                               int B, int T, int C, int NH) {
  int HS = C / NH;
  float *qq = qkvr + (std::size_t)0 * B * T * C;
  float *kk = qkvr + (std::size_t)1 * B * T * C;
  float *vv = qkvr + (std::size_t)2 * B * T * C;
  {
    std::size_t total = (std::size_t)B * NH * T * HS;
    q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> id) {
       std::size_t idx = id[0];
       std::size_t denom = (std::size_t)NH * T * HS;
       int b = (int)(idx / denom);
       std::size_t rest = idx % denom;
       int nh_ = (int)(rest / ((std::size_t)T * HS));
       rest = rest % ((std::size_t)T * HS);
       int n = (int)(rest / (std::size_t)HS);
       int d_ = (int)(rest % (std::size_t)HS);
       std::size_t inp_idx = (std::size_t)b * T * 3 * NH * HS + (std::size_t)n * 3 * NH * HS +
                             (std::size_t)nh_ * HS + (std::size_t)d_;
       qq[idx] = inp[inp_idx];
       kk[idx] = inp[inp_idx + (std::size_t)NH * HS];
       vv[idx] = inp[inp_idx + (std::size_t)2 * NH * HS];
     }).wait();
  }
  sgemm_colmajor_batch(q, true, false, T, T, HS, kk, HS, (std::int64_t)T * HS, qq, HS,
                       (std::int64_t)T * HS, preatt, T, (std::int64_t)T * T, (std::int64_t)B * NH);
  {
    float scale = 1.0f / sycl::sqrt((float)HS);
    std::int64_t rows = (std::int64_t)B * NH * T;
    std::int64_t cols = T;
    q.parallel_for(sycl::range<1>((std::size_t)rows), [=](sycl::id<1> id) {
       std::size_t r = id[0];
       int own_pos = (int)(r % (std::size_t)T);
       const float *x = preatt + r * cols;
       float *y = att + r * cols;
       float maxv = -FLT_MAX;
       for (int i = 0; i <= own_pos; ++i) maxv = sycl::fmax(maxv, x[i]);
       float sum = 0.0f;
       for (int i = 0; i <= own_pos; ++i) sum += sycl::exp(scale * (x[i] - maxv));
       float norm = (sum == 0.0f) ? 0.0f : 1.0f / sum;
       for (int i = 0; i < cols; ++i) {
         if (i <= own_pos) y[i] = sycl::exp(scale * (x[i] - maxv)) * norm;
         else y[i] = 0.0f;
       }
     }).wait();
  }
  sgemm_colmajor_batch(q, false, false, HS, T, T, vv, HS, (std::int64_t)T * HS, att, T,
                       (std::int64_t)T * T, vaccum, HS, (std::int64_t)T * HS, (std::int64_t)B * NH);
  {
    std::size_t total = (std::size_t)B * T * C;
    q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> id) {
       std::size_t idx = id[0];
       std::size_t denom = (std::size_t)NH * T * HS;
       int b = (int)(idx / denom);
       std::size_t rest = idx % denom;
       int nh_ = (int)(rest / ((std::size_t)T * HS));
       rest = rest % ((std::size_t)T * HS);
       int n = (int)(rest / (std::size_t)HS);
       int d_ = (int)(rest % (std::size_t)HS);
       std::size_t other_idx =
           (std::size_t)b * NH * T * HS + (std::size_t)n * NH * HS + (std::size_t)nh_ * HS + d_;
       out[other_idx] = vaccum[idx];
     }).wait();
  }
}

// k5 fp32 build identical math, own kernels
static void attention_forward5(sycl::queue &q, float *out, float *vaccum, float *qkvr,
                               float *preatt, float *att, const float *inp,
                               int B, int T, int C, int NH) {
  int HS = C / NH;
  float *qq = qkvr + (std::size_t)0 * B * T * C;
  float *kk = qkvr + (std::size_t)1 * B * T * C;
  float *vv = qkvr + (std::size_t)2 * B * T * C;
  {
    std::size_t total = (std::size_t)B * NH * T * HS;
    q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> id) {
       std::size_t idx = id[0];
       std::size_t denom = (std::size_t)NH * T * HS;
       int b = (int)(idx / denom);
       std::size_t rest = idx % denom;
       int nh_ = (int)(rest / ((std::size_t)T * HS));
       rest = rest % ((std::size_t)T * HS);
       int n = (int)(rest / (std::size_t)HS);
       int d_ = (int)(rest % (std::size_t)HS);
       std::size_t inp_idx = (std::size_t)b * T * 3 * NH * HS + (std::size_t)n * 3 * NH * HS +
                             (std::size_t)nh_ * HS + (std::size_t)d_;
       qq[idx] = inp[inp_idx];
       kk[idx] = inp[inp_idx + (std::size_t)NH * HS];
       vv[idx] = inp[inp_idx + (std::size_t)2 * NH * HS];
     }).wait();
  }
  sgemm_colmajor_batch(q, true, false, T, T, HS, kk, HS, (std::int64_t)T * HS, qq, HS,
                       (std::int64_t)T * HS, preatt, T, (std::int64_t)T * T, (std::int64_t)B * NH);
  {
    float scale = 1.0f / sycl::sqrt((float)HS);
    std::int64_t rows = (std::int64_t)B * NH * T;
    std::int64_t cols = T;
    q.parallel_for(sycl::range<1>((std::size_t)rows), [=](sycl::id<1> id) {
       std::size_t r = id[0];
       int own_pos = (int)(r % (std::size_t)T);
       const float *x = preatt + r * cols;
       float *y = att + r * cols;
       float maxv = -FLT_MAX;
       for (int i = 0; i <= own_pos; ++i) maxv = sycl::fmax(maxv, x[i]);
       float sum = 0.0f;
       for (int i = 0; i <= own_pos; ++i) sum += sycl::exp(scale * (x[i] - maxv));
       float norm = (sum == 0.0f) ? 0.0f : 1.0f / sum;
       for (int i = 0; i < cols; ++i) {
         if (i <= own_pos) y[i] = sycl::exp(scale * (x[i] - maxv)) * norm;
         else y[i] = 0.0f;
       }
     }).wait();
  }
  sgemm_colmajor_batch(q, false, false, HS, T, T, vv, HS, (std::int64_t)T * HS, att, T,
                       (std::int64_t)T * T, vaccum, HS, (std::int64_t)T * HS, (std::int64_t)B * NH);
  {
    std::size_t total = (std::size_t)B * T * C;
    q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> id) {
       std::size_t idx = id[0];
       std::size_t denom = (std::size_t)NH * T * HS;
       int b = (int)(idx / denom);
       std::size_t rest = idx % denom;
       int nh_ = (int)(rest / ((std::size_t)T * HS));
       rest = rest % ((std::size_t)T * HS);
       int n = (int)(rest / (std::size_t)HS);
       int d_ = (int)(rest % (std::size_t)HS);
       std::size_t other_idx =
           (std::size_t)b * NH * T * HS + (std::size_t)n * NH * HS + (std::size_t)nh_ * HS + d_;
       out[other_idx] = vaccum[idx];
     }).wait();
  }
}

// k6 same math, own kernels (first-run validation forces permute)
static void attention_forward6(sycl::queue &q, float *out, float *vaccum, float *qkvr,
                               float *preatt, float *att, const float *inp,
                               int B, int T, int C, int NH) {
  int HS = C / NH;
  float *qq = qkvr + (std::size_t)0 * B * T * C;
  float *kk = qkvr + (std::size_t)1 * B * T * C;
  float *vv = qkvr + (std::size_t)2 * B * T * C;
  {
    std::size_t total = (std::size_t)B * NH * T * HS;
    q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> id) {
       std::size_t idx = id[0];
       std::size_t denom = (std::size_t)NH * T * HS;
       int b = (int)(idx / denom);
       std::size_t rest = idx % denom;
       int nh_ = (int)(rest / ((std::size_t)T * HS));
       rest = rest % ((std::size_t)T * HS);
       int n = (int)(rest / (std::size_t)HS);
       int d_ = (int)(rest % (std::size_t)HS);
       std::size_t inp_idx = (std::size_t)b * T * 3 * NH * HS + (std::size_t)n * 3 * NH * HS +
                             (std::size_t)nh_ * HS + (std::size_t)d_;
       qq[idx] = inp[inp_idx];
       kk[idx] = inp[inp_idx + (std::size_t)NH * HS];
       vv[idx] = inp[inp_idx + (std::size_t)2 * NH * HS];
     }).wait();
  }
  sgemm_colmajor_batch(q, true, false, T, T, HS, kk, HS, (std::int64_t)T * HS, qq, HS,
                       (std::int64_t)T * HS, preatt, T, (std::int64_t)T * T, (std::int64_t)B * NH);
  {
    float scale = 1.0f / sycl::sqrt((float)HS);
    std::int64_t rows = (std::int64_t)B * NH * T;
    std::int64_t cols = T;
    q.parallel_for(sycl::range<1>((std::size_t)rows), [=](sycl::id<1> id) {
       std::size_t r = id[0];
       int own_pos = (int)(r % (std::size_t)T);
       const float *x = preatt + r * cols;
       float *y = att + r * cols;
       float maxv = -FLT_MAX;
       for (int i = 0; i <= own_pos; ++i) maxv = sycl::fmax(maxv, x[i]);
       float sum = 0.0f;
       for (int i = 0; i <= own_pos; ++i) sum += sycl::exp(scale * (x[i] - maxv));
       float norm = (sum == 0.0f) ? 0.0f : 1.0f / sum;
       for (int i = 0; i < cols; ++i) {
         if (i <= own_pos) y[i] = sycl::exp(scale * (x[i] - maxv)) * norm;
         else y[i] = 0.0f;
       }
     }).wait();
  }
  sgemm_colmajor_batch(q, false, false, HS, T, T, vv, HS, (std::int64_t)T * HS, att, T,
                       (std::int64_t)T * T, vaccum, HS, (std::int64_t)T * HS, (std::int64_t)B * NH);
  {
    std::size_t total = (std::size_t)B * T * C;
    q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> id) {
       std::size_t idx = id[0];
       std::size_t denom = (std::size_t)NH * T * HS;
       int b = (int)(idx / denom);
       std::size_t rest = idx % denom;
       int nh_ = (int)(rest / ((std::size_t)T * HS));
       rest = rest % ((std::size_t)T * HS);
       int n = (int)(rest / (std::size_t)HS);
       int d_ = (int)(rest % (std::size_t)HS);
       std::size_t other_idx =
           (std::size_t)b * NH * T * HS + (std::size_t)n * NH * HS + (std::size_t)nh_ * HS + d_;
       out[other_idx] = vaccum[idx];
     }).wait();
  }
}

static void attention_forward_dispatch(int kernel_num, sycl::queue &q, float *out,
                                       float *vaccum, float *qkvr, float *preatt,
                                       float *att, const float *inp, int B, int T,
                                       int C, int NH) {
  switch (kernel_num) {
  case 1: attention_forward1(q, out, preatt, att, inp, B, T, C, NH); break;
  case 2: attention_forward2(q, out, inp, B, T, C, NH); break;
  case 3: attention_forward3(q, out, vaccum, qkvr, preatt, att, inp, B, T, C, NH); break;
  case 4: attention_forward4(q, out, vaccum, qkvr, preatt, att, inp, B, T, C, NH); break;
  case 5: attention_forward5(q, out, vaccum, qkvr, preatt, att, inp, B, T, C, NH); break;
  case 6: attention_forward6(q, out, vaccum, qkvr, preatt, att, inp, B, T, C, NH); break;
  default: std::fprintf(stderr, "Invalid kernel number\n"); std::exit(1);
  }
}

int main(int argc, char **argv) {
  if (argc != 7) {
    std::fprintf(stderr, "usage: %s <input> <output> B T C NH\n", argv[0]);
    return 2;
  }
  const int B = std::atoi(argv[3]);
  const int T = std::atoi(argv[4]);
  const int C = std::atoi(argv[5]);
  const int NH = std::atoi(argv[6]);
  FILE *fin = std::fopen(argv[1], "rb");
  if (!fin) { std::perror(argv[1]); return 2; }
  std::size_t inp_n = (std::size_t)B * T * 3 * C;
  float *inp = (float *)std::malloc(inp_n * sizeof(float));
  if (std::fread(inp, sizeof(float), inp_n, fin) != inp_n) {
    std::fprintf(stderr, "input too short at inp\n");
    return 2;
  }
  std::fclose(fin);
  FILE *fout = std::fopen(argv[2], "wb");
  if (!fout) { std::perror(argv[2]); return 2; }

  sycl::queue q;
  float *d_inp = sycl::malloc_device<float>(inp_n, q);
  std::size_t out_n = (std::size_t)B * T * C;
  float *d_out = sycl::malloc_device<float>(out_n, q);
  float *d_stats = sycl::malloc_device<float>((std::size_t)B * NH * T, q);
  float *d_vaccum = sycl::malloc_device<float>((std::size_t)B * T * C, q);
  float *d_qkvr = sycl::malloc_device<float>((std::size_t)B * T * 3 * C, q);
  float *d_preatt = sycl::malloc_device<float>((std::size_t)B * NH * T * T, q);
  float *d_att = sycl::malloc_device<float>((std::size_t)B * NH * T * T, q);
  float *out = (float *)std::malloc(out_n * sizeof(float));
  q.memcpy(d_inp, inp, inp_n * sizeof(float)).wait();
  const int variants[6] = {1, 2, 3, 4, 5, 6};
  for (int vi = 0; vi < 6; ++vi) {
    const int k = variants[vi];
    q.fill(d_out, 0.0f, out_n).wait();
    attention_forward_dispatch(k, q, d_out, d_vaccum, d_qkvr, d_preatt, d_att, d_inp, B, T,
                               C, NH);
    q.wait();
    q.memcpy(out, d_out, out_n * sizeof(float)).wait();
    std::fwrite(out, sizeof(float), out_n, fout);
  }
  std::fclose(fout);
  sycl::free(d_inp, q);
  sycl::free(d_out, q);
  sycl::free(d_stats, q);
  sycl::free(d_vaccum, q);
  sycl::free(d_qkvr, q);
  sycl::free(d_preatt, q);
  sycl::free(d_att, q);
  std::free(inp);
  std::free(out);
  return 0;
}
