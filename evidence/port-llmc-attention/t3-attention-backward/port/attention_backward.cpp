#include <sycl/sycl.hpp>
#include <oneapi/math.hpp>

#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <cmath>
#include <cstdint>

template <typename T>
inline T ceil_div(T a, T b) {
  return (a + b - 1) / b;
}

// ----------------------------------------------------------------------------
// CPU reference forward (host only, no OpenMP dependency)
void attention_forward_cpu(float *out, float *preatt, float *att, float *inp,
                           int B, int T, int C, int NH) {
  int C3 = C * 3;
  int hs = C / NH;
  float scale = 1.0f / sqrtf((float)hs);
  for (int b = 0; b < B; b++) {
    for (int t = 0; t < T; t++) {
      for (int h = 0; h < NH; h++) {
        float *query_t = inp + b * T * C3 + t * C3 + h * hs;
        float *preatt_bth = preatt + b * NH * T * T + h * T * T + t * T;
        float *att_bth = att + b * NH * T * T + h * T * T + t * T;
        float maxval = -FLT_MAX;
        for (int t2 = 0; t2 < T; t2++) {
          float *key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C;
          float val = 0.0f;
          for (int i = 0; i < hs; i++) val += query_t[i] * key_t2[i];
          if (val > maxval) maxval = val;
          preatt_bth[t2] = val;
        }
        float expsum = 0.0f;
        for (int t2 = 0; t2 <= t; t2++) {
          float expv = expf(scale * (preatt_bth[t2] - maxval));
          expsum += expv;
          att_bth[t2] = expv;
        }
        float inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;
        for (int t2 = 0; t2 < T; t2++) {
          if (t2 <= t) att_bth[t2] *= inv;
          else att_bth[t2] = 0.0f;
        }
        float *out_bth = out + b * T * C + t * C + h * hs;
        for (int i = 0; i < hs; i++) out_bth[i] = 0.0f;
        for (int t2 = 0; t2 <= t; t2++) {
          float *value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C * 2;
          float a = att_bth[t2];
          for (int i = 0; i < hs; i++) out_bth[i] += a * value_t2[i];
        }
      }
    }
  }
}

// ----------------------------------------------------------------------------
// Device helpers: permute / unpermute / softmax forward (all 1-D launches)

static void dev_permute_forward(sycl::queue &q, float *qq, float *kk, float *vv,
                                const float *inp, int B, int N, int NH, int d) {
  size_t total = (size_t)B * (size_t)NH * (size_t)N * (size_t)d;
  size_t wg = 256;
  size_t global = ceil_div(total, wg) * wg;
  q.parallel_for(sycl::nd_range<1>(global, wg), [=](sycl::nd_item<1> it) {
     size_t idx = it.get_global_id(0);
     if (idx >= total) return;
     size_t tmp = idx;
     size_t d_ = tmp % (size_t)d; tmp /= (size_t)d;
     size_t n = tmp % (size_t)N; tmp /= (size_t)N;
     size_t nh_ = tmp % (size_t)NH; tmp /= (size_t)NH;
     size_t b = tmp;
     size_t inp_idx = (b * (size_t)N * 3 * (size_t)NH * (size_t)d) +
                      (n * 3 * (size_t)NH * (size_t)d) + ((size_t)nh_ * (size_t)d) + d_;
     qq[idx] = inp[inp_idx];
     kk[idx] = inp[inp_idx + (size_t)NH * (size_t)d];
     vv[idx] = inp[inp_idx + 2 * (size_t)NH * (size_t)d];
   }).wait();
}

static void dev_permute_backward(sycl::queue &q, float *dinp, const float *dq,
                                 const float *dk, const float *dv, int B, int N,
                                 int NH, int d) {
  size_t total = (size_t)B * (size_t)NH * (size_t)N * (size_t)d;
  size_t wg = 256;
  size_t global = ceil_div(total, wg) * wg;
  q.parallel_for(sycl::nd_range<1>(global, wg), [=](sycl::nd_item<1> it) {
     size_t idx = it.get_global_id(0);
     if (idx >= total) return;
     size_t tmp = idx;
     size_t d_ = tmp % (size_t)d; tmp /= (size_t)d;
     size_t n = tmp % (size_t)N; tmp /= (size_t)N;
     size_t nh_ = tmp % (size_t)NH; tmp /= (size_t)NH;
     size_t b = tmp;
     size_t inp_idx = (b * (size_t)N * 3 * (size_t)NH * (size_t)d) +
                      (n * 3 * (size_t)NH * (size_t)d) + ((size_t)nh_ * (size_t)d) + d_;
     dinp[inp_idx] += dq[idx];
     dinp[inp_idx + (size_t)NH * (size_t)d] += dk[idx];
     dinp[inp_idx + 2 * (size_t)NH * (size_t)d] += dv[idx];
   }).wait();
}

static void dev_unpermute_forward(sycl::queue &q, const float *inp, float *out,
                                  int B, int N, int NH, int d) {
  size_t total = (size_t)B * (size_t)NH * (size_t)N * (size_t)d;
  size_t wg = 256;
  size_t global = ceil_div(total, wg) * wg;
  q.parallel_for(sycl::nd_range<1>(global, wg), [=](sycl::nd_item<1> it) {
     size_t idx = it.get_global_id(0);
     if (idx >= total) return;
     size_t tmp = idx;
     size_t d_ = tmp % (size_t)d; tmp /= (size_t)d;
     size_t n = tmp % (size_t)N; tmp /= (size_t)N;
     size_t nh_ = tmp % (size_t)NH; tmp /= (size_t)NH;
     size_t b = tmp;
     size_t other_idx = (b * (size_t)NH * (size_t)N * (size_t)d) +
                        (n * (size_t)NH * (size_t)d) + ((size_t)nh_ * (size_t)d) + d_;
     out[other_idx] = inp[idx];
   }).wait();
}

static void dev_unpermute_backward(sycl::queue &q, float *dinp, const float *dout,
                                   int B, int N, int NH, int d) {
  size_t total = (size_t)B * (size_t)NH * (size_t)N * (size_t)d;
  size_t wg = 256;
  size_t global = ceil_div(total, wg) * wg;
  q.parallel_for(sycl::nd_range<1>(global, wg), [=](sycl::nd_item<1> it) {
     size_t idx = it.get_global_id(0);
     if (idx >= total) return;
     size_t tmp = idx;
     size_t d_ = tmp % (size_t)d; tmp /= (size_t)d;
     size_t n = tmp % (size_t)N; tmp /= (size_t)N;
     size_t nh_ = tmp % (size_t)NH; tmp /= (size_t)NH;
     size_t b = tmp;
     size_t other_idx = (b * (size_t)NH * (size_t)N * (size_t)d) +
                        (n * (size_t)NH * (size_t)d) + ((size_t)nh_ * (size_t)d) + d_;
     dinp[idx] += dout[other_idx];
   }).wait();
}

// softmax forward: one work-item per row, scalar two-pass, causal
static void dev_softmax_forward(sycl::queue &q, float *out, float inv_temp,
                                const float *inp, int N, int T) {
  size_t rows = (size_t)N * (size_t)T;
  size_t wg = 256;
  size_t global = ceil_div(rows, wg) * wg;
  q.parallel_for(sycl::nd_range<1>(global, wg), [=](sycl::nd_item<1> it) {
     size_t idx = it.get_global_id(0);
     if (idx >= rows) return;
     int own = (int)(idx % (size_t)T);
     const float *x = inp + idx * (size_t)T;
     float *y = out + idx * (size_t)T;
     float maxv = -FLT_MAX;
     for (int i = 0; i <= own; i++) maxv = sycl::fmax(maxv, x[i]);
     float sum = 0.0f;
     for (int i = 0; i <= own; i++)
       sum += sycl::exp(inv_temp * (x[i] - maxv));
     float inv = 1.0f / sum;
     for (int i = 0; i <= own; i++)
       y[i] = sycl::exp(inv_temp * (x[i] - maxv)) * inv;
   }).wait();
}

// ----------------------------------------------------------------------------
// oneMath column-major strided batched GEMM (cuBLAS semantics, swapped operands)
static void col_gemm_batch(sycl::queue &q, oneapi::math::transpose ta,
                           oneapi::math::transpose tb, int64_t m, int64_t n,
                           int64_t k, float alpha, const float *A, int64_t lda,
                           int64_t strideA, const float *B, int64_t ldb,
                           int64_t strideB, float beta, float *C, int64_t ldc,
                           int64_t strideC, int64_t batch) {
  oneapi::math::blas::column_major::gemm_batch(
      q, ta, tb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc,
      strideC, batch)
      .wait();
}

// ----------------------------------------------------------------------------
// Softmax backward variants 1..8 : each is its own SYCL kernel (1-D launch)

// k1: parallel over t3 only, serial over batch/head
static void launch_softmax_bwd1(sycl::queue &q, float *dpreatt, const float *datt,
                                const float *att, int B, int T, int C, int NH) {
  size_t wg = 256;
  size_t global = ceil_div((size_t)T, wg) * wg;
  int hs = C / NH;
  float scale = 1.0f / sycl::sqrt((float)hs);
  q.parallel_for(sycl::nd_range<1>(global, wg), [=](sycl::nd_item<1> it) {
     size_t t3 = it.get_global_id(0);
     if ((int)t3 >= T) return;
     for (int b = 0; b < B; b++) {
       for (int h = 0; h < NH; h++) {
         for (int t = (int)t3; t < T; t++) {
           const float *ab = att + (size_t)b * NH * T * T + (size_t)h * T * T + (size_t)t * T;
           const float *db = datt + (size_t)b * NH * T * T + (size_t)h * T * T + (size_t)t * T;
           float *gb = dpreatt + (size_t)b * NH * T * T + (size_t)h * T * T + (size_t)t * T;
           float acc = 0.0f;
           float a3 = ab[t3];
           for (int t2 = 0; t2 <= t; t2++) {
             float ind = (t2 == (int)t3) ? 1.0f : 0.0f;
             acc += scale * ab[t2] * (ind - a3) * db[t2];
           }
           gb[t3] = acc;
         }
       }
     }
   }).wait();
}

// k2: parallel over (batch*head, t3) flattened
static void launch_softmax_bwd2(sycl::queue &q, float *dpreatt, const float *datt,
                                const float *att, int B, int T, int C, int NH) {
  size_t total = (size_t)B * (size_t)NH * (size_t)T;
  size_t wg = 256;
  size_t global = ceil_div(total, wg) * wg;
  int hs = C / NH;
  float scale = 1.0f / sycl::sqrt((float)hs);
  q.parallel_for(sycl::nd_range<1>(global, wg), [=](sycl::nd_item<1> it) {
     size_t gid = it.get_global_id(0);
     if (gid >= total) return;
     int t3 = (int)(gid % (size_t)T);
     size_t bh = gid / (size_t)T;
     for (int t = t3; t < T; t++) {
       const float *ab = att + bh * (size_t)T * (size_t)T + (size_t)t * T;
       const float *db = datt + bh * (size_t)T * (size_t)T + (size_t)t * T;
       float *gb = dpreatt + bh * (size_t)T * (size_t)T + (size_t)t * T;
       float acc = 0.0f;
       for (int t2 = 0; t2 <= t; t2++) {
         float ind = (t2 == t3) ? 1.0f : 0.0f;
         acc += scale * ab[t2] * (ind - ab[t3]) * db[t2];
       }
       gb[t3] = acc;
     }
   }).wait();
}

// k3: same decomposition as k2 but hoists att[t3] out of inner loop
static void launch_softmax_bwd3(sycl::queue &q, float *dpreatt, const float *datt,
                                const float *att, int B, int T, int C, int NH) {
  size_t total = (size_t)B * (size_t)NH * (size_t)T;
  size_t wg = 256;
  size_t global = ceil_div(total, wg) * wg;
  int hs = C / NH;
  float scale = 1.0f / sycl::sqrt((float)hs);
  q.parallel_for(sycl::nd_range<1>(global, wg), [=](sycl::nd_item<1> it) {
     size_t gid = it.get_global_id(0);
     if (gid >= total) return;
     int t3 = (int)(gid % (size_t)T);
     size_t bh = gid / (size_t)T;
     for (int t = t3; t < T; t++) {
       const float *ab = att + bh * (size_t)T * (size_t)T + (size_t)t * T;
       const float *db = datt + bh * (size_t)T * (size_t)T + (size_t)t * T;
       float *gb = dpreatt + bh * (size_t)T * (size_t)T + (size_t)t * T;
       float a3 = ab[t3];
       float acc = 0.0f;
       for (int t2 = 0; t2 <= t; t2++) {
         float ind = (t2 == t3) ? 1.0f : 0.0f;
         acc += ab[t2] * (ind - a3) * db[t2];
       }
       gb[t3] = scale * acc;
     }
   }).wait();
}

// k4: each work-item handles up to 8 consecutive t3 (UNROLL), naive inner loop
static void launch_softmax_bwd4(sycl::queue &q, float *dpreatt, const float *datt,
                                const float *att, int B, int T, int C, int NH) {
  constexpr int UNROLL = 8;
  size_t n8 = ((size_t)T + UNROLL - 1) / UNROLL;
  size_t total = (size_t)B * (size_t)NH * n8;
  size_t wg = 256;
  size_t global = ceil_div(total, wg) * wg;
  int hs = C / NH;
  float scale = 1.0f / sycl::sqrt((float)hs);
  q.parallel_for(sycl::nd_range<1>(global, wg), [=](sycl::nd_item<1> it) {
     size_t gid = it.get_global_id(0);
     if (gid >= total) return;
     size_t bh = gid / n8;
     int base = (int)((gid % n8) * UNROLL);
     for (int t = base; t < T; t++) {
       const float *ab = att + bh * (size_t)T * (size_t)T + (size_t)t * T;
       const float *db = datt + bh * (size_t)T * (size_t)T + (size_t)t * T;
       float *gb = dpreatt + bh * (size_t)T * (size_t)T + (size_t)t * T;
       float a3[UNROLL];
       for (int k = 0; k < UNROLL; k++) {
         int c = base + k;
         if (t < c || c >= T) continue;
         a3[k] = ab[c];
       }
       float res[UNROLL] = {};
       for (int t2 = 0; t2 <= t; t2++) {
         float at2 = ab[t2];
         float dt2 = db[t2];
         for (int k = 0; k < UNROLL; k++) {
           int c = base + k;
           if (t < c || c >= T) continue;
           float ind = (t2 == c) ? 1.0f : 0.0f;
           res[k] += at2 * (ind - a3[k]) * dt2;
         }
       }
       for (int k = 0; k < UNROLL; k++) {
         int c = base + k;
         if (t < c || c >= T) continue;
         gb[c] = scale * res[k];
       }
     }
   }).wait();
}

// k5: same 8-wide grouping but split inner loop to avoid branch in hot path
static void launch_softmax_bwd5(sycl::queue &q, float *dpreatt, const float *datt,
                                const float *att, int B, int T, int C, int NH) {
  constexpr int UNROLL = 8;
  size_t n8 = ((size_t)T + UNROLL - 1) / UNROLL;
  size_t total = (size_t)B * (size_t)NH * n8;
  size_t wg = 256;
  size_t global = ceil_div(total, wg) * wg;
  int hs = C / NH;
  float scale = 1.0f / sycl::sqrt((float)hs);
  q.parallel_for(sycl::nd_range<1>(global, wg), [=](sycl::nd_item<1> it) {
     size_t gid = it.get_global_id(0);
     if (gid >= total) return;
     size_t bh = gid / n8;
     int base = (int)((gid % n8) * UNROLL);
     for (int t = base; t < T; t++) {
       const float *ab = att + bh * (size_t)T * (size_t)T + (size_t)t * T;
       const float *db = datt + bh * (size_t)T * (size_t)T + (size_t)t * T;
       float *gb = dpreatt + bh * (size_t)T * (size_t)T + (size_t)t * T;
       float a3[UNROLL];
       for (int k = 0; k < UNROLL; k++) {
         int c = base + k;
         int cc = c > t ? t : c;
         cc = cc < 0 ? 0 : cc;
         if (cc >= T) cc = T - 1;
         a3[k] = ab[cc];
       }
       float res[UNROLL] = {};
       // part 1: t2 < base
       for (int t2 = 0; t2 < base && t2 <= t; t2++) {
         float p = ab[t2] * db[t2];
         for (int k = 0; k < UNROLL; k++) res[k] -= p * a3[k];
       }
       // part 2: the single t2 that may equal one of base..base+7
       for (int t2 = base; t2 < base + UNROLL && t2 <= t; t2++) {
         float p = ab[t2] * db[t2];
         for (int k = 0; k < UNROLL; k++) {
           float ind = (t2 == base + k) ? 1.0f : 0.0f;
           res[k] += p * (ind - a3[k]);
         }
       }
       // part 3: remainder
       for (int t2 = base + UNROLL; t2 <= t; t2++) {
         float p = ab[t2] * db[t2];
         for (int k = 0; k < UNROLL; k++) res[k] -= p * a3[k];
       }
       for (int k = 0; k < UNROLL; k++) {
         int c = base + k;
         if (c >= T || t < c) continue;
         gb[c] = scale * res[k];
       }
     }
   }).wait();
}

// k6: work-group per (batch*head, t) with local-memory tiling
template <int BlockSize>
static void launch_softmax_bwd6_impl(sycl::queue &q, float *dpreatt,
                                     const float *datt, const float *att, int B,
                                     int T, int C, int NH) {
  size_t groups = (size_t)B * (size_t)NH * (size_t)T;
  size_t global = groups * (size_t)BlockSize;
  int hs = C / NH;
  float scale = 1.0f / sqrtf((float)hs);
  q.submit([&](sycl::handler &h) {
     sycl::local_accessor<float, 1> tile(sycl::range<1>(BlockSize), h);
     h.parallel_for(sycl::nd_range<1>(global, BlockSize),
                    [=](sycl::nd_item<1> it) {
       size_t g = it.get_group(0);
       size_t lid = it.get_local_id(0);
       int bh = (int)(g / (size_t)T);
       int t = (int)(g % (size_t)T);
       const float *ab = att + (size_t)bh * T * T + (size_t)t * T;
       const float *db = datt + (size_t)bh * T * T + (size_t)t * T;
       float *gb = dpreatt + (size_t)bh * T * T + (size_t)t * T;
       int steps = (t + 1 + BlockSize - 1) / BlockSize;
       for (int f = 0; f < steps; f++) {
         int t3 = f * BlockSize + (int)lid;
         float at3 = 0.0f;
         if (t3 < T) at3 = ab[t3];
         float acc = 0.0f;
         for (int b2 = 0; b2 <= t; b2 += BlockSize) {
           int end = t + 1 - b2;
           if (end > BlockSize) end = BlockSize;
           sycl::group_barrier(it.get_group(), sycl::memory_scope::work_group);
           {
             int t2 = b2 + (int)lid;
             if (t2 > t) t2 = t;
             tile[(size_t)lid] = ab[t2] * db[t2];
           }
           sycl::group_barrier(it.get_group(), sycl::memory_scope::work_group);
           if (f * BlockSize == b2) {
             for (int i = 0; i < end; i++) {
               int t2 = b2 + i;
               float ind = (t2 == t3) ? 1.0f : 0.0f;
               acc += tile[(size_t)i] * (ind - at3);
             }
           } else {
             for (int i = 0; i < end; i++) acc += tile[(size_t)i] * (0.0f - at3);
           }
           it.barrier(sycl::access::fence_space::local_space);
         }
         if (t3 < T) gb[t3] = scale * acc;
       }
     });
   }).wait();
}

static void launch_softmax_bwd6(sycl::queue &q, float *dpreatt, const float *datt,
                                const float *att, int B, int T, int C, int NH,
                                int block_size) {
  switch (block_size) {
    case 32: launch_softmax_bwd6_impl<32>(q, dpreatt, datt, att, B, T, C, NH); break;
    case 64: launch_softmax_bwd6_impl<64>(q, dpreatt, datt, att, B, T, C, NH); break;
    case 128: launch_softmax_bwd6_impl<128>(q, dpreatt, datt, att, B, T, C, NH); break;
    case 256: launch_softmax_bwd6_impl<256>(q, dpreatt, datt, att, B, T, C, NH); break;
    case 512: launch_softmax_bwd6_impl<512>(q, dpreatt, datt, att, B, T, C, NH); break;
    case 1024: launch_softmax_bwd6_impl<1024>(q, dpreatt, datt, att, B, T, C, NH); break;
    default: launch_softmax_bwd6_impl<256>(q, dpreatt, datt, att, B, T, C, NH); break;
  }
}

// k7: work-group per row, whole-group reduction, simplified math
template <int BlockSize>
static void launch_softmax_bwd7_core(sycl::queue &q, float *dpreatt,
                                     const float *datt, const float *att, int B,
                                     int T, int NH, float scale) {
  size_t groups = (size_t)B * (size_t)NH * (size_t)T;
  size_t global = groups * (size_t)BlockSize;
  q.parallel_for(sycl::nd_range<1>(global, BlockSize),
                 [=](sycl::nd_item<1> it) {
     size_t g = it.get_group(0);
     size_t lid = it.get_local_id(0);
     int bh = (int)(g / (size_t)T);
     int t = (int)(g % (size_t)T);
     const float *ab = att + (size_t)bh * T * T + (size_t)t * T;
     const float *db = datt + (size_t)bh * T * T + (size_t)t * T;
     float *gb = dpreatt + (size_t)bh * T * T + (size_t)t * T;
     float local = 0.0f;
     for (int t2 = (int)lid; t2 <= t; t2 += BlockSize)
       local += ab[t2] * db[t2];
     float total = sycl::reduce_over_group(it.get_group(), local, sycl::plus<float>());
     for (int t3 = (int)lid; t3 <= t; t3 += BlockSize) {
       float acc = ab[t3] * (db[t3] - total);
       gb[t3] = scale * acc;
     }
   }).wait();
}

static void launch_softmax_bwd7(sycl::queue &q, float *dpreatt, const float *datt,
                                const float *att, int B, int T, int C, int NH,
                                int block_size) {
  int hs = C / NH;
  float scale = 1.0f / sqrtf((float)hs);
  switch (block_size) {
    case 32: launch_softmax_bwd7_core<32>(q, dpreatt, datt, att, B, T, NH, scale); break;
    case 64: launch_softmax_bwd7_core<64>(q, dpreatt, datt, att, B, T, NH, scale); break;
    case 128: launch_softmax_bwd7_core<128>(q, dpreatt, datt, att, B, T, NH, scale); break;
    case 256: launch_softmax_bwd7_core<256>(q, dpreatt, datt, att, B, T, NH, scale); break;
    case 512: launch_softmax_bwd7_core<512>(q, dpreatt, datt, att, B, T, NH, scale); break;
    case 1024: launch_softmax_bwd7_core<1024>(q, dpreatt, datt, att, B, T, NH, scale); break;
    default: launch_softmax_bwd7_core<256>(q, dpreatt, datt, att, B, T, NH, scale); break;
  }
}

// k8: work-group handles up to 4 rows in reverse order, whole-group reduction
template <int BlockSize>
static void launch_softmax_bwd8_core(sycl::queue &q, float *dpreatt,
                                     const float *datt, const float *att, int B,
                                     int T, int NH, float scale) {
  constexpr int Tper = 4;
  size_t nblocks = ((size_t)T + Tper - 1) / Tper;
  size_t groups = (size_t)B * (size_t)NH * nblocks;
  size_t global = groups * (size_t)BlockSize;
  q.parallel_for(sycl::nd_range<1>(global, BlockSize),
                 [=](sycl::nd_item<1> it) {
     size_t g = it.get_group(0);
     size_t lid = it.get_local_id(0);
     size_t bh = g / nblocks;
     size_t tb = g % nblocks;
     int t0 = T - 1 - Tper * (int)tb;
     for (int o = 0; o < Tper; o++) {
       int t = t0 - o;
       if (t < 0) break;
       const float *ab = att + bh * (size_t)T * (size_t)T + (size_t)t * T;
       const float *db = datt + bh * (size_t)T * (size_t)T + (size_t)t * T;
       float *gb = dpreatt + bh * (size_t)T * (size_t)T + (size_t)t * T;
       float local = 0.0f;
       for (int t2 = (int)lid; t2 <= t; t2 += BlockSize)
         local += ab[t2] * db[t2];
       float total = sycl::reduce_over_group(it.get_group(), local, sycl::plus<float>());
       for (int t3 = (int)lid; t3 <= t; t3 += BlockSize) {
         float acc = ab[t3] * (db[t3] - total);
         gb[t3] = scale * acc;
       }
     }
   }).wait();
}

static void launch_softmax_bwd8(sycl::queue &q, float *dpreatt, const float *datt,
                                const float *att, int B, int T, int C, int NH,
                                int block_size) {
  int hs = C / NH;
  float scale = 1.0f / sqrtf((float)hs);
  switch (block_size) {
    case 32: launch_softmax_bwd8_core<32>(q, dpreatt, datt, att, B, T, NH, scale); break;
    case 64: launch_softmax_bwd8_core<64>(q, dpreatt, datt, att, B, T, NH, scale); break;
    case 128: launch_softmax_bwd8_core<128>(q, dpreatt, datt, att, B, T, NH, scale); break;
    case 256: launch_softmax_bwd8_core<256>(q, dpreatt, datt, att, B, T, NH, scale); break;
    case 512: launch_softmax_bwd8_core<512>(q, dpreatt, datt, att, B, T, NH, scale); break;
    case 1024: launch_softmax_bwd8_core<1024>(q, dpreatt, datt, att, B, T, NH, scale); break;
    default: launch_softmax_bwd8_core<256>(q, dpreatt, datt, att, B, T, NH, scale); break;
  }
}

// ----------------------------------------------------------------------------
// Forward and backward pipelines

static void attention_forward(sycl::queue &q, float *out, float *vaccum, float *qkvr,
                              float *preatt, float *att, const float *inp, int B,
                              int T, int C, int NH, int block_size) {
  (void)block_size;
  int HS = C / NH;
  float *qq = qkvr + 0 * (size_t)B * T * C;
  float *kk = qkvr + 1 * (size_t)B * T * C;
  float *vv = qkvr + 2 * (size_t)B * T * C;
  dev_permute_forward(q, qq, kk, vv, inp, B, T, NH, HS);
  // preatt = K^T * Q equivalent via column-major GEMM
  col_gemm_batch(q, oneapi::math::transpose::trans, oneapi::math::transpose::nontrans,
                 T, T, HS, 1.0f, kk, HS, (int64_t)T * HS, qq, HS,
                 (int64_t)T * HS, 0.0f, preatt, T, (int64_t)T * T, (int64_t)B * NH);
  float scale = 1.0f / sqrtf((float)HS);
  dev_softmax_forward(q, att, scale, preatt, B * NH, T);
  // vaccum = V * Att? column-major: (HS x T) = (HS x T)*(T x T)? vaccum col = HS x T
  col_gemm_batch(q, oneapi::math::transpose::nontrans, oneapi::math::transpose::nontrans,
                 HS, T, T, 1.0f, vv, HS, (int64_t)T * HS, att, T,
                 (int64_t)T * T, 0.0f, vaccum, HS, (int64_t)T * HS, (int64_t)B * NH);
  dev_unpermute_forward(q, vaccum, out, B, T, NH, HS);
}

using SoftmaxBwdFn = void (*)(sycl::queue &, float *, const float *, const float *,
                              int, int, int, int, int);

static void attention_backward_impl(sycl::queue &q, float *dinp, float *dqkvr,
                                    float *dpreatt, float *datt, float *dvaccum,
                                    const float *dout, const float *inp,
                                    const float *qkvr, const float *preatt,
                                    const float *att, const float *vaccum, int B,
                                    int T, int C, int NH, int variant,
                                    int block_size) {
  (void)inp; (void)preatt; (void)vaccum;
  int HS = C / NH;
  const float alpha = 1.0f;
  const float beta = 1.0f;
  const float *qq = qkvr + 0 * (size_t)B * T * C;
  const float *kk = qkvr + 1 * (size_t)B * T * C;
  const float *vv = qkvr + 2 * (size_t)B * T * C;
  float *dq = dqkvr + 0 * (size_t)B * T * C;
  float *dk = dqkvr + 1 * (size_t)B * T * C;
  float *dv = dqkvr + 2 * (size_t)B * T * C;

  dev_unpermute_backward(q, dvaccum, dout, B, T, NH, HS);

  col_gemm_batch(q, oneapi::math::transpose::trans, oneapi::math::transpose::nontrans,
                 T, T, HS, alpha, vv, HS, (int64_t)T * HS, dvaccum, HS,
                 (int64_t)T * HS, beta, datt, T, (int64_t)T * T, (int64_t)B * NH);

  col_gemm_batch(q, oneapi::math::transpose::nontrans, oneapi::math::transpose::trans,
                 HS, T, T, alpha, dvaccum, HS, (int64_t)T * HS, att, T,
                 (int64_t)T * T, beta, dv, HS, (int64_t)T * HS, (int64_t)B * NH);

  switch (variant) {
    case 1: launch_softmax_bwd1(q, dpreatt, datt, att, B, T, C, NH); break;
    case 2: launch_softmax_bwd2(q, dpreatt, datt, att, B, T, C, NH); break;
    case 3: launch_softmax_bwd3(q, dpreatt, datt, att, B, T, C, NH); break;
    case 4: launch_softmax_bwd4(q, dpreatt, datt, att, B, T, C, NH); break;
    case 5: launch_softmax_bwd5(q, dpreatt, datt, att, B, T, C, NH); break;
    case 6: launch_softmax_bwd6(q, dpreatt, datt, att, B, T, C, NH, block_size); break;
    case 7: launch_softmax_bwd7(q, dpreatt, datt, att, B, T, C, NH, block_size); break;
    case 8: launch_softmax_bwd8(q, dpreatt, datt, att, B, T, C, NH, block_size); break;
    default: break;
  }

  col_gemm_batch(q, oneapi::math::transpose::nontrans, oneapi::math::transpose::nontrans,
                 HS, T, T, alpha, kk, HS, (int64_t)T * HS, dpreatt, T,
                 (int64_t)T * T, beta, dq, HS, (int64_t)T * HS, (int64_t)B * NH);

  col_gemm_batch(q, oneapi::math::transpose::nontrans, oneapi::math::transpose::trans,
                 HS, T, T, alpha, qq, HS, (int64_t)T * HS, dpreatt, T,
                 (int64_t)T * T, beta, dk, HS, (int64_t)T * HS, (int64_t)B * NH);

  dev_permute_backward(q, dinp, dq, dk, dv, B, T, NH, HS);
}

// ----------------------------------------------------------------------------
// Contract main
int main(int argc, char **argv) {
  if (argc != 7) {
    fprintf(stderr, "usage: %s <input> <output> B T C NH\n", argv[0]);
    return 2;
  }
  int B = atoi(argv[3]);
  int T = atoi(argv[4]);
  int C = atoi(argv[5]);
  int NH = atoi(argv[6]);

  FILE *fin = fopen(argv[1], "rb");
  if (!fin) { perror(argv[1]); return 2; }
  size_t n_inp = (size_t)B * T * 3 * C;
  size_t n_dout = (size_t)B * T * C;
  float *inp = (float *)malloc(sizeof(float) * n_inp);
  if (fread(inp, sizeof(float), n_inp, fin) != n_inp) {
    fprintf(stderr, "input too short at inp\n");
    return 2;
  }
  float *dout = (float *)malloc(sizeof(float) * n_dout);
  if (fread(dout, sizeof(float), n_dout, fin) != n_dout) {
    fprintf(stderr, "input too short at dout\n");
    return 2;
  }
  fclose(fin);
  FILE *fout = fopen(argv[2], "wb");
  if (!fout) { perror(argv[2]); return 2; }

  // host forward reference (not used for device path, keeps parity)
  float *fwd_out = (float *)malloc(sizeof(float) * (size_t)B * T * C);
  float *preatt_h = (float *)malloc(sizeof(float) * (size_t)B * NH * T * T);
  float *att_h = (float *)malloc(sizeof(float) * (size_t)B * NH * T * T);
  attention_forward_cpu(fwd_out, preatt_h, att_h, inp, B, T, C, NH);
  free(fwd_out); free(preatt_h); free(att_h);

  sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order{}};

  auto dev_malloc = [&](size_t n) -> float * {
    return sycl::malloc_device<float>(n, q);
  };
  size_t sz_qkvr = (size_t)B * T * 3 * C;
  size_t sz_pre = (size_t)B * NH * T * T;
  size_t sz_vac = (size_t)B * T * C;
  size_t sz_inp = (size_t)B * T * 3 * C;

  float *d_inp = dev_malloc(sz_inp);
  float *d_dout = dev_malloc(n_dout);
  float *d_dinp = dev_malloc(sz_inp);
  float *d_qkvr = dev_malloc(sz_qkvr);
  float *d_preatt = dev_malloc(sz_pre);
  float *d_att = dev_malloc(sz_pre);
  float *d_vaccum = dev_malloc(sz_vac);
  float *d_fwd_out = dev_malloc(sz_vac);
  float *d_dqkvr = dev_malloc(sz_qkvr);
  float *d_dpreatt = dev_malloc(sz_pre);
  float *d_datt = dev_malloc(sz_pre);
  float *d_dvaccum = dev_malloc(sz_vac);

  q.memcpy(d_inp, inp, sizeof(float) * sz_inp).wait();
  q.memcpy(d_dout, dout, sizeof(float) * n_dout).wait();
  q.memset(d_att, 0, sizeof(float) * sz_pre).wait();

  attention_forward(q, d_fwd_out, d_vaccum, d_qkvr, d_preatt, d_att, d_inp, B, T,
                    C, NH, 256);
  q.wait();

  float *dinp = (float *)malloc(sizeof(float) * sz_inp);
  const int variants[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  for (int vi = 0; vi < 8; vi++) {
    int k = variants[vi];
    q.memset(d_dinp, 0, sizeof(float) * sz_inp).wait();
    q.memset(d_dqkvr, 0, sizeof(float) * sz_qkvr).wait();
    q.memset(d_dpreatt, 0, sizeof(float) * sz_pre).wait();
    q.memset(d_datt, 0, sizeof(float) * sz_pre).wait();
    q.memset(d_dvaccum, 0, sizeof(float) * sz_vac).wait();
    attention_backward_impl(q, d_dinp, d_dqkvr, d_dpreatt, d_datt, d_dvaccum,
                            d_dout, d_inp, d_qkvr, d_preatt, d_att, d_vaccum, B,
                            T, C, NH, k, 256);
    q.wait();
    q.memcpy(dinp, d_dinp, sizeof(float) * sz_inp).wait();
    fwrite(dinp, sizeof(float), sz_inp, fout);
  }
  fclose(fout);

  sycl::free(d_inp, q); sycl::free(d_dout, q); sycl::free(d_dinp, q);
  sycl::free(d_qkvr, q); sycl::free(d_preatt, q); sycl::free(d_att, q);
  sycl::free(d_vaccum, q); sycl::free(d_fwd_out, q); sycl::free(d_dqkvr, q);
  sycl::free(d_dpreatt, q); sycl::free(d_datt, q); sycl::free(d_dvaccum, q);
  free(inp); free(dout); free(dinp);
  return 0;
}
