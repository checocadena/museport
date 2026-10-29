// Idiomatic SYCL 2020 port of fused_residual_forward (fp32 build).
// Six kernel variants, each its own SYCL kernel, 1-D launches only.
#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <algorithm>

using floatX = float;

template <typename T>
inline T ceil_div(T a, T b) { return (a + b - 1) / b; }

// ---- kernel name tags (one SYCL kernel per variant; variant 1 has two) ----
class ResK1Tag;
class LnK1Tag;
class FusedK2Tag;
class FusedK3Tag;
class FusedK4Tag;
class FusedK5Tag;
class FusedK6Tag;

// ---------------------------------------------------------------------------
// variant 1: residual elementwise + naive layernorm (two SYCL kernels)
// ---------------------------------------------------------------------------
void fused_residual_forward1(sycl::queue &q, floatX *residual, floatX *normed,
                             floatX *mean, floatX *rstd, const floatX *inp1,
                             const floatX *inp2, const floatX *weight,
                             const floatX *bias, int N, int C) {
  const size_t nc = static_cast<size_t>(N) * static_cast<size_t>(C);
  q.parallel_for<ResK1Tag>(sycl::range<1>(nc), [=](sycl::id<1> i) {
     size_t idx = i[0];
     residual[idx] = inp1[idx] + inp2[idx];
  }).wait();

  q.parallel_for<LnK1Tag>(sycl::range<1>(static_cast<size_t>(N)),
                          [=](sycl::id<1> id) {
    int idx = static_cast<int>(id[0]);
    const float eps = 1e-5f;
    const floatX *x = residual + static_cast<size_t>(idx) * C;
    float m = 0.0f;
    for (int i = 0; i < C; ++i) m += static_cast<float>(x[i]);
    m /= C;
    float v = 0.0f;
    for (int i = 0; i < C; ++i) {
      float d = static_cast<float>(x[i]) - m;
      v += d * d;
    }
    v /= C;
    float s = 1.0f / sycl::sqrt(v + eps);
    floatX *o = normed + static_cast<size_t>(idx) * C;
    for (int i = 0; i < C; ++i) {
      float n = s * (static_cast<float>(x[i]) - m);
      o[i] = static_cast<float>(n * static_cast<float>(weight[i]) +
                                static_cast<float>(bias[i]));
    }
    mean[idx] = m;
    rstd[idx] = s;
  }).wait();
}

// ---------------------------------------------------------------------------
// variant 2: naive fused, one work-item per row
// ---------------------------------------------------------------------------
void fused_residual_forward2(sycl::queue &q, floatX *residual, floatX *normed,
                             floatX *mean, floatX *rstd, const floatX *inp1,
                             const floatX *inp2, const floatX *weight,
                             const floatX *bias, int N, int C) {
  q.parallel_for<FusedK2Tag>(sycl::range<1>(static_cast<size_t>(N)),
                             [=](sycl::id<1> id) {
    int idx = static_cast<int>(id[0]);
    floatX *res = residual + static_cast<size_t>(idx) * C;
    floatX *out = normed + static_cast<size_t>(idx) * C;
    const floatX *a = inp1 + static_cast<size_t>(idx) * C;
    const floatX *b = inp2 + static_cast<size_t>(idx) * C;
    const float eps = 1e-5f;
    float m = 0.0f;
    for (int c = 0; c < C; ++c) {
      float o = static_cast<float>(a[c]) + static_cast<float>(b[c]);
      m += o;
      res[c] = o;
    }
    m /= C;
    float v = 0.0f;
    for (int c = 0; c < C; ++c) {
      float d = static_cast<float>(res[c]) - m;
      v += d * d;
    }
    v /= C;
    float s = 1.0f / sycl::sqrt(v + eps);
    for (int c = 0; c < C; ++c) {
      float n = s * (static_cast<float>(res[c]) - m);
      out[c] = n * static_cast<float>(weight[c]) + static_cast<float>(bias[c]);
    }
    mean[idx] = m;
    rstd[idx] = s;
  }).wait();
}

// ---------------------------------------------------------------------------
// variant 3: one work-group per row, strided cooperating items + group reduce
// ---------------------------------------------------------------------------
void fused_residual_forward3(sycl::queue &q, floatX *residual, floatX *normed,
                             floatX *mean, floatX *rstd, const floatX *inp1,
                             const floatX *inp2, const floatX *weight,
                             const floatX *bias, int N, int C) {
  if (N <= 0) return;
  const size_t ls = 128;
  const size_t gs = static_cast<size_t>(N) * ls;
  q.parallel_for<FusedK3Tag>(
     sycl::nd_range<1>(sycl::range<1>(gs), sycl::range<1>(ls)),
     [=](sycl::nd_item<1> it) {
    sycl::group<1> g = it.get_group();
    size_t row = g.get_group_id(0);
    size_t lid = it.get_local_id(0);
    size_t lsize = it.get_local_range(0);
    floatX *res = residual + row * static_cast<size_t>(C);
    floatX *out = normed + row * static_cast<size_t>(C);
    const floatX *a = inp1 + row * static_cast<size_t>(C);
    const floatX *b = inp2 + row * static_cast<size_t>(C);
    const float eps = 1e-5f;
    float part = 0.0f;
    for (int c = static_cast<int>(lid); c < C; c += static_cast<int>(lsize)) {
      float o = a[c] + b[c];
      res[c] = o;
      part += o;
    }
    float sum = sycl::reduce_over_group(g, part, sycl::plus<float>{});
    float m = sum / static_cast<float>(C);
    sycl::group_barrier(g);
    float vpart = 0.0f;
    for (int c = static_cast<int>(lid); c < C; c += static_cast<int>(lsize)) {
      float d = res[c] - m;
      vpart += d * d;
    }
    float vsum = sycl::reduce_over_group(g, vpart, sycl::plus<float>{});
    float v = vsum / static_cast<float>(C);
    float s = 1.0f / sycl::sqrt(v + eps);
    for (int c = static_cast<int>(lid); c < C; c += static_cast<int>(lsize)) {
      float n = s * (res[c] - m);
      out[c] = n * weight[c] + bias[c];
    }
    if (lid == 0) {
      mean[row] = m;
      rstd[row] = s;
    }
  }).wait();
}

// ---------------------------------------------------------------------------
// variant 4: vectorized (float4) single-pass stats, streaming style
// ---------------------------------------------------------------------------
void fused_residual_forward4(sycl::queue &q, floatX *residual, floatX *normed,
                             floatX *mean, floatX *rstd, const floatX *inp1,
                             const floatX *inp2, const floatX *weight,
                             const floatX *bias, int N, int C) {
  if (N <= 0) return;
  const size_t ls = 128;
  const size_t gs = static_cast<size_t>(N) * ls;
  q.parallel_for<FusedK4Tag>(
     sycl::nd_range<1>(sycl::range<1>(gs), sycl::range<1>(ls)),
     [=](sycl::nd_item<1> it) {
    sycl::group<1> g = it.get_group();
    size_t row = g.get_group_id(0);
    size_t lid = it.get_local_id(0);
    size_t lsize = it.get_local_range(0);
    floatX *res = residual + row * static_cast<size_t>(C);
    floatX *out = normed + row * static_cast<size_t>(C);
    const floatX *a = inp1 + row * static_cast<size_t>(C);
    const floatX *b = inp2 + row * static_cast<size_t>(C);
    const float eps = 1e-5f;
    using vec4 = sycl::vec<float, 4>;
    float sum = 0.0f, sum_sq = 0.0f;
    if (C % 4 == 0) {
      const vec4 *pa = reinterpret_cast<const vec4 *>(a);
      const vec4 *pb = reinterpret_cast<const vec4 *>(b);
      vec4 *pr = reinterpret_cast<vec4 *>(res);
      int nvec = C / 4;
      for (int vi = static_cast<int>(lid); vi < nvec;
           vi += static_cast<int>(lsize)) {
        vec4 x = pa[vi] + pb[vi];
        pr[vi] = x;
        sum += x.x() + x.y() + x.z() + x.w();
        sum_sq += x.x() * x.x() + x.y() * x.y() + x.z() * x.z() + x.w() * x.w();
      }
      sum = sycl::reduce_over_group(g, sum, sycl::plus<float>{});
      sum_sq = sycl::reduce_over_group(g, sum_sq, sycl::plus<float>{});
      float m = sum / static_cast<float>(C);
      float v = sum_sq / static_cast<float>(C) - m * m;
      float s = sycl::rsqrt(v + eps);
      sycl::group_barrier(g);
      const vec4 *pw = reinterpret_cast<const vec4 *>(weight);
      const vec4 *pbi = reinterpret_cast<const vec4 *>(bias);
      const vec4 *pr2 = reinterpret_cast<const vec4 *>(res);
      vec4 *po = reinterpret_cast<vec4 *>(out);
      for (int vi = static_cast<int>(lid); vi < nvec;
           vi += static_cast<int>(lsize)) {
        vec4 r = pr2[vi];
        vec4 w = pw[vi];
        vec4 bb = pbi[vi];
        vec4 o;
        o.x() = s * (r.x() - m) * w.x() + bb.x();
        o.y() = s * (r.y() - m) * w.y() + bb.y();
        o.z() = s * (r.z() - m) * w.z() + bb.z();
        o.w() = s * (r.w() - m) * w.w() + bb.w();
        po[vi] = o;
      }
      if (lid == 0) {
        mean[row] = m;
        rstd[row] = s;
      }
    } else {
      // generic fallback for C not divisible by 4 (same single-pass math)
      for (int c = static_cast<int>(lid); c < C; c += static_cast<int>(lsize)) {
        float o = a[c] + b[c];
        res[c] = o;
        sum += o;
        sum_sq += o * o;
      }
      sum = sycl::reduce_over_group(g, sum, sycl::plus<float>{});
      sum_sq = sycl::reduce_over_group(g, sum_sq, sycl::plus<float>{});
      float m = sum / static_cast<float>(C);
      float v = sum_sq / static_cast<float>(C) - m * m;
      float s = sycl::rsqrt(v + eps);
      sycl::group_barrier(g);
      for (int c = static_cast<int>(lid); c < C; c += static_cast<int>(lsize)) {
        float n = s * (res[c] - m);
        out[c] = n * weight[c] + bias[c];
      }
      if (lid == 0) {
        mean[row] = m;
        rstd[row] = s;
      }
    }
  }).wait();
}

// ---------------------------------------------------------------------------
// variant 5: work-group per row with local-memory cache (weights/bias/resid)
// ---------------------------------------------------------------------------
void fused_residual_forward5(sycl::queue &q, floatX *residual, floatX *normed,
                             floatX *mean, floatX *rstd, const floatX *inp1,
                             const floatX *inp2, const floatX *weight,
                             const floatX *bias, int N, int C) {
  if (N <= 0) return;
  const size_t ls = 128;
  const size_t gs = static_cast<size_t>(N) * ls;
  size_t scratch_elems = static_cast<size_t>(3) * static_cast<size_t>(C);
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> scr(sycl::range<1>(scratch_elems), h);
    h.parallel_for<FusedK5Tag>(
       sycl::nd_range<1>(sycl::range<1>(gs), sycl::range<1>(ls)),
       [=](sycl::nd_item<1> it) {
      sycl::group<1> g = it.get_group();
      size_t row = g.get_group_id(0);
      size_t lid = it.get_local_id(0);
      size_t lsize = it.get_local_range(0);
      // scr layout: [0,C) weights, [C,2C) bias, [2C,3C) residual cache
      for (int c = static_cast<int>(lid); c < C; c += static_cast<int>(lsize)) {
        scr[c] = weight[c];
        scr[static_cast<size_t>(C) + c] = bias[c];
      }
      sycl::group_barrier(g);
      floatX *res = residual + row * static_cast<size_t>(C);
      floatX *out = normed + row * static_cast<size_t>(C);
      const floatX *a = inp1 + row * static_cast<size_t>(C);
      const floatX *b = inp2 + row * static_cast<size_t>(C);
      const float eps = 1e-5f;
      float part = 0.0f;
      for (int c = static_cast<int>(lid); c < C; c += static_cast<int>(lsize)) {
        float o = a[c] + b[c];
        res[c] = o;
        scr[static_cast<size_t>(2) * static_cast<size_t>(C) + c] = o;
        part += o;
      }
      float sum = sycl::reduce_over_group(g, part, sycl::plus<float>{});
      float m = sum / static_cast<float>(C);
      sycl::group_barrier(g);
      float vpart = 0.0f;
      for (int c = static_cast<int>(lid); c < C; c += static_cast<int>(lsize)) {
        float d = scr[static_cast<size_t>(2) * static_cast<size_t>(C) + c] - m;
        vpart += d * d;
      }
      float vsum = sycl::reduce_over_group(g, vpart, sycl::plus<float>{});
      float v = vsum / static_cast<float>(C);
      float s = sycl::rsqrt(v + eps);
      for (int c = static_cast<int>(lid); c < C; c += static_cast<int>(lsize)) {
        float r = scr[static_cast<size_t>(2) * static_cast<size_t>(C) + c];
        float n = s * (r - m);
        out[c] = n * scr[c] + scr[static_cast<size_t>(C) + c];
      }
      if (lid == 0) {
        mean[row] = m;
        rstd[row] = s;
      }
    });
  }).wait();
}

// ---------------------------------------------------------------------------
// variant 6: persistent work-groups, each loops over rows grid-stride
// ---------------------------------------------------------------------------
void fused_residual_forward6(sycl::queue &q, floatX *residual, floatX *normed,
                             floatX *mean, floatX *rstd, const floatX *inp1,
                             const floatX *inp2, const floatX *weight,
                             const floatX *bias, int N, int C) {
  if (N <= 0) return;
  const size_t ls = 128;
  int ng = std::min<int>(N, 32);
  if (ng < 1) ng = 1;
  const size_t gs = static_cast<size_t>(ng) * ls;
  size_t scratch_elems = static_cast<size_t>(3) * static_cast<size_t>(C);
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> scr(sycl::range<1>(scratch_elems), h);
    h.parallel_for<FusedK6Tag>(
       sycl::nd_range<1>(sycl::range<1>(gs), sycl::range<1>(ls)),
       [=](sycl::nd_item<1> it) {
      sycl::group<1> g = it.get_group();
      size_t gid = g.get_group_id(0);
      size_t ngrps = g.get_group_range(0);
      size_t lid = it.get_local_id(0);
      size_t lsize = it.get_local_range(0);
      for (int c = static_cast<int>(lid); c < C; c += static_cast<int>(lsize)) {
        scr[c] = weight[c];
        scr[static_cast<size_t>(C) + c] = bias[c];
      }
      sycl::group_barrier(g);
      const float eps = 1e-5f;
      for (size_t row = gid; row < static_cast<size_t>(N); row += ngrps) {
        floatX *res = residual + row * static_cast<size_t>(C);
        floatX *out = normed + row * static_cast<size_t>(C);
        const floatX *a = inp1 + row * static_cast<size_t>(C);
        const floatX *b = inp2 + row * static_cast<size_t>(C);
        float part = 0.0f;
        for (int c = static_cast<int>(lid); c < C; c += static_cast<int>(lsize)) {
          float o = a[c] + b[c];
          res[c] = o;
          scr[static_cast<size_t>(2) * static_cast<size_t>(C) + c] = o;
          part += o;
        }
        float sum = sycl::reduce_over_group(g, part, sycl::plus<float>{});
        float m = sum / static_cast<float>(C);
        sycl::group_barrier(g);
        float vpart = 0.0f;
        for (int c = static_cast<int>(lid); c < C; c += static_cast<int>(lsize)) {
          float d = scr[static_cast<size_t>(2) * static_cast<size_t>(C) + c] - m;
          vpart += d * d;
        }
        float vsum = sycl::reduce_over_group(g, vpart, sycl::plus<float>{});
        float v = vsum / static_cast<float>(C);
        float s = sycl::rsqrt(v + eps);
        for (int c = static_cast<int>(lid); c < C; c += static_cast<int>(lsize)) {
          float r = scr[static_cast<size_t>(2) * static_cast<size_t>(C) + c];
          float n = s * (r - m);
          out[c] = n * scr[c] + scr[static_cast<size_t>(C) + c];
        }
        if (lid == 0) {
          mean[row] = m;
          rstd[row] = s;
        }
        // make local reuse safe across loop iterations
        sycl::group_barrier(g);
      }
    });
  }).wait();
}

void fused_residual_forward(sycl::queue &q, int kernel_num, floatX *residual,
                            floatX *normed, floatX *mean, floatX *rstd,
                            const floatX *inp1, const floatX *inp2,
                            const floatX *weight, const floatX *bias, int N,
                            int C) {
  switch (kernel_num) {
    case 1: fused_residual_forward1(q, residual, normed, mean, rstd, inp1, inp2,
                                    weight, bias, N, C); break;
    case 2: fused_residual_forward2(q, residual, normed, mean, rstd, inp1, inp2,
                                    weight, bias, N, C); break;
    case 3: fused_residual_forward3(q, residual, normed, mean, rstd, inp1, inp2,
                                    weight, bias, N, C); break;
    case 4: fused_residual_forward4(q, residual, normed, mean, rstd, inp1, inp2,
                                    weight, bias, N, C); break;
    case 5: fused_residual_forward5(q, residual, normed, mean, rstd, inp1, inp2,
                                    weight, bias, N, C); break;
    case 6: fused_residual_forward6(q, residual, normed, mean, rstd, inp1, inp2,
                                    weight, bias, N, C); break;
    default: fprintf(stderr, "Invalid kernel number\n"); exit(1);
  }
}

int main(int argc, char **argv) {
  if (argc != 6) {
    fprintf(stderr, "usage: %s <input> <output> B T C\n", argv[0]);
    return 2;
  }
  const int B = atoi(argv[3]);
  const int T = atoi(argv[4]);
  const int C = atoi(argv[5]);
  const size_t ntc = static_cast<size_t>(B) * T * C;
  const size_t nt = static_cast<size_t>(B) * T;
  FILE *fin = fopen(argv[1], "rb");
  if (!fin) { perror(argv[1]); return 2; }
  float *inp1 = (float *)malloc(sizeof(float) * ntc);
  float *inp2 = (float *)malloc(sizeof(float) * ntc);
  float *weight = (float *)malloc(sizeof(float) * (size_t)C);
  float *bias = (float *)malloc(sizeof(float) * (size_t)C);
  if (fread(inp1, sizeof(float), ntc, fin) != ntc) {
    fprintf(stderr, "input too short at inp1\n"); return 2;
  }
  if (fread(inp2, sizeof(float), ntc, fin) != ntc) {
    fprintf(stderr, "input too short at inp2\n"); return 2;
  }
  if (fread(weight, sizeof(float), (size_t)C, fin) != (size_t)C) {
    fprintf(stderr, "input too short at weight\n"); return 2;
  }
  if (fread(bias, sizeof(float), (size_t)C, fin) != (size_t)C) {
    fprintf(stderr, "input too short at bias\n"); return 2;
  }
  fclose(fin);

  sycl::queue q{sycl::default_selector_v};
  float *d_inp1 = sycl::malloc_device<float>(ntc, q);
  float *d_inp2 = sycl::malloc_device<float>(ntc, q);
  float *d_weight = sycl::malloc_device<float>((size_t)C, q);
  float *d_bias = sycl::malloc_device<float>((size_t)C, q);
  float *d_residual = sycl::malloc_device<float>(ntc, q);
  float *d_normed = sycl::malloc_device<float>(ntc, q);
  float *d_mean = sycl::malloc_device<float>(nt, q);
  float *d_rstd = sycl::malloc_device<float>(nt, q);
  q.memcpy(d_inp1, inp1, sizeof(float) * ntc).wait();
  q.memcpy(d_inp2, inp2, sizeof(float) * ntc).wait();
  q.memcpy(d_weight, weight, sizeof(float) * (size_t)C).wait();
  q.memcpy(d_bias, bias, sizeof(float) * (size_t)C).wait();

  float *residual = (float *)malloc(sizeof(float) * ntc);
  float *normed = (float *)malloc(sizeof(float) * ntc);
  float *mean = (float *)malloc(sizeof(float) * nt);
  float *rstd = (float *)malloc(sizeof(float) * nt);

  FILE *fout = fopen(argv[2], "wb");
  if (!fout) { perror(argv[2]); return 2; }
  const int variants[] = {1, 2, 3, 4, 5, 6};
  for (int vi = 0; vi < 6; ++vi) {
    const int k = variants[vi];
    q.memset(d_residual, 0, sizeof(float) * ntc).wait();
    q.memset(d_normed, 0, sizeof(float) * ntc).wait();
    q.memset(d_mean, 0, sizeof(float) * nt).wait();
    q.memset(d_rstd, 0, sizeof(float) * nt).wait();
    fused_residual_forward(q, k, d_residual, d_normed, d_mean, d_rstd, d_inp1,
                           d_inp2, d_weight, d_bias, B * T, C);
    q.memcpy(residual, d_residual, sizeof(float) * ntc).wait();
    fwrite(residual, sizeof(float), ntc, fout);
    q.memcpy(normed, d_normed, sizeof(float) * ntc).wait();
    fwrite(normed, sizeof(float), ntc, fout);
    q.memcpy(mean, d_mean, sizeof(float) * nt).wait();
    fwrite(mean, sizeof(float), nt, fout);
    q.memcpy(rstd, d_rstd, sizeof(float) * nt).wait();
    fwrite(rstd, sizeof(float), nt, fout);
  }
  fclose(fout);
  sycl::free(d_inp1, q);
  sycl::free(d_inp2, q);
  sycl::free(d_weight, q);
  sycl::free(d_bias, q);
  sycl::free(d_residual, q);
  sycl::free(d_normed, q);
  sycl::free(d_mean, q);
  sycl::free(d_rstd, q);
  free(inp1); free(inp2); free(weight); free(bias);
  free(residual); free(normed); free(mean); free(rstd);
  return 0;
}
