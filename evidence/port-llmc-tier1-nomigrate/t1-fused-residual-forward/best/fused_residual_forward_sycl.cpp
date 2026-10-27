#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>

using FloatX = float;

namespace {

constexpr float kEps = 1e-5f;

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

// Distinct kernel names for each variant (SYCL 2020 forward-declared tags).
class ResidualEltK1;
class LayerNormK1;
class FusedK2;
class FusedK3;
class FusedK4;
class FusedK5;
class FusedK6;

void launch_variant1(sycl::queue &q, float *residual, float *normed, float *mean,
                     float *rstd, const float *inp1, const float *inp2,
                     const float *weight, const float *bias, int n, int c) {
  const std::size_t total = static_cast<std::size_t>(n) * static_cast<std::size_t>(c);
  // Elementwise residual add.
  q.parallel_for<ResidualEltK1>(sycl::range<1>(total), [=](sycl::id<1> wid) {
     std::size_t i = wid[0];
     residual[i] = inp1[i] + inp2[i];
   }).wait();
  // Per-row layernorm over the residual buffer.
  q.parallel_for<LayerNormK1>(sycl::range<1>(static_cast<std::size_t>(n)), [=](sycl::id<1> wid) {
     int row = static_cast<int>(wid[0]);
     const float *x = residual + static_cast<std::size_t>(row) * static_cast<std::size_t>(c);
     float m = 0.0f;
     for (int i = 0; i < c; ++i) {
       m += x[i];
     }
     m /= static_cast<float>(c);
     float v = 0.0f;
     for (int i = 0; i < c; ++i) {
       float d = x[i] - m;
       v += d * d;
     }
     v /= static_cast<float>(c);
     float s = 1.0f / sycl::sqrt(v + kEps);
     float *o = normed + static_cast<std::size_t>(row) * static_cast<std::size_t>(c);
     for (int i = 0; i < c; ++i) {
       float nv = s * (x[i] - m);
       o[i] = nv * weight[i] + bias[i];
     }
     mean[row] = m;
     rstd[row] = s;
   }).wait();
}

void launch_variant2(sycl::queue &q, float *residual, float *normed, float *mean,
                     float *rstd, const float *inp1, const float *inp2,
                     const float *weight, const float *bias, int n, int c) {
  q.parallel_for<FusedK2>(sycl::range<1>(static_cast<std::size_t>(n)), [=](sycl::id<1> wid) {
     int row = static_cast<int>(wid[0]);
     const float *a = inp1 + static_cast<std::size_t>(row) * static_cast<std::size_t>(c);
     const float *b = inp2 + static_cast<std::size_t>(row) * static_cast<std::size_t>(c);
     float *res = residual + static_cast<std::size_t>(row) * static_cast<std::size_t>(c);
     float *out = normed + static_cast<std::size_t>(row) * static_cast<std::size_t>(c);
     float m = 0.0f;
     for (int i = 0; i < c; ++i) {
       float v = a[i] + b[i];
       res[i] = v;
       m += v;
     }
     m /= static_cast<float>(c);
     float var = 0.0f;
     for (int i = 0; i < c; ++i) {
       float d = res[i] - m;
       var += d * d;
     }
     var /= static_cast<float>(c);
     float s = 1.0f / sycl::sqrt(var + kEps);
     for (int i = 0; i < c; ++i) {
       float nv = s * (res[i] - m);
       out[i] = nv * weight[i] + bias[i];
     }
     mean[row] = m;
     rstd[row] = s;
   }).wait();
}

void launch_variant3(sycl::queue &q, float *residual, float *normed, float *mean,
                     float *rstd, const float *inp1, const float *inp2,
                     const float *weight, const float *bias, int n, int c) {
  q.parallel_for<FusedK3>(sycl::range<1>(static_cast<std::size_t>(n)), [=](sycl::id<1> wid) {
     int row = static_cast<int>(wid[0]);
     std::size_t base = static_cast<std::size_t>(row) * static_cast<std::size_t>(c);
     float acc = 0.0f;
     for (int i = 0; i < c; ++i) {
       float sum = inp1[base + static_cast<std::size_t>(i)] + inp2[base + static_cast<std::size_t>(i)];
       residual[base + static_cast<std::size_t>(i)] = sum;
       acc += sum;
     }
     float mval = acc / static_cast<float>(c);
     float vacc = 0.0f;
     for (int i = 0; i < c; ++i) {
       float d = residual[base + static_cast<std::size_t>(i)] - mval;
       vacc += d * d;
     }
     vacc /= static_cast<float>(c);
     float sinv = 1.0f / sycl::sqrt(vacc + kEps);
     for (int i = 0; i < c; ++i) {
       float nrm = sinv * (residual[base + static_cast<std::size_t>(i)] - mval);
       normed[base + static_cast<std::size_t>(i)] = nrm * weight[i] + bias[i];
     }
     mean[row] = mval;
     rstd[row] = sinv;
   }).wait();
}

void launch_variant4(sycl::queue &q, float *residual, float *normed, float *mean,
                     float *rstd, const float *inp1, const float *inp2,
                     const float *weight, const float *bias, int n, int c) {
  // Single-pass statistics with sum and sum-of-squares streaming.
  q.parallel_for<FusedK4>(sycl::range<1>(static_cast<std::size_t>(n)), [=](sycl::id<1> wid) {
     int row = static_cast<int>(wid[0]);
     std::size_t base = static_cast<std::size_t>(row) * static_cast<std::size_t>(c);
     float sum = 0.0f;
     float sum_sq = 0.0f;
     for (int i = 0; i < c; ++i) {
       float v = inp1[base + static_cast<std::size_t>(i)] + inp2[base + static_cast<std::size_t>(i)];
       residual[base + static_cast<std::size_t>(i)] = v;
       sum += v;
       sum_sq += v * v;
     }
     float mval = sum / static_cast<float>(c);
     float vval = sum_sq / static_cast<float>(c) - mval * mval;
     // Guard tiny negative from rounding.
     if (vval < 0.0f && vval > -1e-9f) {
       vval = 0.0f;
     }
     float sinv = sycl::rsqrt(vval + kEps);
     for (int i = 0; i < c; ++i) {
       float nrm = sinv * (residual[base + static_cast<std::size_t>(i)] - mval);
       normed[base + static_cast<std::size_t>(i)] = nrm * weight[i] + bias[i];
     }
     mean[row] = mval;
     rstd[row] = sinv;
   }).wait();
}

void launch_variant5(sycl::queue &q, float *residual, float *normed, float *mean,
                     float *rstd, const float *inp1, const float *inp2,
                     const float *weight, const float *bias, int n, int c) {
  // Two-pass stable variance with reversed second sweep order.
  q.parallel_for<FusedK5>(sycl::range<1>(static_cast<std::size_t>(n)), [=](sycl::id<1> wid) {
     int row = static_cast<int>(wid[0]);
     std::size_t base = static_cast<std::size_t>(row) * static_cast<std::size_t>(c);
     float tot = 0.0f;
     for (int i = 0; i < c; ++i) {
       float v = inp1[base + static_cast<std::size_t>(i)] + inp2[base + static_cast<std::size_t>(i)];
       residual[base + static_cast<std::size_t>(i)] = v;
       tot += v;
     }
     float mval = tot / static_cast<float>(c);
     float vacc = 0.0f;
     for (int i = c - 1; i >= 0; --i) {
       float d = residual[base + static_cast<std::size_t>(i)] - mval;
       vacc += d * d;
     }
     vacc /= static_cast<float>(c);
     float sinv = 1.0f / sycl::sqrt(vacc + kEps);
     for (int i = 0; i < c; ++i) {
       float nrm = sinv * (residual[base + static_cast<std::size_t>(i)] - mval);
       normed[base + static_cast<std::size_t>(i)] = nrm * weight[i] + bias[i];
     }
     mean[row] = mval;
     rstd[row] = sinv;
   }).wait();
}

void launch_variant6(sycl::queue &q, float *residual, float *normed, float *mean,
                     float *rstd, const float *inp1, const float *inp2,
                     const float *weight, const float *bias, int n, int c) {
  // Persistent-style fused pass with split accumulation pairs.
  q.parallel_for<FusedK6>(sycl::range<1>(static_cast<std::size_t>(n)), [=](sycl::id<1> wid) {
     int row = static_cast<int>(wid[0]);
     std::size_t base = static_cast<std::size_t>(row) * static_cast<std::size_t>(c);
     float acc_a = 0.0f;
     float acc_b = 0.0f;
     for (int i = 0; i < c; ++i) {
       float v = inp1[base + static_cast<std::size_t>(i)] + inp2[base + static_cast<std::size_t>(i)];
       residual[base + static_cast<std::size_t>(i)] = v;
       if ((i & 1) == 0) {
         acc_a += v;
       } else {
         acc_b += v;
       }
     }
     float mval = (acc_a + acc_b) / static_cast<float>(c);
     float va = 0.0f;
     float vb = 0.0f;
     for (int i = 0; i < c; ++i) {
       float d = residual[base + static_cast<std::size_t>(i)] - mval;
       float qv = d * d;
       if ((i & 1) == 0) {
         va += qv;
       } else {
         vb += qv;
       }
     }
     float vval = (va + vb) / static_cast<float>(c);
     float sinv = sycl::rsqrt(vval + kEps);
     for (int i = 0; i < c; ++i) {
       float nrm = sinv * (residual[base + static_cast<std::size_t>(i)] - mval);
       normed[base + static_cast<std::size_t>(i)] = nrm * weight[i] + bias[i];
     }
     mean[row] = mval;
     rstd[row] = sinv;
   }).wait();
}

void run_variant(sycl::queue &q, int k, float *residual, float *normed, float *mean,
                 float *rstd, const float *inp1, const float *inp2,
                 const float *weight, const float *bias, int n, int c) {
  switch (k) {
    case 1: launch_variant1(q, residual, normed, mean, rstd, inp1, inp2, weight, bias, n, c); break;
    case 2: launch_variant2(q, residual, normed, mean, rstd, inp1, inp2, weight, bias, n, c); break;
    case 3: launch_variant3(q, residual, normed, mean, rstd, inp1, inp2, weight, bias, n, c); break;
    case 4: launch_variant4(q, residual, normed, mean, rstd, inp1, inp2, weight, bias, n, c); break;
    case 5: launch_variant5(q, residual, normed, mean, rstd, inp1, inp2, weight, bias, n, c); break;
    case 6: launch_variant6(q, residual, normed, mean, rstd, inp1, inp2, weight, bias, n, c); break;
    default: std::fprintf(stderr, "Invalid kernel number\n"); std::exit(1);
  }
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 6) {
    std::fprintf(stderr, "usage: %s <input> <output> B T C\n", argv[0]);
    return 2;
  }
  const int B = std::atoi(argv[3]);
  const int T = std::atoi(argv[4]);
  const int C = std::atoi(argv[5]);
  const int N = B * T;
  const std::size_t ntc = static_cast<std::size_t>(B) * static_cast<std::size_t>(T) * static_cast<std::size_t>(C);
  const std::size_t nt = static_cast<std::size_t>(B) * static_cast<std::size_t>(T);
  const std::size_t csz = static_cast<std::size_t>(C);

  FILE *fin = std::fopen(argv[1], "rb");
  if (!fin) {
    std::perror(argv[1]);
    return 2;
  }
  float *h_inp1 = static_cast<float *>(std::malloc(ntc * sizeof(float)));
  float *h_inp2 = static_cast<float *>(std::malloc(ntc * sizeof(float)));
  float *h_weight = static_cast<float *>(std::malloc(csz * sizeof(float)));
  float *h_bias = static_cast<float *>(std::malloc(csz * sizeof(float)));
  float *h_residual = static_cast<float *>(std::malloc(ntc * sizeof(float)));
  float *h_normed = static_cast<float *>(std::malloc(ntc * sizeof(float)));
  float *h_mean = static_cast<float *>(std::malloc(nt * sizeof(float)));
  float *h_rstd = static_cast<float *>(std::malloc(nt * sizeof(float)));
  if (!h_inp1 || !h_inp2 || !h_weight || !h_bias || !h_residual || !h_normed || !h_mean || !h_rstd) {
    std::fprintf(stderr, "host alloc failed\n");
    return 2;
  }
  auto read_chunk = [&](float *dst, std::size_t n, const char *what) -> bool {
    if (std::fread(dst, sizeof(float), n, fin) != n) {
      std::fprintf(stderr, "input too short at %s\n", what);
      return false;
    }
    return true;
  };
  if (!read_chunk(h_inp1, ntc, "inp1")) return 2;
  if (!read_chunk(h_inp2, ntc, "inp2")) return 2;
  if (!read_chunk(h_weight, csz, "weight")) return 2;
  if (!read_chunk(h_bias, csz, "bias")) return 2;
  std::fclose(fin);

  FILE *fout = std::fopen(argv[2], "wb");
  if (!fout) {
    std::perror(argv[2]);
    return 2;
  }

  sycl::queue q{sycl::default_selector_v};
  float *d_inp1 = sycl::malloc_device<float>(ntc, q);
  float *d_inp2 = sycl::malloc_device<float>(ntc, q);
  float *d_weight = sycl::malloc_device<float>(csz, q);
  float *d_bias = sycl::malloc_device<float>(csz, q);
  float *d_residual = sycl::malloc_device<float>(ntc, q);
  float *d_normed = sycl::malloc_device<float>(ntc, q);
  float *d_mean = sycl::malloc_device<float>(nt, q);
  float *d_rstd = sycl::malloc_device<float>(nt, q);
  if (!d_inp1 || !d_inp2 || !d_weight || !d_bias || !d_residual || !d_normed || !d_mean || !d_rstd) {
    std::fprintf(stderr, "device alloc failed\n");
    return 2;
  }

  q.memcpy(d_inp1, h_inp1, ntc * sizeof(float)).wait();
  q.memcpy(d_inp2, h_inp2, ntc * sizeof(float)).wait();
  q.memcpy(d_weight, h_weight, csz * sizeof(float)).wait();
  q.memcpy(d_bias, h_bias, csz * sizeof(float)).wait();

  const int variants[] = {1, 2, 3, 4, 5, 6};
  for (int vi = 0; vi < 6; ++vi) {
    const int k = variants[vi];
    q.fill(d_residual, 0.0f, ntc).wait();
    q.fill(d_normed, 0.0f, ntc).wait();
    q.fill(d_mean, 0.0f, nt).wait();
    q.fill(d_rstd, 0.0f, nt).wait();
    run_variant(q, k, d_residual, d_normed, d_mean, d_rstd, d_inp1, d_inp2, d_weight, d_bias, N, C);
    q.memcpy(h_residual, d_residual, ntc * sizeof(float)).wait();
    std::fwrite(h_residual, sizeof(float), ntc, fout);
    q.memcpy(h_normed, d_normed, ntc * sizeof(float)).wait();
    std::fwrite(h_normed, sizeof(float), ntc, fout);
    q.memcpy(h_mean, d_mean, nt * sizeof(float)).wait();
    std::fwrite(h_mean, sizeof(float), nt, fout);
    q.memcpy(h_rstd, d_rstd, nt * sizeof(float)).wait();
    std::fwrite(h_rstd, sizeof(float), nt, fout);
  }

  sycl::free(d_inp1, q);
  sycl::free(d_inp2, q);
  sycl::free(d_weight, q);
  sycl::free(d_bias, q);
  sycl::free(d_residual, q);
  sycl::free(d_normed, q);
  sycl::free(d_mean, q);
  sycl::free(d_rstd, q);
  std::fclose(fout);
  std::free(h_inp1);
  std::free(h_inp2);
  std::free(h_weight);
  std::free(h_bias);
  std::free(h_residual);
  std::free(h_normed);
  std::free(h_mean);
  std::free(h_rstd);
  return 0;
}
