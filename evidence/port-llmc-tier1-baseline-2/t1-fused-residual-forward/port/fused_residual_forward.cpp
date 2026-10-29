#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>

using floatX = float;

// Each kernel variant k in [1,6] is its own SYCL kernel (distinct kernel
// name tag) executed on the SYCL device. All launches are 1-D; reductions
// are expressed per work-item over C so no fixed sub-group size is assumed.

class ResidualK1;
class LayerNormK1;
class FusedK2;
class FusedK3;
class FusedK4;
class FusedK5;
class FusedK6;

static void fused_residual_forward1(sycl::queue &q, float *residual,
                                    float *normed, float *mean, float *rstd,
                                    const float *inp1, const float *inp2,
                                    const float *weight, const float *bias,
                                    int N, int C) {
  const size_t NC = static_cast<size_t>(N) * static_cast<size_t>(C);
  q.parallel_for<ResidualK1>(sycl::range<1>(NC), [=](sycl::id<1> id) {
     size_t idx = id[0];
     residual[idx] = inp1[idx] + inp2[idx];
   }).wait();
  q.parallel_for<LayerNormK1>(sycl::range<1>(static_cast<size_t>(N)),
                              [=](sycl::id<1> id) {
     int idx = static_cast<int>(id[0]);
     constexpr float eps = 1e-5f;
     const float *x = inp1; // placeholder to keep capture list uniform
     (void)x;
     const float *row = residual + static_cast<size_t>(idx) * static_cast<size_t>(C);
     float m = 0.0f;
     for (int i = 0; i < C; ++i) {
       m += row[i];
     }
     m = m / static_cast<float>(C);
     float v = 0.0f;
     for (int i = 0; i < C; ++i) {
       float d = row[i] - m;
       v += d * d;
     }
     v = v / static_cast<float>(C);
     float s = 1.0f / sycl::sqrt(v + eps);
     float *o = normed + static_cast<size_t>(idx) * static_cast<size_t>(C);
     for (int i = 0; i < C; ++i) {
       float n = s * (row[i] - m);
       o[i] = n * weight[i] + bias[i];
     }
     mean[idx] = m;
     rstd[idx] = s;
   }).wait();
}

static void fused_residual_forward2(sycl::queue &q, float *residual,
                                    float *normed, float *mean, float *rstd,
                                    const float *inp1, const float *inp2,
                                    const float *weight, const float *bias,
                                    int N, int C) {
  q.parallel_for<FusedK2>(sycl::range<1>(static_cast<size_t>(N)),
                          [=](sycl::id<1> id) {
     int idx = static_cast<int>(id[0]);
     constexpr float eps = 1e-5f;
     float *res = residual + static_cast<size_t>(idx) * static_cast<size_t>(C);
     float *out = normed + static_cast<size_t>(idx) * static_cast<size_t>(C);
     const float *a = inp1 + static_cast<size_t>(idx) * static_cast<size_t>(C);
     const float *b = inp2 + static_cast<size_t>(idx) * static_cast<size_t>(C);
     float m = 0.0f;
     for (int c = 0; c < C; ++c) {
       float t = a[c] + b[c];
       m += t;
       res[c] = t;
     }
     m = m / static_cast<float>(C);
     float v = 0.0f;
     for (int c = 0; c < C; ++c) {
       float d = res[c] - m;
       v += d * d;
     }
     v = v / static_cast<float>(C);
     float s = 1.0f / sycl::sqrt(v + eps);
     for (int c = 0; c < C; ++c) {
       float n = s * (res[c] - m);
       out[c] = n * weight[c] + bias[c];
     }
     mean[idx] = m;
     rstd[idx] = s;
   }).wait();
}

// k3: one token per cooperative group in CUDA; here one work-item per row
// with a sequential walk over C (portable, coalescing left to the compiler).
static void fused_residual_forward3(sycl::queue &q, float *residual,
                                    float *normed, float *mean, float *rstd,
                                    const float *inp1, const float *inp2,
                                    const float *weight, const float *bias,
                                    int N, int C) {
  q.parallel_for<FusedK3>(sycl::range<1>(static_cast<size_t>(N)),
                          [=](sycl::id<1> id) {
     int idx = static_cast<int>(id[0]);
     constexpr float eps = 1e-5f;
     float *res = residual + static_cast<size_t>(idx) * static_cast<size_t>(C);
     float *out = normed + static_cast<size_t>(idx) * static_cast<size_t>(C);
     const float *a = inp1 + static_cast<size_t>(idx) * static_cast<size_t>(C);
     const float *b = inp2 + static_cast<size_t>(idx) * static_cast<size_t>(C);
     float acc = 0.0f;
     for (int c = 0; c < C; ++c) {
       float t = a[c] + b[c];
       res[c] = t;
       acc += t;
     }
     float m = acc / static_cast<float>(C);
     float var = 0.0f;
     for (int c = 0; c < C; ++c) {
       float d = res[c] - m;
       var += d * d;
     }
     var = var / static_cast<float>(C);
     float s = 1.0f / sycl::sqrt(var + eps);
     for (int c = 0; c < C; ++c) {
       float n = s * (res[c] - m);
       out[c] = n * weight[c] + bias[c];
     }
     mean[idx] = m;
     rstd[idx] = s;
   }).wait();
}

// k4: single-pass mean/variance from sum and sum-of-squares.
static void fused_residual_forward4(sycl::queue &q, float *residual,
                                    float *normed, float *mean, float *rstd,
                                    const float *inp1, const float *inp2,
                                    const float *weight, const float *bias,
                                    int N, int C) {
  q.parallel_for<FusedK4>(sycl::range<1>(static_cast<size_t>(N)),
                          [=](sycl::id<1> id) {
     int idx = static_cast<int>(id[0]);
     constexpr float eps = 1e-5f;
     float *res = residual + static_cast<size_t>(idx) * static_cast<size_t>(C);
     float *out = normed + static_cast<size_t>(idx) * static_cast<size_t>(C);
     const float *a = inp1 + static_cast<size_t>(idx) * static_cast<size_t>(C);
     const float *b = inp2 + static_cast<size_t>(idx) * static_cast<size_t>(C);
     float sum = 0.0f;
     float sum_sq = 0.0f;
     for (int c = 0; c < C; ++c) {
       float t = a[c] + b[c];
       res[c] = t;
       sum += t;
       sum_sq += t * t;
     }
     float m = sum / static_cast<float>(C);
     float v = sum_sq / static_cast<float>(C) - m * m;
     float s = sycl::rsqrt(v + eps);
     for (int c = 0; c < C; ++c) {
       float n = s * (res[c] - m);
       out[c] = n * weight[c] + bias[c];
     }
     mean[idx] = m;
     rstd[idx] = s;
   }).wait();
}

// k5: numerically stable two-pass pass over a per-row walk (shared-memory
// staging in CUDA is unnecessary for correctness; the compiler caches).
static void fused_residual_forward5(sycl::queue &q, float *residual,
                                    float *normed, float *mean, float *rstd,
                                    const float *inp1, const float *inp2,
                                    const float *weight, const float *bias,
                                    int N, int C) {
  q.parallel_for<FusedK5>(sycl::range<1>(static_cast<size_t>(N)),
                          [=](sycl::id<1> id) {
     int idx = static_cast<int>(id[0]);
     constexpr float eps = 1e-5f;
     float *res = residual + static_cast<size_t>(idx) * static_cast<size_t>(C);
     float *out = normed + static_cast<size_t>(idx) * static_cast<size_t>(C);
     const float *a = inp1 + static_cast<size_t>(idx) * static_cast<size_t>(C);
     const float *b = inp2 + static_cast<size_t>(idx) * static_cast<size_t>(C);
     float total = 0.0f;
     for (int c = 0; c < C; ++c) {
       float t = a[c] + b[c];
       res[c] = t;
       total += t;
     }
     float m = total / static_cast<float>(C);
     float v = 0.0f;
     for (int c = 0; c < C; ++c) {
       float d = res[c] - m;
       v += d * d;
     }
     v = v / static_cast<float>(C);
     float s = sycl::rsqrt(v + eps);
     for (int c = 0; c < C; ++c) {
       float n = s * (res[c] - m);
       out[c] = n * weight[c] + bias[c];
     }
     mean[idx] = m;
     rstd[idx] = s;
   }).wait();
}

// k6: persistent-token idea in CUDA; here each work-item still owns one row,
// which is the portable 1-D spelling with identical numerics.
static void fused_residual_forward6(sycl::queue &q, float *residual,
                                    float *normed, float *mean, float *rstd,
                                    const float *inp1, const float *inp2,
                                    const float *weight, const float *bias,
                                    int N, int C) {
  q.parallel_for<FusedK6>(sycl::range<1>(static_cast<size_t>(N)),
                          [=](sycl::id<1> id) {
     int idx = static_cast<int>(id[0]);
     constexpr float eps = 1e-5f;
     float *res = residual + static_cast<size_t>(idx) * static_cast<size_t>(C);
     float *out = normed + static_cast<size_t>(idx) * static_cast<size_t>(C);
     const float *a = inp1 + static_cast<size_t>(idx) * static_cast<size_t>(C);
     const float *b = inp2 + static_cast<size_t>(idx) * static_cast<size_t>(C);
     float sum = 0.0f;
     for (int c = 0; c < C; ++c) {
       float t = a[c] + b[c];
       res[c] = t;
       sum += t;
     }
     float m = sum / static_cast<float>(C);
     float v = 0.0f;
     for (int c = 0; c < C; ++c) {
       float d = res[c] - m;
       v += d * d;
     }
     v = v / static_cast<float>(C);
     float s = 1.0f / sycl::sqrt(v + eps);
     for (int c = 0; c < C; ++c) {
       float n = s * (res[c] - m);
       out[c] = n * weight[c] + bias[c];
     }
     mean[idx] = m;
     rstd[idx] = s;
   }).wait();
}

static void fused_residual_forward(sycl::queue &q, int kernel_num,
                                   float *residual, float *normed, float *mean,
                                   float *rstd, const float *inp1,
                                   const float *inp2, const float *weight,
                                   const float *bias, int N, int C) {
  switch (kernel_num) {
    case 1:
      fused_residual_forward1(q, residual, normed, mean, rstd, inp1, inp2,
                              weight, bias, N, C);
      break;
    case 2:
      fused_residual_forward2(q, residual, normed, mean, rstd, inp1, inp2,
                              weight, bias, N, C);
      break;
    case 3:
      fused_residual_forward3(q, residual, normed, mean, rstd, inp1, inp2,
                              weight, bias, N, C);
      break;
    case 4:
      fused_residual_forward4(q, residual, normed, mean, rstd, inp1, inp2,
                              weight, bias, N, C);
      break;
    case 5:
      fused_residual_forward5(q, residual, normed, mean, rstd, inp1, inp2,
                              weight, bias, N, C);
      break;
    case 6:
      fused_residual_forward6(q, residual, normed, mean, rstd, inp1, inp2,
                              weight, bias, N, C);
      break;
    default:
      std::fprintf(stderr, "Invalid kernel number\n");
      std::exit(1);
  }
}

int main(int argc, char **argv) {
  if (argc != 6) {
    std::fprintf(stderr, "usage: %s <input> <output> B T C\n", argv[0]);
    return 2;
  }
  const int B = std::atoi(argv[3]);
  const int T = std::atoi(argv[4]);
  const int C = std::atoi(argv[5]);
  const size_t BTC = static_cast<size_t>(B) * static_cast<size_t>(T) *
                     static_cast<size_t>(C);
  const size_t BT = static_cast<size_t>(B) * static_cast<size_t>(T);

  FILE *fin = std::fopen(argv[1], "rb");
  if (!fin) {
    std::perror(argv[1]);
    return 2;
  }
  float *inp1 = (float *)std::malloc(sizeof(float) * BTC);
  float *inp2 = (float *)std::malloc(sizeof(float) * BTC);
  float *weight = (float *)std::malloc(sizeof(float) * static_cast<size_t>(C));
  float *bias = (float *)std::malloc(sizeof(float) * static_cast<size_t>(C));
  if (std::fread(inp1, sizeof(float), BTC, fin) != BTC) {
    std::fprintf(stderr, "input too short at inp1\n");
    return 2;
  }
  if (std::fread(inp2, sizeof(float), BTC, fin) != BTC) {
    std::fprintf(stderr, "input too short at inp2\n");
    return 2;
  }
  if (std::fread(weight, sizeof(float), static_cast<size_t>(C), fin) !=
      static_cast<size_t>(C)) {
    std::fprintf(stderr, "input too short at weight\n");
    return 2;
  }
  if (std::fread(bias, sizeof(float), static_cast<size_t>(C), fin) !=
      static_cast<size_t>(C)) {
    std::fprintf(stderr, "input too short at bias\n");
    return 2;
  }
  std::fclose(fin);

  FILE *fout = std::fopen(argv[2], "wb");
  if (!fout) {
    std::perror(argv[2]);
    return 2;
  }

  sycl::queue q{sycl::default_selector_v};

  float *d_inp1 = sycl::malloc_device<float>(BTC, q);
  float *d_inp2 = sycl::malloc_device<float>(BTC, q);
  float *d_weight =
      sycl::malloc_device<float>(static_cast<size_t>(C), q);
  float *d_bias = sycl::malloc_device<float>(static_cast<size_t>(C), q);
  float *d_residual = sycl::malloc_device<float>(BTC, q);
  float *d_normed = sycl::malloc_device<float>(BTC, q);
  float *d_mean = sycl::malloc_device<float>(BT, q);
  float *d_rstd = sycl::malloc_device<float>(BT, q);

  q.memcpy(d_inp1, inp1, sizeof(float) * BTC).wait();
  q.memcpy(d_inp2, inp2, sizeof(float) * BTC).wait();
  q.memcpy(d_weight, weight, sizeof(float) * static_cast<size_t>(C)).wait();
  q.memcpy(d_bias, bias, sizeof(float) * static_cast<size_t>(C)).wait();

  float *residual = (float *)std::malloc(sizeof(float) * BTC);
  float *normed = (float *)std::malloc(sizeof(float) * BTC);
  float *mean = (float *)std::malloc(sizeof(float) * BT);
  float *rstd = (float *)std::malloc(sizeof(float) * BT);

  const int variants[] = {1, 2, 3, 4, 5, 6};
  for (int vi = 0; vi < 6; ++vi) {
    const int k = variants[vi];
    q.fill(d_residual, 0.0f, BTC).wait();
    q.fill(d_normed, 0.0f, BTC).wait();
    q.fill(d_mean, 0.0f, BT).wait();
    q.fill(d_rstd, 0.0f, BT).wait();
    fused_residual_forward(q, k, d_residual, d_normed, d_mean, d_rstd, d_inp1,
                           d_inp2, d_weight, d_bias, B * T, C);
    q.memcpy(residual, d_residual, sizeof(float) * BTC).wait();
    std::fwrite(residual, sizeof(float), BTC, fout);
    q.memcpy(normed, d_normed, sizeof(float) * BTC).wait();
    std::fwrite(normed, sizeof(float), BTC, fout);
    q.memcpy(mean, d_mean, sizeof(float) * BT).wait();
    std::fwrite(mean, sizeof(float), BT, fout);
    q.memcpy(rstd, d_rstd, sizeof(float) * BT).wait();
    std::fwrite(rstd, sizeof(float), BT, fout);
  }

  std::fclose(fout);
  sycl::free(d_inp1, q);
  sycl::free(d_inp2, q);
  sycl::free(d_weight, q);
  sycl::free(d_bias, q);
  sycl::free(d_residual, q);
  sycl::free(d_normed, q);
  sycl::free(d_mean, q);
  sycl::free(d_rstd, q);
  std::free(inp1);
  std::free(inp2);
  std::free(weight);
  std::free(bias);
  std::free(residual);
  std::free(normed);
  std::free(mean);
  std::free(rstd);
  return 0;
}
