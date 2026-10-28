// Fused residual + layernorm forward, idiomatic SYCL 2020 (fp32).
// Six variants, each its own SYCL kernel, computed on the device.
// Contract: build/candidate <input> <output> B T C
#include <sycl/sycl.hpp>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstddef>
#include <algorithm>

using floatX = float;

template <typename T>
static inline T ceil_div(T dividend, T divisor) {
  return (dividend + divisor - 1) / divisor;
}

class K1ResidKernel;
class K1NormKernel;
class K2Kernel;
class K3Kernel;
class K4Kernel;
class K5Kernel;
class K6Kernel;

static size_t pick_local(sycl::queue &q, int hint) {
  size_t mx = q.get_device().get_info<sycl::info::device::max_work_group_size>();
  size_t l = (hint <= 0) ? 256 : static_cast<size_t>(hint);
  if (l > mx) l = mx;
  if (l == 0) l = 1;
  // round down to a multiple of subgroup-friendly power of two without
  // hardcoding any device subgroup width
  size_t p = 1;
  while (p * 2 <= l) p *= 2;
  return p;
}

// ---- variant 1: residual elementwise + per-row layernorm (one item per row)
static void fused_residual_forward1(sycl::queue &q, floatX *residual, floatX *normed,
                                    floatX *mean, floatX *rstd, const floatX *inp1,
                                    const floatX *inp2, const floatX *weight,
                                    const floatX *bias, int N, int C, int /*block*/) {
  size_t nc = static_cast<size_t>(N) * static_cast<size_t>(C);
  if (nc > 0) {
    q.parallel_for<K1ResidKernel>(sycl::range<1>(nc), [=](sycl::id<1> id) {
      size_t i = id[0];
      residual[i] = inp1[i] + inp2[i];
    }).wait();
  }
  if (N > 0) {
    q.parallel_for<K1NormKernel>(sycl::range<1>(static_cast<size_t>(N)), [=](sycl::id<1> id) {
      int idx = static_cast<int>(id[0]);
      const float eps = 1e-5f;
      const floatX *x = inp1; // placeholder to keep capture simple
      (void)x;
      const floatX *row = residual + static_cast<size_t>(idx) * static_cast<size_t>(C);
      float m = 0.0f;
      for (int i = 0; i < C; ++i) m += static_cast<float>(row[i]);
      m = m / static_cast<float>(C);
      float v = 0.0f;
      for (int i = 0; i < C; ++i) {
        float d = static_cast<float>(row[i]) - m;
        v += d * d;
      }
      v = v / static_cast<float>(C);
      float s = 1.0f / sycl::sqrt(v + eps);
      floatX *out = normed + static_cast<size_t>(idx) * static_cast<size_t>(C);
      for (int i = 0; i < C; ++i) {
        float n = s * (static_cast<float>(row[i]) - m);
        out[i] = n * static_cast<float>(weight[i]) + static_cast<float>(bias[i]);
      }
      mean[idx] = m;
      rstd[idx] = s;
    }).wait();
  }
}

// ---- variant 2: naive fused, one work-item per token, sequential loops
static void fused_residual_forward2(sycl::queue &q, floatX *residual, floatX *normed,
                                    floatX *mean, floatX *rstd, const floatX *inp1,
                                    const floatX *inp2, const floatX *weight,
                                    const floatX *bias, int N, int C, int /*block*/) {
  if (N <= 0) return;
  q.parallel_for<K2Kernel>(sycl::range<1>(static_cast<size_t>(N)), [=](sycl::id<1> id) {
    int idx = static_cast<int>(id[0]);
    if (idx >= N) return;
    floatX *res = residual + static_cast<size_t>(idx) * static_cast<size_t>(C);
    floatX *nrm = normed + static_cast<size_t>(idx) * static_cast<size_t>(C);
    const floatX *a = inp1 + static_cast<size_t>(idx) * static_cast<size_t>(C);
    const floatX *b = inp2 + static_cast<size_t>(idx) * static_cast<size_t>(C);
    const float eps = 1e-5f;
    float m = 0.0f;
    for (int c = 0; c < C; ++c) {
      float o = static_cast<float>(a[c]) + static_cast<float>(b[c]);
      m += o;
      res[c] = o;
    }
    m = m / static_cast<float>(C);
    float v = 0.0f;
    for (int c = 0; c < C; ++c) {
      float d = static_cast<float>(res[c]) - m;
      v += d * d;
    }
    v = v / static_cast<float>(C);
    float s = 1.0f / sycl::sqrt(v + eps);
    for (int c = 0; c < C; ++c) {
      float n = s * (static_cast<float>(res[c]) - m);
      nrm[c] = n * static_cast<float>(weight[c]) + static_cast<float>(bias[c]);
    }
    mean[idx] = m;
    rstd[idx] = s;
  }).wait();
}

// ---- variants 3,5,6: two-pass stable, one work-group per token, group reduction
template <typename KName>
static void launch_fused_twopass(sycl::queue &q, floatX *residual, floatX *normed,
                                 floatX *mean, floatX *rstd, const floatX *inp1,
                                 const floatX *inp2, const floatX *weight,
                                 const floatX *bias, int N, int C, int block_hint,
                                 bool use_rsqrt) {
  if (N <= 0) return;
  size_t L = pick_local(q, block_hint);
  size_t G = static_cast<size_t>(N) * L;
  q.parallel_for<KName>(sycl::nd_range<1>(G, L), [=](sycl::nd_item<1> it) {
    auto grp = it.get_group();
    int idx = static_cast<int>(grp.get_group_id(0));
    size_t lid = it.get_local_id(0);
    size_t ln = it.get_local_range(0);
    floatX *res_bt = residual + static_cast<size_t>(idx) * static_cast<size_t>(C);
    floatX *nrm_bt = normed + static_cast<size_t>(idx) * static_cast<size_t>(C);
    const floatX *a_bt = inp1 + static_cast<size_t>(idx) * static_cast<size_t>(C);
    const floatX *b_bt = inp2 + static_cast<size_t>(idx) * static_cast<size_t>(C);
    const float eps = 1e-5f;
    float part = 0.0f;
    for (int c = static_cast<int>(lid); c < C; c += static_cast<int>(ln)) {
      float o = static_cast<float>(a_bt[c]) + static_cast<float>(b_bt[c]);
      res_bt[c] = o;
      part += o;
    }
    float sum = sycl::reduce_over_group(grp, part, sycl::plus<float>());
    float m = sum / static_cast<float>(C);
    float partv = 0.0f;
    for (int c = static_cast<int>(lid); c < C; c += static_cast<int>(ln)) {
      float d = static_cast<float>(res_bt[c]) - m;
      partv += d * d;
    }
    float vsum = sycl::reduce_over_group(grp, partv, sycl::plus<float>());
    float v = vsum / static_cast<float>(C);
    float s = use_rsqrt ? sycl::rsqrt(v + eps) : (1.0f / sycl::sqrt(v + eps));
    for (int c = static_cast<int>(lid); c < C; c += static_cast<int>(ln)) {
      float n = s * (static_cast<float>(res_bt[c]) - m);
      nrm_bt[c] = n * static_cast<float>(weight[c]) + static_cast<float>(bias[c]);
    }
    if (lid == 0) {
      mean[idx] = m;
      rstd[idx] = s;
    }
  }).wait();
}

static void fused_residual_forward3(sycl::queue &q, floatX *residual, floatX *normed,
                                    floatX *mean, floatX *rstd, const floatX *inp1,
                                    const floatX *inp2, const floatX *weight,
                                    const floatX *bias, int N, int C, int block_hint) {
  launch_fused_twopass<K3Kernel>(q, residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C,
                                 block_hint, false);
}

// ---- variant 4: single-pass stats (sum + sumsq), zigzag-equivalent streaming
static void fused_residual_forward4(sycl::queue &q, floatX *residual, floatX *normed,
                                    floatX *mean, floatX *rstd, const floatX *inp1,
                                    const floatX *inp2, const floatX *weight,
                                    const floatX *bias, int N, int C, int block_hint) {
  if (N <= 0) return;
  size_t L = pick_local(q, block_hint);
  size_t G = static_cast<size_t>(N) * L;
  q.parallel_for<K4Kernel>(sycl::nd_range<1>(G, L), [=](sycl::nd_item<1> it) {
    auto grp = it.get_group();
    int idx = static_cast<int>(grp.get_group_id(0));
    size_t lid = it.get_local_id(0);
    size_t ln = it.get_local_range(0);
    floatX *res_bt = residual + static_cast<size_t>(idx) * static_cast<size_t>(C);
    floatX *nrm_bt = normed + static_cast<size_t>(idx) * static_cast<size_t>(C);
    const floatX *a_bt = inp1 + static_cast<size_t>(idx) * static_cast<size_t>(C);
    const floatX *b_bt = inp2 + static_cast<size_t>(idx) * static_cast<size_t>(C);
    const float eps = 1e-5f;
    float part = 0.0f;
    float part2 = 0.0f;
    for (int c = static_cast<int>(lid); c < C; c += static_cast<int>(ln)) {
      float o = static_cast<float>(a_bt[c]) + static_cast<float>(b_bt[c]);
      res_bt[c] = o;
      part += o;
      part2 += o * o;
    }
    float sum = sycl::reduce_over_group(grp, part, sycl::plus<float>());
    float sum2 = sycl::reduce_over_group(grp, part2, sycl::plus<float>());
    float m = sum / static_cast<float>(C);
    float v = sum2 / static_cast<float>(C) - m * m;
    float s = sycl::rsqrt(v + eps);
    for (int c = static_cast<int>(lid); c < C; c += static_cast<int>(ln)) {
      float n = s * (static_cast<float>(res_bt[c]) - m);
      nrm_bt[c] = n * static_cast<float>(weight[c]) + static_cast<float>(bias[c]);
    }
    if (lid == 0) {
      mean[idx] = m;
      rstd[idx] = s;
    }
  }).wait();
}

static void fused_residual_forward5(sycl::queue &q, floatX *residual, floatX *normed,
                                    floatX *mean, floatX *rstd, const floatX *inp1,
                                    const floatX *inp2, const floatX *weight,
                                    const floatX *bias, int N, int C, int block_hint) {
  launch_fused_twopass<K5Kernel>(q, residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C,
                                 block_hint, true);
}

static void fused_residual_forward6(sycl::queue &q, floatX *residual, floatX *normed,
                                    floatX *mean, floatX *rstd, const floatX *inp1,
                                    const floatX *inp2, const floatX *weight,
                                    const floatX *bias, int N, int C, int block_hint) {
  // persistent-style grouping is expressed as one work-group per token here;
  // the queue iterates over all tokens so every token is covered.
  launch_fused_twopass<K6Kernel>(q, residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C,
                                 block_hint, true);
}

static void fused_residual_forward(sycl::queue &q, int kernel_num, floatX *residual, floatX *normed,
                                   floatX *mean, floatX *rstd, const floatX *inp1,
                                   const floatX *inp2, const floatX *weight, const floatX *bias,
                                   int N, int C, int block_size) {
  switch (kernel_num) {
    case 1:
      fused_residual_forward1(q, residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C,
                              block_size);
      break;
    case 2:
      fused_residual_forward2(q, residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C,
                              block_size);
      break;
    case 3:
      fused_residual_forward3(q, residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C,
                              block_size);
      break;
    case 4:
      fused_residual_forward4(q, residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C,
                              block_size);
      break;
    case 5:
      fused_residual_forward5(q, residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C,
                              block_size);
      break;
    case 6:
      fused_residual_forward6(q, residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C,
                              block_size);
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
  const int N = B * T;
  std::FILE *fin = std::fopen(argv[1], "rb");
  if (!fin) {
    std::perror(argv[1]);
    return 2;
  }
  size_t ntc = static_cast<size_t>(B) * static_cast<size_t>(T) * static_cast<size_t>(C);
  size_t nt = static_cast<size_t>(B) * static_cast<size_t>(T);
  float *inp1 = static_cast<float *>(std::malloc(sizeof(float) * ntc));
  float *inp2 = static_cast<float *>(std::malloc(sizeof(float) * ntc));
  float *weight = static_cast<float *>(std::malloc(sizeof(float) * static_cast<size_t>(C)));
  float *bias = static_cast<float *>(std::malloc(sizeof(float) * static_cast<size_t>(C)));
  if (std::fread(inp1, sizeof(float), ntc, fin) != ntc) {
    std::fprintf(stderr, "input too short at inp1\n");
    return 2;
  }
  if (std::fread(inp2, sizeof(float), ntc, fin) != ntc) {
    std::fprintf(stderr, "input too short at inp2\n");
    return 2;
  }
  if (std::fread(weight, sizeof(float), static_cast<size_t>(C), fin) != static_cast<size_t>(C)) {
    std::fprintf(stderr, "input too short at weight\n");
    return 2;
  }
  if (std::fread(bias, sizeof(float), static_cast<size_t>(C), fin) != static_cast<size_t>(C)) {
    std::fprintf(stderr, "input too short at bias\n");
    return 2;
  }
  std::fclose(fin);
  std::FILE *fout = std::fopen(argv[2], "wb");
  if (!fout) {
    std::perror(argv[2]);
    return 2;
  }

  sycl::queue q;
  floatX *d_inp1 = sycl::malloc_device<floatX>(ntc, q);
  floatX *d_inp2 = sycl::malloc_device<floatX>(ntc, q);
  floatX *d_weight = sycl::malloc_device<floatX>(static_cast<size_t>(C), q);
  floatX *d_bias = sycl::malloc_device<floatX>(static_cast<size_t>(C), q);
  floatX *d_residual = sycl::malloc_device<floatX>(ntc, q);
  floatX *d_normed = sycl::malloc_device<floatX>(ntc, q);
  floatX *d_mean = sycl::malloc_device<floatX>(nt, q);
  floatX *d_rstd = sycl::malloc_device<floatX>(nt, q);
  float *residual = static_cast<float *>(std::malloc(sizeof(float) * ntc));
  float *normed = static_cast<float *>(std::malloc(sizeof(float) * ntc));
  float *mean = static_cast<float *>(std::malloc(sizeof(float) * nt));
  float *rstd = static_cast<float *>(std::malloc(sizeof(float) * nt));

  q.memcpy(d_inp1, inp1, sizeof(float) * ntc).wait();
  q.memcpy(d_inp2, inp2, sizeof(float) * ntc).wait();
  q.memcpy(d_weight, weight, sizeof(float) * static_cast<size_t>(C)).wait();
  q.memcpy(d_bias, bias, sizeof(float) * static_cast<size_t>(C)).wait();

  const int variants[] = {1, 2, 3, 4, 5, 6};
  for (int vi = 0; vi < 6; ++vi) {
    const int k = variants[vi];
    if (ntc > 0) {
      q.fill(d_residual, floatX(0), ntc).wait();
      q.fill(d_normed, floatX(0), ntc).wait();
    }
    if (nt > 0) {
      q.fill(d_mean, floatX(0), nt).wait();
      q.fill(d_rstd, floatX(0), nt).wait();
    }
    fused_residual_forward(q, k, d_residual, d_normed, d_mean, d_rstd, d_inp1, d_inp2, d_weight,
                           d_bias, N, C, 256);
    q.wait();
    if (ntc > 0) {
      q.memcpy(residual, d_residual, sizeof(float) * ntc).wait();
      q.memcpy(normed, d_normed, sizeof(float) * ntc).wait();
    }
    if (nt > 0) {
      q.memcpy(mean, d_mean, sizeof(float) * nt).wait();
      q.memcpy(rstd, d_rstd, sizeof(float) * nt).wait();
    }
    std::fwrite(residual, sizeof(float), ntc, fout);
    std::fwrite(normed, sizeof(float), ntc, fout);
    std::fwrite(mean, sizeof(float), nt, fout);
    std::fwrite(rstd, sizeof(float), nt, fout);
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
