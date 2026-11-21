// Idiomatic SYCL 2020 port of fused_residual_forward (fp32 build).
// Keeps kernel variants 1..6 as distinct device kernels and preserves the
// harness contract: build/candidate <input> <output> B T C
#include <sycl/sycl.hpp>

#include <cstdio>
#include <cstdlib>

using floatX = float;

namespace {

template <typename T>
T ceil_div(T dividend, T divisor) {
  return (dividend + divisor - 1) / divisor;
}

constexpr float kEps = 1e-5f;

// --- Variant 1: two stages -----------------------------------------------
void fused_residual_forward1(sycl::queue &q, floatX *residual, floatX *normed,
                             floatX *mean, floatX *rstd, const floatX *inp1,
                             const floatX *inp2, const floatX *weight,
                             const floatX *bias, int N, int C) {
  const std::size_t NC = static_cast<std::size_t>(N) * static_cast<std::size_t>(C);
  // Stage A: elementwise residual, 1-D range.
  q.submit([&](sycl::handler &h) {
    h.parallel_for<class ResidualFwdK1>(
        sycl::range<1>(NC), [=](sycl::id<1> id) {
          std::size_t i = id[0];
          residual[i] = inp1[i] + inp2[i];
        });
  }).wait();
  // Stage B: layernorm, one work-item per token.
  q.submit([&](sycl::handler &h) {
    h.parallel_for<class LayernormFwdK1>(
        sycl::range<1>(static_cast<std::size_t>(N)), [=](sycl::id<1> id) {
          int idx = static_cast<int>(id[0]);
          const floatX *x = inp1; // placeholder to keep capture list simple
          (void)x;
          const floatX *row = residual + static_cast<std::size_t>(idx) * static_cast<std::size_t>(C);
          float m = 0.0f;
          for (int i = 0; i < C; ++i) {
            m += static_cast<float>(row[i]);
          }
          m /= static_cast<float>(C);
          float v = 0.0f;
          for (int i = 0; i < C; ++i) {
            float d = static_cast<float>(row[i]) - m;
            v += d * d;
          }
          v /= static_cast<float>(C);
          float s = 1.0f / sycl::sqrt(v + kEps);
          floatX *out = normed + static_cast<std::size_t>(idx) * static_cast<std::size_t>(C);
          for (int i = 0; i < C; ++i) {
            float n = s * (static_cast<float>(row[i]) - m);
            out[i] = n * static_cast<float>(weight[i]) + static_cast<float>(bias[i]);
          }
          mean[idx] = m;
          rstd[idx] = s;
        });
  }).wait();
}

// --- Variant 2: naive fused, one work-item per token ----------------------
void fused_residual_forward2(sycl::queue &q, floatX *residual, floatX *normed,
                             floatX *mean, floatX *rstd, const floatX *inp1,
                             const floatX *inp2, const floatX *weight,
                             const floatX *bias, int N, int C) {
  q.submit([&](sycl::handler &h) {
    h.parallel_for<class FusedResidualK2>(
        sycl::range<1>(static_cast<std::size_t>(N)), [=](sycl::id<1> id) {
          int idx = static_cast<int>(id[0]);
          floatX *res = residual + static_cast<std::size_t>(idx) * static_cast<std::size_t>(C);
          floatX *out_row = normed + static_cast<std::size_t>(idx) * static_cast<std::size_t>(C);
          const floatX *a = inp1 + static_cast<std::size_t>(idx) * static_cast<std::size_t>(C);
          const floatX *b = inp2 + static_cast<std::size_t>(idx) * static_cast<std::size_t>(C);
          float m = 0.0f;
          for (int c = 0; c < C; ++c) {
            float v = static_cast<float>(a[c]) + static_cast<float>(b[c]);
            m += v;
            res[c] = v;
          }
          m /= static_cast<float>(C);
          float v = 0.0f;
          for (int c = 0; c < C; ++c) {
            float d = static_cast<float>(res[c]) - m;
            v += d * d;
          }
          v /= static_cast<float>(C);
          float s = 1.0f / sycl::sqrt(v + kEps);
          for (int c = 0; c < C; ++c) {
            float n = s * (static_cast<float>(res[c]) - m);
            out_row[c] = n * static_cast<float>(weight[c]) + static_cast<float>(bias[c]);
          }
          mean[idx] = m;
          rstd[idx] = s;
        });
  }).wait();
}

// --- Variant 3: nd_range, one work-item per token (coalesced viewpoint) ---
void fused_residual_forward3(sycl::queue &q, floatX *residual, floatX *normed,
                             floatX *mean, floatX *rstd, const floatX *inp1,
                             const floatX *inp2, const floatX *weight,
                             const floatX *bias, int N, int C) {
  constexpr std::size_t kWorkGroup = 128;
  std::size_t n = static_cast<std::size_t>(N);
  std::size_t n_groups = ceil_div(n, kWorkGroup);
  sycl::nd_range<1> launch(n_groups * kWorkGroup, kWorkGroup);
  q.submit([&](sycl::handler &h) {
    h.parallel_for<class FusedResidualK3>(launch, [=](sycl::nd_item<1> it) {
      std::size_t idx = it.get_global_id(0);
      if (idx >= n) {
        return;
      }
      floatX *res = residual + idx * static_cast<std::size_t>(C);
      floatX *out_row = normed + idx * static_cast<std::size_t>(C);
      const floatX *a = inp1 + idx * static_cast<std::size_t>(C);
      const floatX *b = inp2 + idx * static_cast<std::size_t>(C);
      float m = 0.0f;
      for (int c = 0; c < C; ++c) {
        float v = static_cast<float>(a[c]) + static_cast<float>(b[c]);
        m += v;
        res[c] = v;
      }
      m /= static_cast<float>(C);
      float var = 0.0f;
      for (int c = 0; c < C; ++c) {
        float d = static_cast<float>(res[c]) - m;
        var += d * d;
      }
      var /= static_cast<float>(C);
      float s = 1.0f / sycl::sqrt(var + kEps);
      for (int c = 0; c < C; ++c) {
        float nv = s * (static_cast<float>(res[c]) - m);
        out_row[c] = nv * static_cast<float>(weight[c]) + static_cast<float>(bias[c]);
      }
      mean[idx] = m;
      rstd[idx] = s;
    });
  }).wait();
}

// --- Variant 4: vectorized 128-bit style with unrolled loop ---------------
void fused_residual_forward4(sycl::queue &q, floatX *residual, floatX *normed,
                             floatX *mean, floatX *rstd, const floatX *inp1,
                             const floatX *inp2, const floatX *weight,
                             const floatX *bias, int N, int C) {
  q.submit([&](sycl::handler &h) {
    h.parallel_for<class FusedResidualK4>(
        sycl::range<1>(static_cast<std::size_t>(N)), [=](sycl::id<1> id) {
          int idx = static_cast<int>(id[0]);
          floatX *res = residual + static_cast<std::size_t>(idx) * static_cast<std::size_t>(C);
          floatX *out_row = normed + static_cast<std::size_t>(idx) * static_cast<std::size_t>(C);
          const floatX *a = inp1 + static_cast<std::size_t>(idx) * static_cast<std::size_t>(C);
          const floatX *b = inp2 + static_cast<std::size_t>(idx) * static_cast<std::size_t>(C);
          // Residual + mean accumulation, 4-wide unrolled (maps to 128-bit accesses).
          float sum = 0.0f;
          int c = 0;
          int c_vec_end = (C / 4) * 4;
          for (; c < c_vec_end; c += 4) {
            float v0 = static_cast<float>(a[c]) + static_cast<float>(b[c]);
            float v1 = static_cast<float>(a[c + 1]) + static_cast<float>(b[c + 1]);
            float v2 = static_cast<float>(a[c + 2]) + static_cast<float>(b[c + 2]);
            float v3 = static_cast<float>(a[c + 3]) + static_cast<float>(b[c + 3]);
            res[c] = v0;
            res[c + 1] = v1;
            res[c + 2] = v2;
            res[c + 3] = v3;
            sum += v0 + v1 + v2 + v3;
          }
          for (; c < C; ++c) {
            float v = static_cast<float>(a[c]) + static_cast<float>(b[c]);
            res[c] = v;
            sum += v;
          }
          float m = sum / static_cast<float>(C);
          // Stable two-pass variance, 4-wide unrolled.
          float var = 0.0f;
          c = 0;
          for (; c < c_vec_end; c += 4) {
            float d0 = static_cast<float>(res[c]) - m;
            float d1 = static_cast<float>(res[c + 1]) - m;
            float d2 = static_cast<float>(res[c + 2]) - m;
            float d3 = static_cast<float>(res[c + 3]) - m;
            var += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
          }
          for (; c < C; ++c) {
            float d = static_cast<float>(res[c]) - m;
            var += d * d;
          }
          var /= static_cast<float>(C);
          float s = sycl::rsqrt(var + kEps);
          c = 0;
          for (; c < c_vec_end; c += 4) {
            float n0 = s * (static_cast<float>(res[c]) - m);
            float n1 = s * (static_cast<float>(res[c + 1]) - m);
            float n2 = s * (static_cast<float>(res[c + 2]) - m);
            float n3 = s * (static_cast<float>(res[c + 3]) - m);
            out_row[c] = n0 * static_cast<float>(weight[c]) + static_cast<float>(bias[c]);
            out_row[c + 1] = n1 * static_cast<float>(weight[c + 1]) + static_cast<float>(bias[c + 1]);
            out_row[c + 2] = n2 * static_cast<float>(weight[c + 2]) + static_cast<float>(bias[c + 2]);
            out_row[c + 3] = n3 * static_cast<float>(weight[c + 3]) + static_cast<float>(bias[c + 3]);
          }
          for (; c < C; ++c) {
            float n = s * (static_cast<float>(res[c]) - m);
            out_row[c] = n * static_cast<float>(weight[c]) + static_cast<float>(bias[c]);
          }
          mean[idx] = m;
          rstd[idx] = s;
        });
  }).wait();
}

// --- Variant 5: work-group local cache of weight/bias ----------------------
void fused_residual_forward5(sycl::queue &q, floatX *residual, floatX *normed,
                             floatX *mean, floatX *rstd, const floatX *inp1,
                             const floatX *inp2, const floatX *weight,
                             const floatX *bias, int N, int C) {
  constexpr std::size_t kWorkGroup = 64;
  std::size_t n = static_cast<std::size_t>(N);
  std::size_t cu = static_cast<std::size_t>(C);
  std::size_t n_groups = ceil_div(n, kWorkGroup);
  sycl::nd_range<1> launch(n_groups * kWorkGroup, kWorkGroup);
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> w_local(cu, h);
    sycl::local_accessor<float, 1> b_local(cu, h);
    h.parallel_for<class FusedResidualK5>(launch, [=](sycl::nd_item<1> it) {
      sycl::group<1> grp = it.get_group();
      // Cooperatively stage scale/shift in local memory once per work-group.
      for (std::size_t c = it.get_local_id(0); c < cu; c += kWorkGroup) {
        w_local[c] = weight[c];
        b_local[c] = bias[c];
      }
      it.barrier(sycl::access::fence_space::local_space);
      std::size_t idx = it.get_global_id(0);
      if (idx >= n) {
        return;
      }
      int nn = static_cast<int>(idx);
      int cc = static_cast<int>(cu);
      floatX *res = residual + idx * cu;
      floatX *out_row = normed + idx * cu;
      const floatX *a = inp1 + idx * cu;
      const floatX *b = inp2 + idx * cu;
      float sum = 0.0f;
      for (int c = 0; c < cc; ++c) {
        float v = static_cast<float>(a[c]) + static_cast<float>(b[c]);
        res[c] = v;
        sum += v;
      }
      float m = sum / static_cast<float>(cc);
      float var = 0.0f;
      for (int c = 0; c < cc; ++c) {
        float d = static_cast<float>(res[c]) - m;
        var += d * d;
      }
      var /= static_cast<float>(cc);
      float s = sycl::rsqrt(var + kEps);
      for (int c = 0; c < cc; ++c) {
        float nv = s * (static_cast<float>(res[c]) - m);
        out_row[c] = nv * w_local[static_cast<std::size_t>(c)] + b_local[static_cast<std::size_t>(c)];
      }
      mean[nn] = m;
      rstd[nn] = s;
      (void)grp;
    });
  }).wait();
}

// --- Variant 6: persistent groups, local cache, grid-stride token loop -----
void fused_residual_forward6(sycl::queue &q, floatX *residual, floatX *normed,
                             floatX *mean, floatX *rstd, const floatX *inp1,
                             const floatX *inp2, const floatX *weight,
                             const floatX *bias, int N, int C) {
  constexpr std::size_t kWorkGroup = 64;
  // Small persistent grid that strides over tokens.
  std::size_t n = static_cast<std::size_t>(N);
  std::size_t cu = static_cast<std::size_t>(C);
  std::size_t n_groups = 4;
  if (n < n_groups * kWorkGroup) {
    n_groups = ceil_div(n, kWorkGroup);
    if (n_groups == 0) {
      n_groups = 1;
    }
  }
  std::size_t global = n_groups * kWorkGroup;
  sycl::nd_range<1> launch(global, kWorkGroup);
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> w_local(cu, h);
    sycl::local_accessor<float, 1> b_local(cu, h);
    h.parallel_for<class FusedResidualK6>(
        launch, [=](sycl::nd_item<1> it) {
          for (std::size_t c = it.get_local_id(0); c < cu; c += kWorkGroup) {
            w_local[c] = weight[c];
            b_local[c] = bias[c];
          }
          it.barrier(sycl::access::fence_space::local_space);
          int cc = static_cast<int>(cu);
          for (std::size_t idx = it.get_global_id(0); idx < n; idx += global) {
            floatX *res = residual + idx * cu;
            floatX *out_row = normed + idx * cu;
            const floatX *a = inp1 + idx * cu;
            const floatX *b = inp2 + idx * cu;
            float sum = 0.0f;
            for (int c = 0; c < cc; ++c) {
              float v = static_cast<float>(a[c]) + static_cast<float>(b[c]);
              res[c] = v;
              sum += v;
            }
            float m = sum / static_cast<float>(cc);
            float var = 0.0f;
            for (int c = 0; c < cc; ++c) {
              float d = static_cast<float>(res[c]) - m;
              var += d * d;
            }
            var /= static_cast<float>(cc);
            float s = sycl::rsqrt(var + kEps);
            for (int c = 0; c < cc; ++c) {
              float nv = s * (static_cast<float>(res[c]) - m);
              out_row[c] =
                  nv * w_local[static_cast<std::size_t>(c)] + b_local[static_cast<std::size_t>(c)];
            }
            mean[idx] = m;
            rstd[idx] = s;
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

}  // namespace

int main(int argc, char **argv) {
  if (argc != 6) {
    std::fprintf(stderr, "usage: %s <input> <output> B T C\n", argv[0]);
    return 2;
  }
  const int B = std::atoi(argv[3]);
  const int T = std::atoi(argv[4]);
  const int C = std::atoi(argv[5]);
  const std::size_t n_tokens = static_cast<std::size_t>(B) * static_cast<std::size_t>(T);
  const std::size_t n_elem = n_tokens * static_cast<std::size_t>(C);
  const std::size_t c_size = static_cast<std::size_t>(C);

  std::FILE *fin = std::fopen(argv[1], "rb");
  if (!fin) {
    std::perror(argv[1]);
    return 2;
  }
  float *inp1 = static_cast<float *>(std::malloc(sizeof(float) * n_elem));
  float *inp2 = static_cast<float *>(std::malloc(sizeof(float) * n_elem));
  float *weight = static_cast<float *>(std::malloc(sizeof(float) * c_size));
  float *bias = static_cast<float *>(std::malloc(sizeof(float) * c_size));
  if (!inp1 || !inp2 || !weight || !bias) {
    std::fprintf(stderr, "host malloc failed\n");
    return 2;
  }
  if (std::fread(inp1, sizeof(float), n_elem, fin) != n_elem) {
    std::fprintf(stderr, "input too short at inp1\n");
    return 2;
  }
  if (std::fread(inp2, sizeof(float), n_elem, fin) != n_elem) {
    std::fprintf(stderr, "input too short at inp2\n");
    return 2;
  }
  if (std::fread(weight, sizeof(float), c_size, fin) != c_size) {
    std::fprintf(stderr, "input too short at weight\n");
    return 2;
  }
  if (std::fread(bias, sizeof(float), c_size, fin) != c_size) {
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
  floatX *d_inp1 = sycl::malloc_device<floatX>(n_elem, q);
  floatX *d_inp2 = sycl::malloc_device<floatX>(n_elem, q);
  floatX *d_weight = sycl::malloc_device<floatX>(c_size, q);
  floatX *d_bias = sycl::malloc_device<floatX>(c_size, q);
  floatX *d_residual = sycl::malloc_device<floatX>(n_elem, q);
  floatX *d_normed = sycl::malloc_device<floatX>(n_elem, q);
  floatX *d_mean = sycl::malloc_device<floatX>(n_tokens, q);
  floatX *d_rstd = sycl::malloc_device<floatX>(n_tokens, q);
  if (!d_inp1 || !d_inp2 || !d_weight || !d_bias || !d_residual || !d_normed ||
      !d_mean || !d_rstd) {
    std::fprintf(stderr, "device malloc failed\n");
    return 2;
  }
  q.memcpy(d_inp1, inp1, sizeof(float) * n_elem).wait();
  q.memcpy(d_inp2, inp2, sizeof(float) * n_elem).wait();
  q.memcpy(d_weight, weight, sizeof(float) * c_size).wait();
  q.memcpy(d_bias, bias, sizeof(float) * c_size).wait();

  float *residual = static_cast<float *>(std::malloc(sizeof(float) * n_elem));
  float *normed = static_cast<float *>(std::malloc(sizeof(float) * n_elem));
  float *mean = static_cast<float *>(std::malloc(sizeof(float) * n_tokens));
  float *rstd = static_cast<float *>(std::malloc(sizeof(float) * n_tokens));
  if (!residual || !normed || !mean || !rstd) {
    std::fprintf(stderr, "host malloc failed\n");
    return 2;
  }

  const int variants[] = {1, 2, 3, 4, 5, 6};
  const int N = B * T;
  for (int vi = 0; vi < 6; ++vi) {
    const int k = variants[vi];
    q.fill(d_residual, floatX(0), n_elem).wait();
    q.fill(d_normed, floatX(0), n_elem).wait();
    q.fill(d_mean, floatX(0), n_tokens).wait();
    q.fill(d_rstd, floatX(0), n_tokens).wait();
    fused_residual_forward(q, k, d_residual, d_normed, d_mean, d_rstd, d_inp1,
                           d_inp2, d_weight, d_bias, N, C);
    q.memcpy(residual, d_residual, sizeof(float) * n_elem).wait();
    std::fwrite(residual, sizeof(float), n_elem, fout);
    q.memcpy(normed, d_normed, sizeof(float) * n_elem).wait();
    std::fwrite(normed, sizeof(float), n_elem, fout);
    q.memcpy(mean, d_mean, sizeof(float) * n_tokens).wait();
    std::fwrite(mean, sizeof(float), n_tokens, fout);
    q.memcpy(rstd, d_rstd, sizeof(float) * n_tokens).wait();
    std::fwrite(rstd, sizeof(float), n_tokens, fout);
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
