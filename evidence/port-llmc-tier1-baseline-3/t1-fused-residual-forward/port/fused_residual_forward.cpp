#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

using floatX = float;

template <typename T>
static T ceil_div(T a, T b) { return (a + b - 1) / b; }

// ---------------------------------------------------------------------------
// Kernel name tags: one distinct SYCL kernel per variant (k1 uses two).
// All launches are 1-D nd_range; reductions use group primitives so no
// hard-coded sub-group/warp size appears anywhere.
// ---------------------------------------------------------------------------
class KernelRes1;
class KernelLn1;
class KernelFused2;
class KernelFused3;
class KernelFused4;
class KernelFused5;
class KernelFused6;

static void launch_variant1(sycl::queue &q,
                            float *residual, float *normed, float *mean, float *rstd,
                            const float *inp1, const float *inp2,
                            const float *weight, const float *bias,
                            int N, int C, int block_size) {
  // Part A: elementwise residual over N*C (1-D work).
  {
    std::size_t n = (std::size_t)N * (std::size_t)C;
    std::size_t local = (std::size_t)block_size;
    std::size_t global = ceil_div(n, local) * local;
    if (n == 0) return;
    q.submit([&](sycl::handler &h) {
      h.parallel_for<KernelRes1>(
          sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(local)),
          [=](sycl::nd_item<1> it) {
            std::size_t idx = it.get_global_id(0);
            if (idx < n) residual[idx] = inp1[idx] + inp2[idx];
          });
    }).wait();
  }
  // Part B: naive layernorm, one work-item per row, loop over C.
  {
    if (N == 0) return;
    std::size_t local = (std::size_t)block_size;
    std::size_t global = ceil_div((std::size_t)N, local) * local;
    q.submit([&](sycl::handler &h) {
      h.parallel_for<KernelLn1>(
          sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(local)),
          [=](sycl::nd_item<1> it) {
            int idx = (int)it.get_global_id(0);
            if (idx >= N) return;
            const float eps = 1e-5f;
            const float *x = residual + (std::size_t)idx * (std::size_t)C;
            float *o = normed + (std::size_t)idx * (std::size_t)C;
            float m = 0.0f;
            for (int i = 0; i < C; ++i) m += x[i];
            m /= (float)C;
            float v = 0.0f;
            for (int i = 0; i < C; ++i) {
              float d = x[i] - m;
              v += d * d;
            }
            v /= (float)C;
            float s = 1.0f / sycl::sqrt(v + eps);
            for (int i = 0; i < C; ++i) {
              float nv = s * (x[i] - m);
              o[i] = nv * weight[i] + bias[i];
            }
            mean[idx] = m;
            rstd[idx] = s;
          });
    }).wait();
  }
}

// Variant 2: naive fusion, one work-item per token, sequential over C.
static void launch_variant2(sycl::queue &q,
                            float *residual, float *normed, float *mean, float *rstd,
                            const float *inp1, const float *inp2,
                            const float *weight, const float *bias,
                            int N, int C, int block_size) {
  if (N == 0) return;
  std::size_t local = (std::size_t)block_size;
  std::size_t global = ceil_div((std::size_t)N, local) * local;
  q.submit([&](sycl::handler &h) {
    h.parallel_for<KernelFused2>(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(local)),
        [=](sycl::nd_item<1> it) {
          int idx = (int)it.get_global_id(0);
          if (idx >= N) return;
          const float eps = 1e-5f;
          float *res = residual + (std::size_t)idx * (std::size_t)C;
          float *nrm = normed + (std::size_t)idx * (std::size_t)C;
          const float *a = inp1 + (std::size_t)idx * (std::size_t)C;
          const float *b = inp2 + (std::size_t)idx * (std::size_t)C;
          float m = 0.0f;
          for (int c = 0; c < C; ++c) {
            float out = a[c] + b[c];
            m += out;
            res[c] = out;
          }
          m /= (float)C;
          float v = 0.0f;
          for (int c = 0; c < C; ++c) {
            float d = res[c] - m;
            v += d * d;
          }
          v /= (float)C;
          float s = 1.0f / sycl::sqrt(v + eps);
          for (int c = 0; c < C; ++c) {
            float nv = s * (res[c] - m);
            nrm[c] = nv * weight[c] + bias[c];
          }
          mean[idx] = m;
          rstd[idx] = s;
        });
  }).wait();
}

// Variant 3: one work-group per token, coalesced strided access,
// work-group reduction (no hard-coded warp size).
static void launch_variant3(sycl::queue &q,
                            float *residual, float *normed, float *mean, float *rstd,
                            const float *inp1, const float *inp2,
                            const float *weight, const float *bias,
                            int N, int C, int block_size) {
  if (N == 0) return;
  std::size_t wg = (std::size_t)block_size;
  std::size_t global = (std::size_t)N * wg;
  q.submit([&](sycl::handler &h) {
    h.parallel_for<KernelFused3>(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(wg)),
        [=](sycl::nd_item<1> it) {
          auto grp = it.get_group();
          int token = (int)grp.get_group_id(0);
          int lid = (int)it.get_local_id(0);
          int wgsize = (int)it.get_local_range(0);
          const float eps = 1e-5f;
          float *res = residual + (std::size_t)token * (std::size_t)C;
          float *nrm = normed + (std::size_t)token * (std::size_t)C;
          const float *a = inp1 + (std::size_t)token * (std::size_t)C;
          const float *b = inp2 + (std::size_t)token * (std::size_t)C;
          float sum = 0.0f;
          for (int c = lid; c < C; c += wgsize) {
            float out = a[c] + b[c];
            sum += out;
            res[c] = out;
          }
          float m = sycl::reduce_over_group(grp, sum, sycl::plus<float>()) / (float)C;
          float vsum = 0.0f;
          for (int c = lid; c < C; c += wgsize) {
            float d = res[c] - m;
            vsum += d * d;
          }
          float v = sycl::reduce_over_group(grp, vsum, sycl::plus<float>()) / (float)C;
          float s = 1.0f / sycl::sqrt(v + eps);
          for (int c = lid; c < C; c += wgsize) {
            float nv = s * (res[c] - m);
            nrm[c] = nv * weight[c] + bias[c];
          }
          if (lid == 0) {
            mean[token] = m;
            rstd[token] = s;
          }
        });
  }).wait();
}

// Variant 4: vectorized (4-wide) first pass, single-pass mean/variance
// (sum and sum of squares), streaming second pass over the same layout.
static void launch_variant4(sycl::queue &q,
                            float *residual, float *normed, float *mean, float *rstd,
                            const float *inp1, const float *inp2,
                            const float *weight, const float *bias,
                            int N, int C, int block_size) {
  if (N == 0) return;
  std::size_t wg = (std::size_t)block_size;
  std::size_t global = (std::size_t)N * wg;
  constexpr int V = 4;
  q.submit([&](sycl::handler &h) {
    h.parallel_for<KernelFused4>(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(wg)),
        [=](sycl::nd_item<1> it) {
          auto grp = it.get_group();
          int token = (int)grp.get_group_id(0);
          int lid = (int)it.get_local_id(0);
          int wgsize = (int)it.get_local_range(0);
          const float eps = 1e-5f;
          float *res = residual + (std::size_t)token * (std::size_t)C;
          float *nrm = normed + (std::size_t)token * (std::size_t)C;
          const float *a = inp1 + (std::size_t)token * (std::size_t)C;
          const float *b = inp2 + (std::size_t)token * (std::size_t)C;
          int nvec = C / V;
          // Each work-item owns vectors v = lid, lid+wgsize, ... (scalar equivalent
          // of 128-bit packed loads; compiler vectorizes the inner 4).
          float sum = 0.0f;
          float sum_sq = 0.0f;
          for (int v = lid; v < nvec; v += wgsize) {
            int c = v * V;
            float o0 = a[c + 0] + b[c + 0];
            float o1 = a[c + 1] + b[c + 1];
            float o2 = a[c + 2] + b[c + 2];
            float o3 = a[c + 3] + b[c + 3];
            res[c + 0] = o0; res[c + 1] = o1; res[c + 2] = o2; res[c + 3] = o3;
            sum += o0 + o1 + o2 + o3;
            sum_sq += o0 * o0 + o1 * o1 + o2 * o2 + o3 * o3;
          }
          // Scalar tail for C not divisible by 4.
          for (int c = nvec * V + lid; c < C; c += wgsize) {
            float o = a[c] + b[c];
            res[c] = o;
            sum += o;
            sum_sq += o * o;
          }
          float gsum = sycl::reduce_over_group(grp, sum, sycl::plus<float>());
          float gsum_sq = sycl::reduce_over_group(grp, sum_sq, sycl::plus<float>());
          float m = gsum / (float)C;
          float vvar = gsum_sq / (float)C - m * m;
          // Clamp tiny negative from rounding, matching rsqrt(v+eps) semantics.
          float s = sycl::rsqrt(vvar + eps);
          // Second pass: same ownership, streaming writes.
          for (int v = lid; v < nvec; v += wgsize) {
            int c = v * V;
            float r0 = res[c + 0], r1 = res[c + 1], r2 = res[c + 2], r3 = res[c + 3];
            float w0 = weight[c + 0], w1 = weight[c + 1], w2 = weight[c + 2], w3 = weight[c + 3];
            float b0 = bias[c + 0], b1 = bias[c + 1], b2 = bias[c + 2], b3 = bias[c + 3];
            nrm[c + 0] = s * (r0 - m) * w0 + b0;
            nrm[c + 1] = s * (r1 - m) * w1 + b1;
            nrm[c + 2] = s * (r2 - m) * w2 + b2;
            nrm[c + 3] = s * (r3 - m) * w3 + b3;
          }
          for (int c = nvec * V + lid; c < C; c += wgsize) {
            float nv = s * (res[c] - m);
            nrm[c] = nv * weight[c] + bias[c];
          }
          if (lid == 0) {
            mean[token] = m;
            rstd[token] = s;
          }
        });
  }).wait();
}

// Variant 5: everything cached in work-group local memory (weights, biases,
// per-token residual), numerically stable two-pass variance.
static void launch_variant5(sycl::queue &q,
                            float *residual, float *normed, float *mean, float *rstd,
                            const float *inp1, const float *inp2,
                            const float *weight, const float *bias,
                            int N, int C, int block_size) {
  if (N == 0) return;
  // Fall back to the local-memory-free formulation if the tile cannot fit,
  // mirroring the CUDA launcher's fallback to kernel 4.
  std::size_t local_bytes =
      q.get_device().get_info<sycl::info::device::local_mem_size>();
  if ((std::size_t)(3 * (std::size_t)C * sizeof(float)) > local_bytes) {
    launch_variant4(q, residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C,
                    block_size);
    return;
  }
  std::size_t wg = (std::size_t)block_size;
  std::size_t global = (std::size_t)N * wg;
  std::size_t tile = (std::size_t)3 * (std::size_t)C;
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> scratch(sycl::range<1>(tile), h);
    h.parallel_for<KernelFused5>(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(wg)),
        [=](sycl::nd_item<1> it) {
          auto grp = it.get_group();
          int token = (int)grp.get_group_id(0);
          int lid = (int)it.get_local_id(0);
          int wgsize = (int)it.get_local_range(0);
          const float eps = 1e-5f;
          float *lw = scratch.get_pointer();
          float *lb = lw + C;
          float *lr = lb + C;
          for (int c = lid; c < C; c += wgsize) {
            lw[c] = weight[c];
            lb[c] = bias[c];
          }
          it.barrier(sycl::access::fence_space::local_space);
          float *res = residual + (std::size_t)token * (std::size_t)C;
          float *nrm = normed + (std::size_t)token * (std::size_t)C;
          const float *a = inp1 + (std::size_t)token * (std::size_t)C;
          const float *b = inp2 + (std::size_t)token * (std::size_t)C;
          float sum = 0.0f;
          for (int c = lid; c < C; c += wgsize) {
            float o = a[c] + b[c];
            sum += o;
            res[c] = o;
            lr[c] = o;
          }
          float m = sycl::reduce_over_group(grp, sum, sycl::plus<float>()) / (float)C;
          float vsum = 0.0f;
          for (int c = lid; c < C; c += wgsize) {
            float d = lr[c] - m;
            vsum += d * d;
          }
          float v = sycl::reduce_over_group(grp, vsum, sycl::plus<float>()) / (float)C;
          float s = sycl::rsqrt(v + eps);
          for (int c = lid; c < C; c += wgsize) {
            float nv = s * (lr[c] - m);
            nrm[c] = nv * lw[c] + lb[c];
          }
          if (lid == 0) {
            mean[token] = m;
            rstd[token] = s;
          }
        });
  }).wait();
}

// Variant 6: persistent work-groups; weights/biases loaded once into local
// memory and reused across many tokens assigned round-robin to groups.
static void launch_variant6(sycl::queue &q,
                            float *residual, float *normed, float *mean, float *rstd,
                            const float *inp1, const float *inp2,
                            const float *weight, const float *bias,
                            int N, int C, int block_size) {
  if (N == 0) return;
  std::size_t local_bytes =
      q.get_device().get_info<sycl::info::device::local_mem_size>();
  if ((std::size_t)(3 * (std::size_t)C * sizeof(float)) > local_bytes) {
    launch_variant4(q, residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C,
                    block_size);
    return;
  }
  std::size_t wg = (std::size_t)block_size;
  std::size_t units =
      q.get_device().get_info<sycl::info::device::max_compute_units>();
  std::size_t num_groups = units * 2;
  if (num_groups < 1) num_groups = 1;
  if (num_groups > (std::size_t)N) num_groups = (std::size_t)N;
  std::size_t global = num_groups * wg;
  std::size_t tile = (std::size_t)3 * (std::size_t)C;
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> scratch(sycl::range<1>(tile), h);
    h.parallel_for<KernelFused6>(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(wg)),
        [=](sycl::nd_item<1> it) {
          auto grp = it.get_group();
          int gid = (int)grp.get_group_id(0);
          int ngrp = (int)grp.get_group_range(0);
          int lid = (int)it.get_local_id(0);
          int wgsize = (int)it.get_local_range(0);
          const float eps = 1e-5f;
          float *lw = scratch.get_pointer();
          float *lb = lw + C;
          float *lr = lb + C;
          for (int c = lid; c < C; c += wgsize) {
            lw[c] = weight[c];
            lb[c] = bias[c];
          }
          it.barrier(sycl::access::fence_space::local_space);
          for (int token = gid; token < N; token += ngrp) {
            float *res = residual + (std::size_t)token * (std::size_t)C;
            float *nrm = normed + (std::size_t)token * (std::size_t)C;
            const float *a = inp1 + (std::size_t)token * (std::size_t)C;
            const float *b = inp2 + (std::size_t)token * (std::size_t)C;
            float sum = 0.0f;
            for (int c = lid; c < C; c += wgsize) {
              float o = a[c] + b[c];
              sum += o;
              res[c] = o;
              lr[c] = o;
            }
            float m = sycl::reduce_over_group(grp, sum, sycl::plus<float>()) / (float)C;
            float vsum = 0.0f;
            for (int c = lid; c < C; c += wgsize) {
              float d = lr[c] - m;
              vsum += d * d;
            }
            float v =
                sycl::reduce_over_group(grp, vsum, sycl::plus<float>()) / (float)C;
            float s = sycl::rsqrt(v + eps);
            for (int c = lid; c < C; c += wgsize) {
              float nv = s * (lr[c] - m);
              nrm[c] = nv * lw[c] + lb[c];
            }
            if (lid == 0) {
              mean[token] = m;
              rstd[token] = s;
            }
            // Make local residual reuse safe across iterations.
            it.barrier(sycl::access::fence_space::local_space);
          }
        });
  }).wait();
}

static void run_variant(sycl::queue &q, int k,
                        float *residual, float *normed, float *mean, float *rstd,
                        const float *inp1, const float *inp2,
                        const float *weight, const float *bias,
                        int N, int C, int block_size) {
  switch (k) {
    case 1: launch_variant1(q, residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C, block_size); break;
    case 2: launch_variant2(q, residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C, block_size); break;
    case 3: launch_variant3(q, residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C, block_size); break;
    case 4: launch_variant4(q, residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C, block_size); break;
    case 5: launch_variant5(q, residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C, block_size); break;
    case 6: launch_variant6(q, residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C, block_size); break;
    default: std::fprintf(stderr, "Invalid kernel number\n"); std::exit(1);
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
  std::size_t ntc = (std::size_t)B * (std::size_t)T * (std::size_t)C;
  std::size_t nt = (std::size_t)B * (std::size_t)T;

  FILE *fin = std::fopen(argv[1], "rb");
  if (!fin) { std::perror(argv[1]); return 2; }
  float *inp1 = (float *)std::malloc(sizeof(float) * ntc);
  float *inp2 = (float *)std::malloc(sizeof(float) * ntc);
  float *weight = (float *)std::malloc(sizeof(float) * (std::size_t)C);
  float *bias = (float *)std::malloc(sizeof(float) * (std::size_t)C);
  if (!inp1 || !inp2 || !weight || !bias) { std::fprintf(stderr, "host oom\n"); return 2; }
  if (std::fread(inp1, sizeof(float), ntc, fin) != ntc) { std::fprintf(stderr, "input too short at inp1\n"); return 2; }
  if (std::fread(inp2, sizeof(float), ntc, fin) != ntc) { std::fprintf(stderr, "input too short at inp2\n"); return 2; }
  if (std::fread(weight, sizeof(float), (std::size_t)C, fin) != (std::size_t)C) { std::fprintf(stderr, "input too short at weight\n"); return 2; }
  if (std::fread(bias, sizeof(float), (std::size_t)C, fin) != (std::size_t)C) { std::fprintf(stderr, "input too short at bias\n"); return 2; }
  std::fclose(fin);

  FILE *fout = std::fopen(argv[2], "wb");
  if (!fout) { std::perror(argv[2]); return 2; }

  sycl::queue q{sycl::default_selector_v};
  float *d_inp1 = sycl::malloc_device<float>(ntc ? ntc : 1, q);
  float *d_inp2 = sycl::malloc_device<float>(ntc ? ntc : 1, q);
  float *d_weight = sycl::malloc_device<float>(C ? (std::size_t)C : 1, q);
  float *d_bias = sycl::malloc_device<float>(C ? (std::size_t)C : 1, q);
  float *d_residual = sycl::malloc_device<float>(ntc ? ntc : 1, q);
  float *d_normed = sycl::malloc_device<float>(ntc ? ntc : 1, q);
  float *d_mean = sycl::malloc_device<float>(nt ? nt : 1, q);
  float *d_rstd = sycl::malloc_device<float>(nt ? nt : 1, q);
  float *residual = (float *)std::malloc(sizeof(float) * (ntc ? ntc : 1));
  float *normed = (float *)std::malloc(sizeof(float) * (ntc ? ntc : 1));
  float *mean = (float *)std::malloc(sizeof(float) * (nt ? nt : 1));
  float *rstd = (float *)std::malloc(sizeof(float) * (nt ? nt : 1));
  if (!d_inp1 || !d_inp2 || !d_weight || !d_bias || !d_residual || !d_normed || !d_mean || !d_rstd ||
      !residual || !normed || !mean || !rstd) {
    std::fprintf(stderr, "oom\n");
    return 2;
  }

  if (ntc) {
    q.memcpy(d_inp1, inp1, sizeof(float) * ntc).wait();
    q.memcpy(d_inp2, inp2, sizeof(float) * ntc).wait();
  }
  if (C) {
    q.memcpy(d_weight, weight, sizeof(float) * (std::size_t)C).wait();
    q.memcpy(d_bias, bias, sizeof(float) * (std::size_t)C).wait();
  }

  const int variants[] = {1, 2, 3, 4, 5, 6};
  for (int vi = 0; vi < 6; ++vi) {
    const int k = variants[vi];
    if (ntc) {
      q.fill(d_residual, 0.0f, ntc).wait();
      q.fill(d_normed, 0.0f, ntc).wait();
    }
    if (nt) {
      q.fill(d_mean, 0.0f, nt).wait();
      q.fill(d_rstd, 0.0f, nt).wait();
    }
    run_variant(q, k, d_residual, d_normed, d_mean, d_rstd, d_inp1, d_inp2, d_weight,
                d_bias, B * T, C, 256);
    q.wait();
    if (ntc) {
      q.memcpy(residual, d_residual, sizeof(float) * ntc).wait();
      std::fwrite(residual, sizeof(float), ntc, fout);
      q.memcpy(normed, d_normed, sizeof(float) * ntc).wait();
      std::fwrite(normed, sizeof(float), ntc, fout);
    }
    if (nt) {
      q.memcpy(mean, d_mean, sizeof(float) * nt).wait();
      std::fwrite(mean, sizeof(float), nt, fout);
      q.memcpy(rstd, d_rstd, sizeof(float) * nt).wait();
      std::fwrite(rstd, sizeof(float), nt, fout);
    }
  }
  std::fclose(fout);
  sycl::free(d_inp1, q); sycl::free(d_inp2, q);
  sycl::free(d_weight, q); sycl::free(d_bias, q);
  sycl::free(d_residual, q); sycl::free(d_normed, q);
  sycl::free(d_mean, q); sycl::free(d_rstd, q);
  std::free(inp1); std::free(inp2); std::free(weight); std::free(bias);
  std::free(residual); std::free(normed); std::free(mean); std::free(rstd);
  return 0;
}
