// Idiomatic SYCL 2020 port of fused_residual_forward (fp32).
// Six distinct device kernels, one per variant, 1-D launches, USM.
#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

using floatX = float;

template <typename T>
static T ceil_div(T a, T b) { return (a + b - 1) / b; }

using Vec4 = sycl::vec<float, 4>;

static inline Vec4 load_vec4(const float *p) {
  return *reinterpret_cast<const Vec4 *>(p);
}
static inline void store_vec4(float *p, Vec4 v) {
  *reinterpret_cast<Vec4 *>(p) = v;
}

struct KernelResid1A {};
struct KernelNorm1B {};
struct KernelFused2 {};
struct KernelFused3 {};
struct KernelFused4 {};
struct KernelFused5 {};
struct KernelFused6 {};

static constexpr float kEps = 1e-5f;
static constexpr int kLocal = 256;
static constexpr int kVec = 4;

void launch_variant1(sycl::queue &q, float *residual, float *normed, float *mean,
                     float *rstd, const float *inp1, const float *inp2,
                     const float *weight, const float *bias, int N, int C) {
  int nc = N * C;
  std::size_t g1 = static_cast<std::size_t>(ceil_div(nc, kLocal)) * kLocal;
  q.submit([&](sycl::handler &h) {
    h.parallel_for<KernelResid1A>(
        sycl::nd_range<1>(sycl::range<1>(g1), sycl::range<1>(kLocal)),
        [=](sycl::nd_item<1> it) {
          std::size_t idx = it.get_global_id(0);
          if (idx < static_cast<std::size_t>(nc))
            residual[idx] = inp1[idx] + inp2[idx];
        });
  });
  std::size_t g2 = static_cast<std::size_t>(ceil_div(N, kLocal)) * kLocal;
  q.submit([&](sycl::handler &h) {
    h.parallel_for<KernelNorm1B>(
        sycl::nd_range<1>(sycl::range<1>(g2), sycl::range<1>(kLocal)),
        [=](sycl::nd_item<1> it) {
          int idx = static_cast<int>(it.get_global_id(0));
          if (idx >= N)
            return;
          const float *x = residual + static_cast<std::size_t>(idx) * C;
          float m = 0.0f;
          for (int i = 0; i < C; ++i)
            m += x[i];
          m /= static_cast<float>(C);
          float v = 0.0f;
          for (int i = 0; i < C; ++i) {
            float d = x[i] - m;
            v += d * d;
          }
          v /= static_cast<float>(C);
          float s = 1.0f / sycl::sqrt(v + kEps);
          float *o = normed + static_cast<std::size_t>(idx) * C;
          for (int i = 0; i < C; ++i) {
            float n = s * (x[i] - m);
            o[i] = n * weight[i] + bias[i];
          }
          mean[idx] = m;
          rstd[idx] = s;
        });
  });
  q.wait();
}

void launch_variant2(sycl::queue &q, float *residual, float *normed, float *mean,
                     float *rstd, const float *inp1, const float *inp2,
                     const float *weight, const float *bias, int N, int C) {
  std::size_t g = static_cast<std::size_t>(ceil_div(N, kLocal)) * kLocal;
  q.submit([&](sycl::handler &h) {
    h.parallel_for<KernelFused2>(
        sycl::nd_range<1>(sycl::range<1>(g), sycl::range<1>(kLocal)),
        [=](sycl::nd_item<1> it) {
          int idx = static_cast<int>(it.get_global_id(0));
          if (idx >= N)
            return;
          std::size_t base = static_cast<std::size_t>(idx) * C;
          float *res = residual + base;
          float *out = normed + base;
          const float *a = inp1 + base;
          const float *b = inp2 + base;
          float m = 0.0f;
          for (int c = 0; c < C; ++c) {
            float v = a[c] + b[c];
            m += v;
            res[c] = v;
          }
          m /= static_cast<float>(C);
          float var = 0.0f;
          for (int c = 0; c < C; ++c) {
            float d = res[c] - m;
            var += d * d;
          }
          var /= static_cast<float>(C);
          float s = 1.0f / sycl::sqrt(var + kEps);
          for (int c = 0; c < C; ++c) {
            float n = s * (res[c] - m);
            out[c] = n * weight[c] + bias[c];
          }
          mean[idx] = m;
          rstd[idx] = s;
        });
  });
  q.wait();
}

void launch_variant3(sycl::queue &q, float *residual, float *normed, float *mean,
                     float *rstd, const float *inp1, const float *inp2,
                     const float *weight, const float *bias, int N, int C) {
  std::size_t global = static_cast<std::size_t>(N) * kLocal;
  q.submit([&](sycl::handler &h) {
    h.parallel_for<KernelFused3>(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(kLocal)),
        [=](sycl::nd_item<1> it) {
          int token = static_cast<int>(it.get_group(0));
          auto grp = it.get_group();
          int lid = static_cast<int>(it.get_local_id(0));
          int lsize = static_cast<int>(it.get_local_range(0));
          std::size_t base = static_cast<std::size_t>(token) * C;
          float *res = residual + base;
          float *out = normed + base;
          const float *a = inp1 + base;
          const float *b = inp2 + base;
          float partial = 0.0f;
          for (int c = lid; c < C; c += lsize) {
            float v = a[c] + b[c];
            res[c] = v;
            partial += v;
          }
          float sum = sycl::reduce_over_group(grp, partial, sycl::plus<float>());
          float m = sum / static_cast<float>(C);
          sycl::group_barrier(grp, sycl::memory_scope::device);
          float pvar = 0.0f;
          for (int c = lid; c < C; c += lsize) {
            float d = res[c] - m;
            pvar += d * d;
          }
          float vsum = sycl::reduce_over_group(grp, pvar, sycl::plus<float>());
          float var = vsum / static_cast<float>(C);
          float s = 1.0f / sycl::sqrt(var + kEps);
          for (int c = lid; c < C; c += lsize) {
            float n = s * (res[c] - m);
            out[c] = n * weight[c] + bias[c];
          }
          if (lid == 0) {
            mean[token] = m;
            rstd[token] = s;
          }
        });
  });
  q.wait();
}

void launch_variant4(sycl::queue &q, float *residual, float *normed, float *mean,
                     float *rstd, const float *inp1, const float *inp2,
                     const float *weight, const float *bias, int N, int C) {
  std::size_t global = static_cast<std::size_t>(N) * kLocal;
  q.submit([&](sycl::handler &h) {
    h.parallel_for<KernelFused4>(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(kLocal)),
        [=](sycl::nd_item<1> it) {
          int token = static_cast<int>(it.get_group(0));
          auto grp = it.get_group();
          int lid = static_cast<int>(it.get_local_id(0));
          int lsize = static_cast<int>(it.get_local_range(0));
          std::size_t base = static_cast<std::size_t>(token) * C;
          float *res = residual + base;
          float *out = normed + base;
          const float *a = inp1 + base;
          const float *b = inp2 + base;
          int vecC = C / kVec;
          float psum = 0.0f;
          float psq = 0.0f;
          for (int v = lid; v < vecC; v += lsize) {
            Vec4 in1 = load_vec4(a + v * kVec);
            Vec4 in2 = load_vec4(b + v * kVec);
            Vec4 o = in1 + in2;
            psum += o.x() + o.y() + o.z() + o.w();
            psq += o.x() * o.x() + o.y() * o.y() + o.z() * o.z() + o.w() * o.w();
            store_vec4(res + v * kVec, o);
          }
          for (int c = vecC * kVec + lid; c < C; c += lsize) {
            float v = a[c] + b[c];
            psum += v;
            psq += v * v;
            res[c] = v;
          }
          float sum = sycl::reduce_over_group(grp, psum, sycl::plus<float>());
          float sumsq = sycl::reduce_over_group(grp, psq, sycl::plus<float>());
          float m = sum / static_cast<float>(C);
          float var = sumsq / static_cast<float>(C) - m * m;
          float s = sycl::rsqrt(var + kEps);
          sycl::group_barrier(grp, sycl::memory_scope::device);
          for (int v = vecC - 1 - lid; v >= 0; v -= lsize) {
            Vec4 rv = load_vec4(res + v * kVec);
            Vec4 w = load_vec4(weight + v * kVec);
            Vec4 bi = load_vec4(bias + v * kVec);
            float nx = s * (rv.x() - m);
            float ny = s * (rv.y() - m);
            float nz = s * (rv.z() - m);
            float nw = s * (rv.w() - m);
            Vec4 o(nx * w.x() + bi.x(), ny * w.y() + bi.y(), nz * w.z() + bi.z(),
                   nw * w.w() + bi.w());
            store_vec4(out + v * kVec, o);
          }
          for (int c = C - 1 - lid; c >= vecC * kVec; c -= lsize) {
            float n = s * (res[c] - m);
            out[c] = n * weight[c] + bias[c];
          }
          if (lid == 0) {
            mean[token] = m;
            rstd[token] = s;
          }
        });
  });
  q.wait();
}

void launch_variant5(sycl::queue &q, float *residual, float *normed, float *mean,
                     float *rstd, const float *inp1, const float *inp2,
                     const float *weight, const float *bias, int N, int C) {
  std::size_t global = static_cast<std::size_t>(N) * kLocal;
  std::size_t scratch = static_cast<std::size_t>(3) * static_cast<std::size_t>(C);
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> loc(sycl::range<1>(scratch), h);
    h.parallel_for<KernelFused5>(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(kLocal)),
        [=](sycl::nd_item<1> it) {
          auto grp = it.get_group();
          int token = static_cast<int>(grp.get_group_id(0));
          int lid = static_cast<int>(it.get_local_id(0));
          int lsize = static_cast<int>(it.get_local_range(0));
          sycl::multi_ptr<float, sycl::access::address_space::local_space,
                           sycl::access::decorated::no>
              mp = loc.get_multi_ptr<sycl::access::decorated::no>();
          float *sW = mp.get();
          float *sB = mp.get() + C;
          float *sR = mp.get() + 2 * C;
          for (int i = lid; i < C; i += lsize) {
            sW[i] = weight[i];
            sB[i] = bias[i];
          }
          sycl::group_barrier(grp);
          std::size_t base = static_cast<std::size_t>(token) * C;
          float *res = residual + base;
          float *out = normed + base;
          const float *a = inp1 + base;
          const float *b = inp2 + base;
          int vecC = C / kVec;
          float psum = 0.0f;
          for (int v = lid; v < vecC; v += lsize) {
            Vec4 in1 = load_vec4(a + v * kVec);
            Vec4 in2 = load_vec4(b + v * kVec);
            Vec4 o = in1 + in2;
            psum += o.x() + o.y() + o.z() + o.w();
            store_vec4(res + v * kVec, o);
            store_vec4(sR + v * kVec, o);
          }
          for (int c = vecC * kVec + lid; c < C; c += lsize) {
            float v = a[c] + b[c];
            psum += v;
            res[c] = v;
            sR[c] = v;
          }
          float sum = sycl::reduce_over_group(grp, psum, sycl::plus<float>());
          float m = sum / static_cast<float>(C);
          sycl::group_barrier(grp);
          float pvar = 0.0f;
          for (int v = lid; v < vecC; v += lsize) {
            Vec4 rv = load_vec4(sR + v * kVec);
            float dx = rv.x() - m, dy = rv.y() - m, dz = rv.z() - m,
                  dw = rv.w() - m;
            pvar += dx * dx + dy * dy + dz * dz + dw * dw;
          }
          for (int c = vecC * kVec + lid; c < C; c += lsize) {
            float d = sR[c] - m;
            pvar += d * d;
          }
          float vsum = sycl::reduce_over_group(grp, pvar, sycl::plus<float>());
          float var = vsum / static_cast<float>(C);
          float s = sycl::rsqrt(var + kEps);
          for (int v = lid; v < vecC; v += lsize) {
            Vec4 rv = load_vec4(sR + v * kVec);
            Vec4 w = load_vec4(sW + v * kVec);
            Vec4 bi = load_vec4(sB + v * kVec);
            Vec4 o(s * (rv.x() - m) * w.x() + bi.x(),
                   s * (rv.y() - m) * w.y() + bi.y(),
                   s * (rv.z() - m) * w.z() + bi.z(),
                   s * (rv.w() - m) * w.w() + bi.w());
            store_vec4(out + v * kVec, o);
          }
          for (int c = vecC * kVec + lid; c < C; c += lsize) {
            float n = s * (sR[c] - m);
            out[c] = n * sW[c] + sB[c];
          }
          if (lid == 0) {
            mean[token] = m;
            rstd[token] = s;
          }
        });
  });
  q.wait();
}

void launch_variant6(sycl::queue &q, float *residual, float *normed, float *mean,
                     float *rstd, const float *inp1, const float *inp2,
                     const float *weight, const float *bias, int N, int C) {
  int num_groups = 8;
  if (num_groups > N)
    num_groups = N;
  if (num_groups < 1)
    num_groups = 1;
  std::size_t global = static_cast<std::size_t>(num_groups) * kLocal;
  std::size_t scratch = static_cast<std::size_t>(3) * static_cast<std::size_t>(C);
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> loc(sycl::range<1>(scratch), h);
    h.parallel_for<KernelFused6>(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(kLocal)),
        [=](sycl::nd_item<1> it) {
          auto grp = it.get_group();
          int gid = static_cast<int>(grp.get_group_id(0));
          int ngrp = static_cast<int>(grp.get_group_range(0));
          int lid = static_cast<int>(it.get_local_id(0));
          int lsize = static_cast<int>(it.get_local_range(0));
          sycl::multi_ptr<float, sycl::access::address_space::local_space,
                           sycl::access::decorated::no>
              mp = loc.get_multi_ptr<sycl::access::decorated::no>();
          float *sW = mp.get();
          float *sB = mp.get() + C;
          float *sR = mp.get() + 2 * C;
          for (int i = lid; i < C; i += lsize) {
            sW[i] = weight[i];
            sB[i] = bias[i];
          }
          sycl::group_barrier(grp);
          int vecC = C / kVec;
          for (int token = gid; token < N; token += ngrp) {
            std::size_t base = static_cast<std::size_t>(token) * C;
            float *res = residual + base;
            float *out = normed + base;
            const float *a = inp1 + base;
            const float *b = inp2 + base;
            float psum = 0.0f;
            for (int v = lid; v < vecC; v += lsize) {
              Vec4 in1 = load_vec4(a + v * kVec);
              Vec4 in2 = load_vec4(b + v * kVec);
              Vec4 o = in1 + in2;
              psum += o.x() + o.y() + o.z() + o.w();
              store_vec4(res + v * kVec, o);
              store_vec4(sR + v * kVec, o);
            }
            for (int c = vecC * kVec + lid; c < C; c += lsize) {
              float v = a[c] + b[c];
              psum += v;
              res[c] = v;
              sR[c] = v;
            }
            float sum = sycl::reduce_over_group(grp, psum, sycl::plus<float>());
            float m = sum / static_cast<float>(C);
            sycl::group_barrier(grp);
            float pvar = 0.0f;
            for (int v = lid; v < vecC; v += lsize) {
              Vec4 rv = load_vec4(sR + v * kVec);
              float dx = rv.x() - m, dy = rv.y() - m, dz = rv.z() - m,
                    dw = rv.w() - m;
              pvar += dx * dx + dy * dy + dz * dz + dw * dw;
            }
            for (int c = vecC * kVec + lid; c < C; c += lsize) {
              float d = sR[c] - m;
              pvar += d * d;
            }
            float vsum =
                sycl::reduce_over_group(grp, pvar, sycl::plus<float>());
            float var = vsum / static_cast<float>(C);
            float s = sycl::rsqrt(var + kEps);
            for (int v = lid; v < vecC; v += lsize) {
              Vec4 rv = load_vec4(sR + v * kVec);
              Vec4 w = load_vec4(sW + v * kVec);
              Vec4 bi = load_vec4(sB + v * kVec);
              Vec4 o(s * (rv.x() - m) * w.x() + bi.x(),
                     s * (rv.y() - m) * w.y() + bi.y(),
                     s * (rv.z() - m) * w.z() + bi.z(),
                     s * (rv.w() - m) * w.w() + bi.w());
              store_vec4(out + v * kVec, o);
            }
            for (int c = vecC * kVec + lid; c < C; c += lsize) {
              float n = s * (sR[c] - m);
              out[c] = n * sW[c] + sB[c];
            }
            if (lid == 0) {
              mean[token] = m;
              rstd[token] = s;
            }
            sycl::group_barrier(grp, sycl::memory_scope::device);
          }
        });
  });
  q.wait();
}

int main(int argc, char **argv) {
  if (argc != 6) {
    std::fprintf(stderr, "usage: %s <input> <output> B T C\n", argv[0]);
    return 2;
  }
  const int B = std::atoi(argv[3]);
  const int T = std::atoi(argv[4]);
  const int C = std::atoi(argv[5]);
  std::size_t nTok = static_cast<std::size_t>(B) * static_cast<std::size_t>(T);
  std::size_t nEl =
      static_cast<std::size_t>(B) * static_cast<std::size_t>(T) * static_cast<std::size_t>(C);
  int N = B * T;

  FILE *fin = std::fopen(argv[1], "rb");
  if (!fin) {
    std::perror(argv[1]);
    return 2;
  }
  float *inp1 = (float *)std::malloc(sizeof(float) * nEl);
  float *inp2 = (float *)std::malloc(sizeof(float) * nEl);
  float *weight = (float *)std::malloc(sizeof(float) * (std::size_t)C);
  float *bias = (float *)std::malloc(sizeof(float) * (std::size_t)C);
  if (std::fread(inp1, sizeof(float), nEl, fin) != nEl) {
    std::fprintf(stderr, "input too short at inp1\n");
    return 2;
  }
  if (std::fread(inp2, sizeof(float), nEl, fin) != nEl) {
    std::fprintf(stderr, "input too short at inp2\n");
    return 2;
  }
  if (std::fread(weight, sizeof(float), (std::size_t)C, fin) != (std::size_t)C) {
    std::fprintf(stderr, "input too short at weight\n");
    return 2;
  }
  if (std::fread(bias, sizeof(float), (std::size_t)C, fin) != (std::size_t)C) {
    std::fprintf(stderr, "input too short at bias\n");
    return 2;
  }
  std::fclose(fin);

  sycl::queue q{sycl::default_selector_v, {sycl::property::queue::in_order{}}};

  float *d_inp1 = sycl::malloc_device<float>(nEl, q);
  float *d_inp2 = sycl::malloc_device<float>(nEl, q);
  float *d_weight = sycl::malloc_device<float>((std::size_t)C, q);
  float *d_bias = sycl::malloc_device<float>((std::size_t)C, q);
  float *d_res = sycl::malloc_device<float>(nEl, q);
  float *d_nrm = sycl::malloc_device<float>(nEl, q);
  float *d_mean = sycl::malloc_device<float>(nTok, q);
  float *d_rstd = sycl::malloc_device<float>(nTok, q);
  if (!d_inp1 || !d_inp2 || !d_weight || !d_bias || !d_res || !d_nrm || !d_mean ||
      !d_rstd) {
    std::fprintf(stderr, "device alloc failed\n");
    return 1;
  }
  q.memcpy(d_inp1, inp1, sizeof(float) * nEl).wait();
  q.memcpy(d_inp2, inp2, sizeof(float) * nEl).wait();
  q.memcpy(d_weight, weight, sizeof(float) * (std::size_t)C).wait();
  q.memcpy(d_bias, bias, sizeof(float) * (std::size_t)C).wait();

  float *residual = (float *)std::malloc(sizeof(float) * nEl);
  float *normed = (float *)std::malloc(sizeof(float) * nEl);
  float *mean = (float *)std::malloc(sizeof(float) * nTok);
  float *rstd = (float *)std::malloc(sizeof(float) * nTok);

  FILE *fout = std::fopen(argv[2], "wb");
  if (!fout) {
    std::perror(argv[2]);
    return 2;
  }
  const int variants[6] = {1, 2, 3, 4, 5, 6};
  for (int vi = 0; vi < 6; ++vi) {
    int k = variants[vi];
    q.fill(d_res, 0.0f, nEl).wait();
    q.fill(d_nrm, 0.0f, nEl).wait();
    q.fill(d_mean, 0.0f, nTok).wait();
    q.fill(d_rstd, 0.0f, nTok).wait();
    switch (k) {
    case 1:
      launch_variant1(q, d_res, d_nrm, d_mean, d_rstd, d_inp1, d_inp2, d_weight,
                      d_bias, N, C);
      break;
    case 2:
      launch_variant2(q, d_res, d_nrm, d_mean, d_rstd, d_inp1, d_inp2, d_weight,
                      d_bias, N, C);
      break;
    case 3:
      launch_variant3(q, d_res, d_nrm, d_mean, d_rstd, d_inp1, d_inp2, d_weight,
                      d_bias, N, C);
      break;
    case 4:
      launch_variant4(q, d_res, d_nrm, d_mean, d_rstd, d_inp1, d_inp2, d_weight,
                      d_bias, N, C);
      break;
    case 5:
      launch_variant5(q, d_res, d_nrm, d_mean, d_rstd, d_inp1, d_inp2, d_weight,
                      d_bias, N, C);
      break;
    case 6:
      launch_variant6(q, d_res, d_nrm, d_mean, d_rstd, d_inp1, d_inp2, d_weight,
                      d_bias, N, C);
      break;
    default:
      std::printf("Invalid kernel number\n");
      return 1;
    }
    q.memcpy(residual, d_res, sizeof(float) * nEl).wait();
    std::fwrite(residual, sizeof(float), nEl, fout);
    q.memcpy(normed, d_nrm, sizeof(float) * nEl).wait();
    std::fwrite(normed, sizeof(float), nEl, fout);
    q.memcpy(mean, d_mean, sizeof(float) * nTok).wait();
    std::fwrite(mean, sizeof(float), nTok, fout);
    q.memcpy(rstd, d_rstd, sizeof(float) * nTok).wait();
    std::fwrite(rstd, sizeof(float), nTok, fout);
  }
  std::fclose(fout);
  sycl::free(d_inp1, q);
  sycl::free(d_inp2, q);
  sycl::free(d_weight, q);
  sycl::free(d_bias, q);
  sycl::free(d_res, q);
  sycl::free(d_nrm, q);
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
