#include <sycl/sycl.hpp>
#include <oneapi/math.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstdint>

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

class MatmulK1;
class BiasK2;
class BiasK3;
class MatmulK4;

// kernel 1: naive, one work-item per output element, 2-D launch over (BT, OC)
void matmul_forward1(sycl::queue &q, float *out, const float *inp,
                     const float *weight, const float *bias, int B, int T,
                     int C, int OC) {
  const int BT = B * T;
  constexpr int S = 16;
  const sycl::range<2> local(S, S);
  const sycl::range<2> global(static_cast<size_t>(ceil_div(BT, S) * S),
                              static_cast<size_t>(ceil_div(OC, S) * S));
  q.submit([&](sycl::handler &h) {
    h.parallel_for<MatmulK1>(sycl::nd_range<2>(global, local),
                             [=](sycl::nd_item<2> it) {
                               const int bt = static_cast<int>(
                                   it.get_group(0) * it.get_local_range(0) +
                                   it.get_local_id(0));
                               const int oc = static_cast<int>(
                                   it.get_group(1) * it.get_local_range(1) +
                                   it.get_local_id(1));
                               if (bt < BT && oc < OC) {
                                 float val =
                                     (bias != nullptr) ? bias[oc] : 0.0f;
                                 const float *wrow = weight + (size_t)oc * (size_t)C;
                                 const float *inp_bt = inp + (size_t)bt * (size_t)C;
                                 for (int i = 0; i < C; ++i) {
                                   val += inp_bt[i] * wrow[i];
                                 }
                                 out[(size_t)bt * (size_t)OC + (size_t)oc] = val;
                               }
                             });
  });
  q.wait();
}

// 1-D bias add used by variant 2
void add_bias_k2(sycl::queue &q, float *out, const float *bias, int B, int T,
                 int OC) {
  const size_t n = (size_t)B * (size_t)T * (size_t)OC;
  constexpr size_t WG = 256;
  const size_t groups = (n + WG - 1) / WG;
  const sycl::range<1> local(WG);
  const sycl::range<1> global(groups * WG);
  q.submit([&](sycl::handler &h) {
    h.parallel_for<BiasK2>(sycl::nd_range<1>(global, local),
                           [=](sycl::nd_item<1> it) {
                             const size_t idx =
                                 it.get_group(0) * it.get_local_range(0) +
                                 it.get_local_id(0);
                             const size_t stride =
                                 it.get_local_range(0) * it.get_group_range(0);
                             for (size_t i = idx; i < n; i += stride) {
                               out[i] += bias[i % (size_t)OC];
                             }
                           });
  });
  q.wait();
}

// 1-D bias add used by variant 3 (own kernel name)
void add_bias_k3(sycl::queue &q, float *out, const float *bias, int B, int T,
                 int OC) {
  const size_t n = (size_t)B * (size_t)T * (size_t)OC;
  constexpr size_t WG = 256;
  const size_t groups = (n + WG - 1) / WG;
  const sycl::range<1> local(WG);
  const sycl::range<1> global(groups * WG);
  q.submit([&](sycl::handler &h) {
    h.parallel_for<BiasK3>(sycl::nd_range<1>(global, local),
                           [=](sycl::nd_item<1> it) {
                             const size_t idx =
                                 it.get_group(0) * it.get_local_range(0) +
                                 it.get_local_id(0);
                             const size_t stride =
                                 it.get_local_range(0) * it.get_group_range(0);
                             for (size_t i = idx; i < n; i += stride) {
                               out[i] += bias[i % (size_t)OC];
                             }
                           });
  });
  q.wait();
}

// variant 2: oneMath GEMM + separate bias kernel
void matmul_forward2(sycl::queue &q, float *out, const float *inp,
                     const float *weight, const float *bias, int B, int T,
                     int C, int OC) {
  const std::int64_t m = OC;
  const std::int64_t n = (std::int64_t)B * (std::int64_t)T;
  const std::int64_t k = C;
  // Row-major out = inp @ weight^T seen column-major as out^T = weight @ inp^T:
  // A=weight (trans), B=inp (no-trans).
  oneapi::math::blas::column_major::gemm(
      q, oneapi::math::transpose::trans, oneapi::math::transpose::nontrans, m,
      n, k, 1.0f, weight, k, inp, k, 0.0f, out, m)
      .wait();
  if (bias != nullptr) {
    add_bias_k2(q, out, bias, B, T, OC);
  }
}

// variant 3: oneMath GEMM with fused-bias semantics (bias applied afterwards)
void matmul_forward3(sycl::queue &q, float *out, const float *inp,
                     const float *weight, const float *bias, int B, int T,
                     int C, int OC) {
  const std::int64_t m = OC;
  const std::int64_t n = (std::int64_t)B * (std::int64_t)T;
  const std::int64_t k = C;
  oneapi::math::blas::column_major::gemm(
      q, oneapi::math::transpose::trans, oneapi::math::transpose::nontrans, m,
      n, k, 1.0f, weight, k, inp, k, 0.0f, out, m)
      .wait();
  if (bias != nullptr) {
    add_bias_k3(q, out, bias, B, T, OC);
  }
}

// variant 4: handwritten tiled kernel, 16x16 work-groups, 128x128 tiles,
// each work-item handles 8x8 outputs.
void matmul_forward4(sycl::queue &q, float *out, const float *inp,
                     const float *weight, const float *bias, int B, int T,
                     int C, int OC) {
  const int BT = B * T;
  constexpr int S = 16;
  const int gx = ceil_div(BT, 128);
  const int gy = ceil_div(OC, 128);
  const sycl::range<2> local(S, S);
  const sycl::range<2> global((size_t)gx * (size_t)S,
                              (size_t)gy * (size_t)S);
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> lhs(sycl::range<1>(128 * 32), h);
    sycl::local_accessor<float, 1> rhs(sycl::range<1>(128 * 32), h);
    h.parallel_for<MatmulK4>(
        sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> it) {
          const int bx = static_cast<int>(it.get_group(0));
          const int by = static_cast<int>(it.get_group(1));
          const int tx = static_cast<int>(it.get_local_id(0));
          const int ty = static_cast<int>(it.get_local_id(1));
          const int oc = 8 * (by * S + ty);

          const float *inp_blk = inp + (size_t)128 * (size_t)bx * (size_t)C;
          const float *w_blk = weight + (size_t)128 * (size_t)by * (size_t)C;
          float *out_blk =
              out + (size_t)128 * (size_t)bx * (size_t)OC + (size_t)128 * (size_t)by;

          float vals[8][8] = {};
          if (bias != nullptr) {
            for (int i = 0; i < 8; ++i) {
              for (int j = 0; j < 8; ++j) {
                vals[i][j] = bias[oc + j];
              }
            }
          }

          const int si_start = 4 * (S * ty + tx);
          for (int so = 0; so < C; so += 32) {
            sycl::group_barrier(it.get_group(),
                                sycl::memory_scope::work_group);
            const int xmod8 = tx % 8;
            const int xby8 = tx / 8;
            const int xo = 4 * xmod8;
            for (int y = 2 * ty + xby8; y < 128; y += 32) {
              for (int v = 0; v < 4; ++v) {
                lhs[(size_t)y * 32 + (size_t)xo + (size_t)v] =
                    inp_blk[(size_t)y * (size_t)C + (size_t)so + (size_t)xo + (size_t)v];
                rhs[(size_t)y * 32 + (size_t)xo + (size_t)v] =
                    w_blk[(size_t)y * (size_t)C + (size_t)so + (size_t)xo + (size_t)v];
              }
            }
            sycl::group_barrier(it.get_group(),
                                sycl::memory_scope::work_group);

            for (int si = si_start; si < si_start + 32; si += 4) {
              const int sic = si % 32;
              float r[8][4];
              for (int u = 0; u < 8; ++u) {
                const size_t base = (size_t)(u + 8 * ty) * 32 + (size_t)sic;
                r[u][0] = rhs[base + 0];
                r[u][1] = rhs[base + 1];
                r[u][2] = rhs[base + 2];
                r[u][3] = rhs[base + 3];
              }
              for (int ii = 0; ii < 8; ++ii) {
                const size_t base =
                    (size_t)(ii + 8 * tx) * 32 + (size_t)sic;
                const float l0 = lhs[base + 0];
                const float l1 = lhs[base + 1];
                const float l2 = lhs[base + 2];
                const float l3 = lhs[base + 3];
                for (int ji = 0; ji < 8; ++ji) {
                  vals[ii][ji] += l0 * r[ji][0];
                  vals[ii][ji] += l1 * r[ji][1];
                  vals[ii][ji] += l2 * r[ji][2];
                  vals[ii][ji] += l3 * r[ji][3];
                }
              }
            }
          }

          for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 8; ++j) {
              out_blk[(size_t)(8 * tx + i) * (size_t)OC +
                      (size_t)(8 * ty + j)] = vals[i][j];
            }
          }
        });
  });
  q.wait();
}

void matmul_forward(sycl::queue &q, int kernel_num, float *out,
                    const float *inp, const float *weight, const float *bias,
                    int B, int T, int C, int OC) {
  switch (kernel_num) {
  case 1:
    matmul_forward1(q, out, inp, weight, bias, B, T, C, OC);
    break;
  case 2:
    matmul_forward2(q, out, inp, weight, bias, B, T, C, OC);
    break;
  case 3:
    matmul_forward3(q, out, inp, weight, bias, B, T, C, OC);
    break;
  case 4:
    matmul_forward4(q, out, inp, weight, bias, B, T, C, OC);
    break;
  default:
    std::fprintf(stderr, "Invalid kernel number\n");
    std::exit(1);
  }
}

int main(int argc, char **argv) {
  if (argc != 7) {
    std::fprintf(stderr, "usage: %s <input> <output> B T C OC\n", argv[0]);
    return 2;
  }
  const int B = std::atoi(argv[3]);
  const int T = std::atoi(argv[4]);
  const int C = std::atoi(argv[5]);
  const int OC = std::atoi(argv[6]);

  FILE *fin = std::fopen(argv[1], "rb");
  if (!fin) {
    std::perror(argv[1]);
    return 2;
  }
  float *inp = (float *)std::malloc(sizeof(float) * (size_t)(B * T * C));
  if (std::fread(inp, sizeof(float), (size_t)(B * T * C), fin) !=
      (size_t)(B * T * C)) {
    std::fprintf(stderr, "input too short at inp\n");
    return 2;
  }
  float *weight = (float *)std::malloc(sizeof(float) * (size_t)(OC * C));
  if (std::fread(weight, sizeof(float), (size_t)(OC * C), fin) !=
      (size_t)(OC * C)) {
    std::fprintf(stderr, "input too short at weight\n");
    return 2;
  }
  float *bias = (float *)std::malloc(sizeof(float) * (size_t)OC);
  if (std::fread(bias, sizeof(float), (size_t)OC, fin) != (size_t)OC) {
    std::fprintf(stderr, "input too short at bias\n");
    return 2;
  }
  std::fclose(fin);

  FILE *fout = std::fopen(argv[2], "wb");
  if (!fout) {
    std::perror(argv[2]);
    return 2;
  }

  sycl::queue q{sycl::default_selector_v,
                sycl::property::queue::in_order{}};

  float *d_inp = sycl::malloc_device<float>((size_t)B * (size_t)T * (size_t)C, q);
  q.memcpy(d_inp, inp, sizeof(float) * (size_t)B * (size_t)T * (size_t)C).wait();
  float *d_weight = sycl::malloc_device<float>((size_t)OC * (size_t)C, q);
  q.memcpy(d_weight, weight, sizeof(float) * (size_t)OC * (size_t)C).wait();
  float *d_bias = sycl::malloc_device<float>((size_t)OC, q);
  q.memcpy(d_bias, bias, sizeof(float) * (size_t)OC).wait();
  float *out = (float *)std::malloc(sizeof(float) * (size_t)B * (size_t)T * (size_t)OC);
  float *d_out =
      sycl::malloc_device<float>((size_t)B * (size_t)T * (size_t)OC, q);

  const int variants[] = {1, 2, 3, 4};
  for (int vi = 0; vi < 4; ++vi) {
    const int k = variants[vi];
    q.memset(d_out, 0, sizeof(float) * (size_t)B * (size_t)T * (size_t)OC).wait();
    matmul_forward(q, k, d_out, d_inp, d_weight, d_bias, B, T, C, OC);
    q.wait();
    q.memcpy(out, d_out, sizeof(float) * (size_t)B * (size_t)T * (size_t)OC).wait();
    std::fwrite(out, sizeof(float), (size_t)B * (size_t)T * (size_t)OC, fout);
  }
  std::fclose(fout);

  sycl::free(d_inp, q);
  sycl::free(d_weight, q);
  sycl::free(d_bias, q);
  sycl::free(d_out, q);
  std::free(inp);
  std::free(weight);
  std::free(bias);
  std::free(out);
  return 0;
}
