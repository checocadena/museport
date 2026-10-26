#include <sycl/sycl.hpp>
#include <oneapi/math.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstdint>

template <typename T>
static T ceil_div(T dividend, T divisor) {
  return (dividend + divisor - 1) / divisor;
}

// ---- kernel 1: naive, one work-item per output element (2-D launch) ----
class MatmulForwardK1;
static void matmul_forward1(sycl::queue &q, float *out, const float *inp,
                            const float *weight, const float *bias, int B,
                            int T, int C, int OC) {
  const int BT = B * T;
  sycl::range<2> local(16, 16);
  sycl::range<2> groups(ceil_div(BT, 16), ceil_div(OC, 16));
  sycl::range<2> global(groups[0] * local[0], groups[1] * local[1]);
  q.parallel_for<MatmulForwardK1>(
      sycl::nd_range<2>(global, local),
      [=](sycl::nd_item<2> it) {
        const int bt = static_cast<int>(it.get_global_id(0));
        const int oc = static_cast<int>(it.get_global_id(1));
        if (bt < BT && oc < OC) {
          float val = bias ? bias[oc] : 0.0f;
          const float *wrow = weight + static_cast<std::size_t>(oc) * C;
          const float *inp_bt = inp + static_cast<std::size_t>(bt) * C;
          for (int i = 0; i < C; ++i) val += inp_bt[i] * wrow[i];
          out[static_cast<std::size_t>(bt) * OC + oc] = val;
        }
      }).wait();
}

// ---- bias helper kernels (1-D launches, distinct names per variant) ----
class AddBiasK2;
static void add_bias_k2(sycl::queue &q, float *out, const float *bias,
                        std::size_t n, int OC) {
  constexpr std::size_t local = 256;
  const std::size_t global = ceil_div(n, local) * local;
  q.parallel_for<AddBiasK2>(
      sycl::nd_range<1>(global, local), [=](sycl::nd_item<1> it) {
        const std::size_t idx = it.get_global_id(0);
        if (idx < n) out[idx] += bias[idx % static_cast<std::size_t>(OC)];
      }).wait();
}

class AddBiasK3;
static void add_bias_k3(sycl::queue &q, float *out, const float *bias,
                        std::size_t n, int OC) {
  constexpr std::size_t local = 256;
  const std::size_t global = ceil_div(n, local) * local;
  q.parallel_for<AddBiasK3>(
      sycl::nd_range<1>(global, local), [=](sycl::nd_item<1> it) {
        const std::size_t idx = it.get_global_id(0);
        if (idx < n) out[idx] += bias[idx % static_cast<std::size_t>(OC)];
      }).wait();
}

// ---- kernel 2: oneMath GEMM + separate bias kernel ----
static void matmul_forward2(sycl::queue &q, float *out, const float *inp,
                            const float *weight, const float *bias, int B,
                            int T, int C, int OC) {
  const std::int64_t m = OC;
  const std::int64_t n = static_cast<std::int64_t>(B) * T;
  const std::int64_t k = C;
  oneapi::math::blas::column_major::gemm(
      q, oneapi::math::transpose::trans, oneapi::math::transpose::nontrans, m,
      n, k, 1.0f, weight, C, inp, C, 0.0f, out, OC).wait();
  if (bias) {
    const std::size_t total = static_cast<std::size_t>(B) * T * OC;
    add_bias_k2(q, out, bias, total, OC);
  }
}

// ---- kernel 3: oneMath GEMM with fused-bias semantics (bias epilogue) ----
static void matmul_forward3(sycl::queue &q, float *out, const float *inp,
                            const float *weight, const float *bias, int B,
                            int T, int C, int OC) {
  const std::int64_t m = OC;
  const std::int64_t n = static_cast<std::int64_t>(B) * T;
  const std::int64_t k = C;
  oneapi::math::blas::column_major::gemm(
      q, oneapi::math::transpose::trans, oneapi::math::transpose::nontrans, m,
      n, k, 1.0f, weight, C, inp, C, 0.0f, out, OC).wait();
  if (bias) {
    const std::size_t total = static_cast<std::size_t>(B) * T * OC;
    add_bias_k3(q, out, bias, total, OC);
  }
}

// ---- kernel 4: tiled 128x128 handwritten kernel (2-D launch) ----
class MatmulForwardK4;
static void matmul_forward4(sycl::queue &q, float *out, const float *inp,
                            const float *weight, const float *bias, int B,
                            int T, int C, int OC) {
  const int BT = B * T;
  constexpr int tile = 128;
  constexpr int local_dim = 16;
  const int nbx = ceil_div(BT, tile);
  const int nby = ceil_div(OC, tile);
  sycl::range<2> local(local_dim, local_dim);
  sycl::range<2> global(static_cast<std::size_t>(nbx) * local_dim,
                        static_cast<std::size_t>(nby) * local_dim);
  constexpr std::size_t scratch = 128 * 32;
  q.submit([&](sycl::handler &h) {
     sycl::local_accessor<float, 1> lhs(sycl::range<1>(scratch), h);
     sycl::local_accessor<float, 1> rhs(sycl::range<1>(scratch), h);
     h.parallel_for<MatmulForwardK4>(
         sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> it) {
           const int bx = static_cast<int>(it.get_group(0));
           const int by = static_cast<int>(it.get_group(1));
           const int tx = static_cast<int>(it.get_local_id(0));
           const int ty = static_cast<int>(it.get_local_id(1));
           const int oc_base = 8 * (by * local_dim + ty);
           const float *inp_blk =
               inp + static_cast<std::size_t>(128 * bx) * C;
           const float *w_blk =
               weight + static_cast<std::size_t>(128 * by) * C;
           float *out_blk = out +
                            static_cast<std::size_t>(128 * bx) * OC +
                            static_cast<std::size_t>(128 * by);
           float vals[8][8] = {};
           if (bias) {
             for (int i = 0; i < 8; ++i)
               for (int j = 0; j < 8; ++j)
                 vals[i][j] = bias[oc_base + j];
           }
           const int si_start = 4 * (16 * ty + tx);
           for (int so = 0; so < C; so += 32) {
             sycl::group_barrier(it.get_group());
             const int xmod8 = tx % 8;
             const int xby8 = tx / 8;
             const int xo = 4 * xmod8;
             for (int y = 2 * ty + xby8; y < 128; y += 32) {
               const float *src_l =
                   inp_blk + static_cast<std::size_t>(y) * C + so + xo;
               std::size_t dst_base =
                   static_cast<std::size_t>(y) * 32 + xo;
               lhs[dst_base + 0] = src_l[0];
               lhs[dst_base + 1] = src_l[1];
               lhs[dst_base + 2] = src_l[2];
               lhs[dst_base + 3] = src_l[3];
               const float *src_r =
                   w_blk + static_cast<std::size_t>(y) * C + so + xo;
               rhs[dst_base + 0] = src_r[0];
               rhs[dst_base + 1] = src_r[1];
               rhs[dst_base + 2] = src_r[2];
               rhs[dst_base + 3] = src_r[3];
             }
             sycl::group_barrier(it.get_group());
             for (int si = si_start; si < si_start + 32; si += 4) {
               const int col = si % 32;
               float rhs_buf[8][4];
               for (int u = 0; u < 8; ++u) {
                 std::size_t p =
                     static_cast<std::size_t>(u + 8 * ty) * 32 + col;
                 rhs_buf[u][0] = rhs[p + 0];
                 rhs_buf[u][1] = rhs[p + 1];
                 rhs_buf[u][2] = rhs[p + 2];
                 rhs_buf[u][3] = rhs[p + 3];
               }
               for (int ii = 0; ii < 8; ++ii) {
                 std::size_t p =
                     static_cast<std::size_t>(ii + 8 * tx) * 32 + col;
                 const float l0 = lhs[p + 0];
                 const float l1 = lhs[p + 1];
                 const float l2 = lhs[p + 2];
                 const float l3 = lhs[p + 3];
                 for (int ji = 0; ji < 8; ++ji) {
                   vals[ii][ji] += l0 * rhs_buf[ji][0] +
                                   l1 * rhs_buf[ji][1] +
                                   l2 * rhs_buf[ji][2] + l3 * rhs_buf[ji][3];
                 }
               }
             }
           }
           for (int i = 0; i < 8; ++i) {
             float *row =
                 out_blk + static_cast<std::size_t>(8 * tx + i) * OC + 8 * ty;
             for (int j = 0; j < 8; ++j) row[j] = vals[i][j];
           }
         });
   }).wait();
}

static void matmul_forward(sycl::queue &q, int kernel_num, float *out,
                           const float *inp, const float *weight,
                           const float *bias, int B, int T, int C, int OC) {
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
  float *inp = static_cast<float *>(
      std::malloc(sizeof(float) * static_cast<std::size_t>(B) * T * C));
  if (std::fread(inp, sizeof(float),
                 static_cast<std::size_t>(B) * T * C, fin) !=
      static_cast<std::size_t>(B) * T * C) {
    std::fprintf(stderr, "input too short at inp\n");
    return 2;
  }
  float *weight = static_cast<float *>(
      std::malloc(sizeof(float) * static_cast<std::size_t>(OC) * C));
  if (std::fread(weight, sizeof(float),
                 static_cast<std::size_t>(OC) * C, fin) !=
      static_cast<std::size_t>(OC) * C) {
    std::fprintf(stderr, "input too short at weight\n");
    return 2;
  }
  float *bias = static_cast<float *>(
      std::malloc(sizeof(float) * static_cast<std::size_t>(OC)));
  if (std::fread(bias, sizeof(float), static_cast<std::size_t>(OC), fin) !=
      static_cast<std::size_t>(OC)) {
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
  float *d_inp = sycl::malloc_device<float>(
      static_cast<std::size_t>(B) * T * C, q);
  float *d_weight =
      sycl::malloc_device<float>(static_cast<std::size_t>(OC) * C, q);
  float *d_bias = sycl::malloc_device<float>(static_cast<std::size_t>(OC), q);
  float *d_out = sycl::malloc_device<float>(
      static_cast<std::size_t>(B) * T * OC, q);
  float *out = static_cast<float *>(
      std::malloc(sizeof(float) * static_cast<std::size_t>(B) * T * OC));

  q.memcpy(d_inp, inp, sizeof(float) * static_cast<std::size_t>(B) * T * C)
      .wait();
  q.memcpy(d_weight, weight,
           sizeof(float) * static_cast<std::size_t>(OC) * C).wait();
  q.memcpy(d_bias, bias, sizeof(float) * static_cast<std::size_t>(OC)).wait();

  const int variants[] = {1, 2, 3, 4};
  for (int vi = 0; vi < 4; ++vi) {
    const int k = variants[vi];
    q.memset(d_out, 0, sizeof(float) * static_cast<std::size_t>(B) * T * OC)
        .wait();
    matmul_forward(q, k, d_out, d_inp, d_weight, d_bias, B, T, C, OC);
    q.wait();
    q.memcpy(out, d_out,
             sizeof(float) * static_cast<std::size_t>(B) * T * OC).wait();
    std::fwrite(out, sizeof(float),
                static_cast<std::size_t>(B) * T * OC, fout);
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
