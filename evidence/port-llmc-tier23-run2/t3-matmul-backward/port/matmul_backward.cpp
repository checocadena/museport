#include <sycl/sycl.hpp>
#include <oneapi/math.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstdint>

// ---------------------------------------------------------------------------
// Device kernels (one SYCL kernel per variant helper)

// Naive bias reduction: one work-item per output channel.
// Kept as its own SYCL kernel (mirrors matmul_backward_bias_kernel_naive).
class BiasNaiveKernel;

static void launch_bias_naive(sycl::queue &q, float *dbias, const float *dout,
                              int B, int T, int OC) {
  const int N = B * T;
  q.submit([&](sycl::handler &h) {
     h.parallel_for<BiasNaiveKernel>(sycl::range<1>(static_cast<size_t>(OC)),
       [=](sycl::item<1> it) {
         const int o = static_cast<int>(it.get_id(0));
         double sum = 0.0;
         for (int i = 0; i < N; ++i) {
           sum += dout[static_cast<size_t>(i) * static_cast<size_t>(OC) +
                       static_cast<size_t>(o)];
         }
         dbias[o] = static_cast<float>(sum);
       });
   }).wait();
}

// Faster bias reduction: one work-group per output channel, coarsened over
// B*T with a work-group-wide reduction (mirrors
// matmul_backward_bias_kernel_faster, idiomatic SYCL 2020 style).
class BiasFasterKernel;

static void launch_bias_faster(sycl::queue &q, float *dbias, const float *dout,
                               int B, int T, int OC) {
  constexpr int WG = 256;
  const int N = B * T;
  q.submit([&](sycl::handler &h) {
     h.parallel_for<BiasFasterKernel>(
       sycl::nd_range<1>(static_cast<size_t>(OC) * WG, WG),
       [=](sycl::nd_item<1> it) {
         const int o = static_cast<int>(it.get_group(0));
         const auto grp = it.get_group();
         const int tid = static_cast<int>(it.get_local_id(0));
         const int n = static_cast<int>(it.get_local_range(0));
         const float *col = dout + o;
         double sum = 0.0;
         for (int i = tid; i < N; i += n) {
           sum += col[static_cast<size_t>(i) * static_cast<size_t>(OC)];
         }
         float partial = static_cast<float>(sum);
         float total =
           sycl::reduce_over_group(grp, partial, sycl::plus<float>{});
         if (tid == 0) {
           dbias[o] = total;
         }
       });
   }).wait();
}

// ---------------------------------------------------------------------------
// Variant 1: oneMath GEMMs + faster bias kernel (mirrors matmul_backward1).

static void matmul_backward1(sycl::queue &q, float *dinp, float *dweight,
                             float *dbias, const float *dout, const float *inp,
                             const float *weight, int B, int T, int C, int OC) {
  const float alpha = 1.0f;
  const float beta = 1.0f; // gradients accumulate (+=), buffers are zeroed
  const std::int64_t BT = static_cast<std::int64_t>(B) * T;
  const std::int64_t Ci = C;
  const std::int64_t OCi = OC;

  // dinp (BT x C, row-major) = dout (BT x OC) * weight (OC x C)
  sycl::event e_dinp = oneapi::math::blas::row_major::gemm(
      q, oneapi::math::transpose::nontrans, oneapi::math::transpose::nontrans,
      BT, Ci, OCi, alpha, dout, OCi, weight, Ci, beta, dinp, Ci);
  // dweight (OC x C, row-major) = dout^T (OC x BT) * inp (BT x C)
  sycl::event e_dw = oneapi::math::blas::row_major::gemm(
      q, oneapi::math::transpose::trans, oneapi::math::transpose::nontrans,
      OCi, Ci, BT, alpha, dout, OCi, inp, Ci, beta, dweight, Ci);
  e_dinp.wait();
  e_dw.wait();

  if (dbias != nullptr) {
    launch_bias_faster(q, dbias, dout, B, T, OC);
  }
}

static void matmul_backward(int kernel_num, sycl::queue &q, float *dinp,
                            float *dweight, float *dbias, const float *dout,
                            const float *inp, const float *weight, int B, int T,
                            int C, int OC) {
  switch (kernel_num) {
    case 1:
      matmul_backward1(q, dinp, dweight, dbias, dout, inp, weight, B, T, C, OC);
      break;
    default:
      std::printf("Invalid kernel number\n");
      std::exit(1);
  }
  // Silence unused-function warning while keeping the naive kernel part of
  // the translation unit as its own SYCL kernel.
  (void)launch_bias_naive;
}

// ---------------------------------------------------------------------------
// Contract main: <input> <output> B T C OC; runs every variant and writes
// each variant's outputs in order as raw float32.
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
  float *dout = static_cast<float *>(
      std::malloc(sizeof(float) * static_cast<size_t>(B) * T * OC));
  if (std::fread(dout, sizeof(float), static_cast<size_t>(B) * T * OC, fin) !=
      static_cast<size_t>(B) * T * OC) {
    std::fprintf(stderr, "input too short at dout\n");
    return 2;
  }
  float *inp = static_cast<float *>(
      std::malloc(sizeof(float) * static_cast<size_t>(B) * T * C));
  if (std::fread(inp, sizeof(float), static_cast<size_t>(B) * T * C, fin) !=
      static_cast<size_t>(B) * T * C) {
    std::fprintf(stderr, "input too short at inp\n");
    return 2;
  }
  float *weight = static_cast<float *>(
      std::malloc(sizeof(float) * static_cast<size_t>(OC) * C));
  if (std::fread(weight, sizeof(float), static_cast<size_t>(OC) * C, fin) !=
      static_cast<size_t>(OC) * C) {
    std::fprintf(stderr, "input too short at weight\n");
    return 2;
  }
  float *ones = static_cast<float *>(
      std::malloc(sizeof(float) * static_cast<size_t>(OC)));
  if (std::fread(ones, sizeof(float), static_cast<size_t>(OC), fin) !=
      static_cast<size_t>(OC)) {
    std::fprintf(stderr, "input too short at ones\n");
    return 2;
  }
  std::fclose(fin);
  FILE *fout = std::fopen(argv[2], "wb");
  if (!fout) {
    std::perror(argv[2]);
    return 2;
  }

  sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order{}};

  float *d_dout =
      sycl::malloc_device<float>(static_cast<size_t>(B) * T * OC, q);
  q.memcpy(d_dout, dout, sizeof(float) * static_cast<size_t>(B) * T * OC)
      .wait();
  float *d_inp =
      sycl::malloc_device<float>(static_cast<size_t>(B) * T * C, q);
  q.memcpy(d_inp, inp, sizeof(float) * static_cast<size_t>(B) * T * C).wait();
  float *d_weight =
      sycl::malloc_device<float>(static_cast<size_t>(OC) * C, q);
  q.memcpy(d_weight, weight, sizeof(float) * static_cast<size_t>(OC) * C)
      .wait();
  float *d_ones = sycl::malloc_device<float>(static_cast<size_t>(OC), q);
  q.memcpy(d_ones, ones, sizeof(float) * static_cast<size_t>(OC)).wait();

  float *dinp = static_cast<float *>(
      std::malloc(sizeof(float) * static_cast<size_t>(B) * T * C));
  float *d_dinp =
      sycl::malloc_device<float>(static_cast<size_t>(B) * T * C, q);
  float *dweight = static_cast<float *>(
      std::malloc(sizeof(float) * static_cast<size_t>(OC) * C));
  float *d_dweight =
      sycl::malloc_device<float>(static_cast<size_t>(OC) * C, q);
  float *dbias =
      static_cast<float *>(std::malloc(sizeof(float) * static_cast<size_t>(OC)));
  float *d_dbias = sycl::malloc_device<float>(static_cast<size_t>(OC), q);

  const int variants[] = {1};
  for (int vi = 0; vi < 1; ++vi) {
    const int k = variants[vi];
    q.memset(d_dinp, 0, sizeof(float) * static_cast<size_t>(B) * T * C).wait();
    q.memset(d_dweight, 0, sizeof(float) * static_cast<size_t>(OC) * C).wait();
    q.memset(d_dbias, 0, sizeof(float) * static_cast<size_t>(OC)).wait();
    matmul_backward(k, q, d_dinp, d_dweight, d_dbias, d_dout, d_inp, d_weight,
                    B, T, C, OC);
    q.wait();
    q.memcpy(dinp, d_dinp, sizeof(float) * static_cast<size_t>(B) * T * C)
        .wait();
    std::fwrite(dinp, sizeof(float), static_cast<size_t>(B) * T * C, fout);
    q.memcpy(dweight, d_dweight, sizeof(float) * static_cast<size_t>(OC) * C)
        .wait();
    std::fwrite(dweight, sizeof(float), static_cast<size_t>(OC) * C, fout);
    q.memcpy(dbias, d_dbias, sizeof(float) * static_cast<size_t>(OC)).wait();
    std::fwrite(dbias, sizeof(float), static_cast<size_t>(OC), fout);
  }
  std::fclose(fout);

  sycl::free(d_dout, q);
  sycl::free(d_inp, q);
  sycl::free(d_weight, q);
  sycl::free(d_ones, q);
  sycl::free(d_dinp, q);
  sycl::free(d_dweight, q);
  sycl::free(d_dbias, q);
  std::free(dout);
  std::free(inp);
  std::free(weight);
  std::free(ones);
  std::free(dinp);
  std::free(dweight);
  std::free(dbias);
  return 0;
}
