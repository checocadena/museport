#include <sycl/sycl.hpp>
#include <oneapi/math.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cstdint>

// Bias backward: sum dout over (B,T) for each output channel.
// dout layout is (B*T) x OC row-major: dout[(b*T+t)*OC + o].
// Kept as its own SYCL kernel (naive variant, 1-D range).
static void bias_naive_kernel(sycl::queue &q, float *dbias, const float *dout,
                              int B, int T, int OC) {
  const std::size_t n = static_cast<std::size_t>(OC);
  q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
     int o = static_cast<int>(idx[0]);
     double sum = 0.0;
     for (int b = 0; b < B; ++b) {
       for (int t = 0; t < T; ++t) {
         sum += dout[static_cast<std::size_t>(b) * static_cast<std::size_t>(T) *
                       static_cast<std::size_t>(OC) +
                     static_cast<std::size_t>(t) * static_cast<std::size_t>(OC) +
                     static_cast<std::size_t>(o)];
       }
     }
     dbias[o] = static_cast<float>(sum);
   }).wait();
}

// Faster bias backward: one work-group per output channel, thread coarsening
// over B*T plus a local-memory tree reduction. 1-D nd_range.
static void bias_faster_kernel(sycl::queue &q, float *dbias, const float *dout,
                               int B, int T, int OC) {
  constexpr int block_size = 256;
  const std::size_t global = static_cast<std::size_t>(OC) * block_size;
  const std::size_t local = static_cast<std::size_t>(block_size);
  const int n = B * T;

  q.submit([&](sycl::handler &h) {
     sycl::local_accessor<float, 1> shared(sycl::range<1>(block_size), h);
     h.parallel_for(
         sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(local)),
         [=](sycl::nd_item<1> it) {
           const int o = static_cast<int>(it.get_group(0));
           const int tid = static_cast<int>(it.get_local_id(0));
           const float *x = dout + o;
           double sum = 0.0;
           for (int i = tid; i < n; i += block_size) {
             sum += x[static_cast<std::size_t>(i) * static_cast<std::size_t>(OC)];
           }
           shared[tid] = static_cast<float>(sum);
           it.barrier(sycl::access::fence_space::local_space);
           for (int stride = block_size / 2; stride >= 1; stride /= 2) {
             it.barrier(sycl::access::fence_space::local_space);
             if (tid < stride) {
               shared[tid] += shared[tid + stride];
             }
           }
           if (tid == 0) {
             dbias[o] = shared[0];
           }
         });
   }).wait();
}

// version 1: oneMath column-major GEMMs (cuBLAS mapping is 1:1 because cuBLAS
// is itself column-major) plus the faster bias SYCL kernel.
static void matmul_backward1(sycl::queue &q, float *dinp, float *dweight,
                             float *dbias, const float *dout, const float *inp,
                             const float *weight, const float *ones, int B,
                             int T, int C, int OC) {
  (void)ones;
  const float alpha = 1.0f;
  // Gradients accumulate, so use beta = 1.0 (outputs are zeroed by caller).
  const float beta = 1.0f;
  const std::int64_t bt = static_cast<std::int64_t>(B) * static_cast<std::int64_t>(T);
  const std::int64_t cc = static_cast<std::int64_t>(C);
  const std::int64_t oc = static_cast<std::int64_t>(OC);

  // backward to input:
  //   cublasSgemm(N, N, C, B*T, OC, weight[C x OC], dout[OC x B*T], dinp[C x B*T])
  oneapi::math::blas::column_major::gemm(
      q, oneapi::math::transpose::nontrans, oneapi::math::transpose::nontrans,
      cc, bt, oc, alpha, weight, cc, dout, oc, beta, dinp, cc)
      .wait();

  // backward to weight:
  //   cublasSgemm(N, T, C, OC, B*T, inp[C x B*T], dout[OC x B*T]^T, dweight[C x OC])
  oneapi::math::blas::column_major::gemm(
      q, oneapi::math::transpose::nontrans, oneapi::math::transpose::trans, cc,
      oc, bt, alpha, inp, cc, dout, oc, beta, dweight, cc)
      .wait();

  if (dbias != nullptr) {
    bias_faster_kernel(q, dbias, dout, B, T, OC);
  }
}

static void matmul_backward(int kernel_num, sycl::queue &q, float *dinp,
                            float *dweight, float *dbias, float *dout,
                            float *inp, float *weight, float *ones, int B,
                            int T, int C, int OC) {
  switch (kernel_num) {
  case 1:
    matmul_backward1(q, dinp, dweight, dbias, dout, inp, weight, ones, B, T, C,
                     OC);
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
  float *dout =
      static_cast<float *>(std::malloc(sizeof(float) * (std::size_t)(B * T * OC)));
  if (std::fread(dout, sizeof(float), (std::size_t)(B * T * OC), fin) !=
      (std::size_t)(B * T * OC)) {
    std::fprintf(stderr, "input too short at dout\n");
    return 2;
  }
  float *inp =
      static_cast<float *>(std::malloc(sizeof(float) * (std::size_t)(B * T * C)));
  if (std::fread(inp, sizeof(float), (std::size_t)(B * T * C), fin) !=
      (std::size_t)(B * T * C)) {
    std::fprintf(stderr, "input too short at inp\n");
    return 2;
  }
  float *weight =
      static_cast<float *>(std::malloc(sizeof(float) * (std::size_t)(OC * C)));
  if (std::fread(weight, sizeof(float), (std::size_t)(OC * C), fin) !=
      (std::size_t)(OC * C)) {
    std::fprintf(stderr, "input too short at weight\n");
    return 2;
  }
  float *ones =
      static_cast<float *>(std::malloc(sizeof(float) * (std::size_t)(OC)));
  if (std::fread(ones, sizeof(float), (std::size_t)(OC), fin) != (std::size_t)(OC)) {
    std::fprintf(stderr, "input too short at ones\n");
    return 2;
  }
  std::fclose(fin);

  FILE *fout = std::fopen(argv[2], "wb");
  if (!fout) {
    std::perror(argv[2]);
    return 2;
  }

  sycl::queue q{sycl::property::queue::in_order{}};

  float *d_dout = sycl::malloc_device<float>((std::size_t)(B * T * OC), q);
  q.memcpy(d_dout, dout, sizeof(float) * (std::size_t)(B * T * OC)).wait();
  float *d_inp = sycl::malloc_device<float>((std::size_t)(B * T * C), q);
  q.memcpy(d_inp, inp, sizeof(float) * (std::size_t)(B * T * C)).wait();
  float *d_weight = sycl::malloc_device<float>((std::size_t)(OC * C), q);
  q.memcpy(d_weight, weight, sizeof(float) * (std::size_t)(OC * C)).wait();
  float *d_ones = sycl::malloc_device<float>((std::size_t)(OC), q);
  q.memcpy(d_ones, ones, sizeof(float) * (std::size_t)(OC)).wait();

  float *dinp =
      static_cast<float *>(std::malloc(sizeof(float) * (std::size_t)(B * T * C)));
  float *d_dinp = sycl::malloc_device<float>((std::size_t)(B * T * C), q);
  float *dweight =
      static_cast<float *>(std::malloc(sizeof(float) * (std::size_t)(OC * C)));
  float *d_dweight = sycl::malloc_device<float>((std::size_t)(OC * C), q);
  float *dbias = static_cast<float *>(std::malloc(sizeof(float) * (std::size_t)(OC)));
  float *d_dbias = sycl::malloc_device<float>((std::size_t)(OC), q);

  // Reference the naive kernel so both original kernels stay present as SYCL
  // kernels; the timed variant below uses the faster one, as in CUDA.
  (void)&bias_naive_kernel;

  const int variants[] = {1};
  for (int vi = 0; vi < 1; ++vi) {
    const int k = variants[vi];
    q.memset(d_dinp, 0, sizeof(float) * (std::size_t)(B * T * C)).wait();
    q.memset(d_dweight, 0, sizeof(float) * (std::size_t)(OC * C)).wait();
    q.memset(d_dbias, 0, sizeof(float) * (std::size_t)(OC)).wait();
    matmul_backward(k, q, d_dinp, d_dweight, d_dbias, d_dout, d_inp, d_weight,
                    d_ones, B, T, C, OC);
    q.wait();
    q.memcpy(dinp, d_dinp, sizeof(float) * (std::size_t)(B * T * C)).wait();
    std::fwrite(dinp, sizeof(float), (std::size_t)(B * T * C), fout);
    q.memcpy(dweight, d_dweight, sizeof(float) * (std::size_t)(OC * C)).wait();
    std::fwrite(dweight, sizeof(float), (std::size_t)(OC * C), fout);
    q.memcpy(dbias, d_dbias, sizeof(float) * (std::size_t)(OC)).wait();
    std::fwrite(dbias, sizeof(float), (std::size_t)(OC), fout);
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
