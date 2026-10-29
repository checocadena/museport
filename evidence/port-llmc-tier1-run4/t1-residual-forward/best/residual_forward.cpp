#include <sycl/sycl.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstddef>

// fp32 build: floatX is float
using floatX = float;

// ----------------------------------------------------------------------------
// SYCL kernels: one per variant, 1-D range launches (1-D work).

class ResidualForwardKernel1;
class ResidualForwardKernel2;
class ResidualForwardKernel2Tail;

void residual_forward1(sycl::queue &q, floatX *out, const floatX *inp1,
                       const floatX *inp2, std::size_t N) {
  if (N == 0) return;
  q.parallel_for<ResidualForwardKernel1>(sycl::range<1>(N),
                                         [=](sycl::id<1> i) {
                                           const std::size_t idx = i[0];
                                           out[idx] = inp1[idx] + inp2[idx];
                                         });
}

void residual_forward2(sycl::queue &q, floatX *out, const floatX *inp1,
                       const floatX *inp2, std::size_t N) {
  if (N == 0) return;
  constexpr std::size_t kWidth = 4; // 128-bit = 4 x fp32 per work-item
  const std::size_t nvec = N / kWidth;
  const std::size_t tail = N % kWidth;
  if (nvec > 0) {
    const auto *vin1 = reinterpret_cast<const sycl::vec<float, 4> *>(inp1);
    const auto *vin2 = reinterpret_cast<const sycl::vec<float, 4> *>(inp2);
    auto *vout = reinterpret_cast<sycl::vec<float, 4> *>(out);
    q.parallel_for<ResidualForwardKernel2>(sycl::range<1>(nvec),
                                           [=](sycl::id<1> i) {
                                             const std::size_t v = i[0];
                                             vout[v] = vin1[v] + vin2[v];
                                           });
  }
  if (tail > 0) {
    const std::size_t base = nvec * kWidth;
    q.parallel_for<ResidualForwardKernel2Tail>(sycl::range<1>(tail),
                                               [=](sycl::id<1> i) {
                                                 const std::size_t idx =
                                                     base + i[0];
                                                 out[idx] =
                                                     inp1[idx] + inp2[idx];
                                               });
  }
}

// kernel version dispatch
void residual_forward(int kernel_num, sycl::queue &q, floatX *out,
                      const floatX *inp1, const floatX *inp2, std::size_t N) {
  switch (kernel_num) {
    case 1:
      residual_forward1(q, out, inp1, inp2, N);
      break;
    case 2:
      residual_forward2(q, out, inp1, inp2, N);
      break;
    default:
      std::fprintf(stderr, "Invalid kernel number\n");
      std::exit(1);
  }
}

// ----------------------------------------------------------------------------
// Contract main: <input> <output> B T C; runs every kernel variant and
// writes each variant's outputs, in order, as raw float32.
int main(int argc, char **argv) {
  if (argc != 6) {
    std::fprintf(stderr, "usage: %s <input> <output> B T C\n", argv[0]);
    return 2;
  }
  const long B = std::atol(argv[3]);
  const long T = std::atol(argv[4]);
  const long C = std::atol(argv[5]);
  const std::size_t n =
      static_cast<std::size_t>(B) * static_cast<std::size_t>(T) *
      static_cast<std::size_t>(C);

  FILE *fin = std::fopen(argv[1], "rb");
  if (!fin) {
    std::perror(argv[1]);
    return 2;
  }
  float *inp1 = static_cast<float *>(std::malloc(sizeof(float) * n));
  float *inp2 = static_cast<float *>(std::malloc(sizeof(float) * n));
  float *out = static_cast<float *>(std::malloc(sizeof(float) * n));
  if (!inp1 || !inp2 || !out) {
    std::fprintf(stderr, "host malloc failed\n");
    return 2;
  }
  if (std::fread(inp1, sizeof(float), n, fin) != n) {
    std::fprintf(stderr, "input too short at inp1\n");
    return 2;
  }
  if (std::fread(inp2, sizeof(float), n, fin) != n) {
    std::fprintf(stderr, "input too short at inp2\n");
    return 2;
  }
  std::fclose(fin);

  FILE *fout = std::fopen(argv[2], "wb");
  if (!fout) {
    std::perror(argv[2]);
    return 2;
  }

  try {
    sycl::queue q{sycl::default_selector_v};

    float *d_inp1 = sycl::malloc_device<float>(n, q);
    float *d_inp2 = sycl::malloc_device<float>(n, q);
    float *d_out = sycl::malloc_device<float>(n, q);
    if (!d_inp1 || !d_inp2 || !d_out) {
      std::fprintf(stderr, "device malloc failed\n");
      return 2;
    }

    q.memcpy(d_inp1, inp1, sizeof(float) * n).wait();
    q.memcpy(d_inp2, inp2, sizeof(float) * n).wait();

    const int variants[] = {1, 2};
    for (int vi = 0; vi < 2; ++vi) {
      const int k = variants[vi];
      q.memset(d_out, 0, sizeof(float) * n).wait();
      residual_forward(k, q, d_out, d_inp1, d_inp2, n);
      q.wait();
      q.memcpy(out, d_out, sizeof(float) * n).wait();
      std::fwrite(out, sizeof(float), n, fout);
    }

    sycl::free(d_inp1, q);
    sycl::free(d_inp2, q);
    sycl::free(d_out, q);
  } catch (const sycl::exception &e) {
    std::fprintf(stderr, "SYCL error: %s\n", e.what());
    return 1;
  }

  std::fclose(fout);
  std::free(inp1);
  std::free(inp2);
  std::free(out);
  return 0;
}
