#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <sycl/sycl.hpp>

using floatX = float;

// ceil_div helper (host + device)
template <typename T>
inline T ceil_div(T dividend, T divisor) {
  return (dividend + divisor - 1) / divisor;
}

class ResidualForwardKernel1;
class ResidualForwardKernel2;

static void residual_forward1(sycl::queue &q, floatX *out, const floatX *inp1,
                              const floatX *inp2, int N) {
  if (N <= 0) {
    return;
  }
  constexpr int block_size = 256;
  const size_t n = static_cast<size_t>(N);
  const size_t global = ceil_div(n, static_cast<size_t>(block_size)) *
                        static_cast<size_t>(block_size);
  q.parallel_for<ResidualForwardKernel1>(
      sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(block_size)),
      [=](sycl::nd_item<1> it) {
        const size_t idx = it.get_global_id(0);
        if (idx < n) {
          out[idx] = inp1[idx] + inp2[idx];
        }
      });
}

static void residual_forward2(sycl::queue &q, floatX *out, const floatX *inp1,
                              const floatX *inp2, int N) {
  if (N <= 0) {
    return;
  }
  constexpr int block_size = 256;
  constexpr int vec_width = 4; // 128-bit = 4 x fp32
  const size_t n = static_cast<size_t>(N);
  const size_t nvec = ceil_div(n, static_cast<size_t>(vec_width));
  const size_t global = ceil_div(nvec, static_cast<size_t>(block_size)) *
                        static_cast<size_t>(block_size);
  q.parallel_for<ResidualForwardKernel2>(
      sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(block_size)),
      [=](sycl::nd_item<1> it) {
        const size_t vidx = it.get_global_id(0);
        const size_t idx = vidx * static_cast<size_t>(vec_width);
        if (idx + static_cast<size_t>(vec_width) <= n) {
          // Aligned 128-bit vector path.
          const auto *p1 =
              reinterpret_cast<const sycl::float4 *>(inp1 + idx);
          const auto *p2 =
              reinterpret_cast<const sycl::float4 *>(inp2 + idx);
          auto *po = reinterpret_cast<sycl::float4 *>(out + idx);
          sycl::float4 a = *p1;
          sycl::float4 b = *p2;
          *po = a + b;
        } else if (idx < n) {
          // Scalar tail for N not divisible by 4.
          for (int k = 0; k < vec_width; ++k) {
            const size_t j = idx + static_cast<size_t>(k);
            if (j < n) {
              out[j] = inp1[j] + inp2[j];
            }
          }
        }
      });
}

static void residual_forward(sycl::queue &q, int kernel_num, floatX *out,
                             const floatX *inp1, const floatX *inp2, int N) {
  switch (kernel_num) {
  case 1:
    residual_forward1(q, out, inp1, inp2, N);
    break;
  case 2:
    residual_forward2(q, out, inp1, inp2, N);
    break;
  default:
    std::printf("Invalid kernel number\n");
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
  const size_t total = static_cast<size_t>(B) * static_cast<size_t>(T) *
                       static_cast<size_t>(C);

  FILE *fin = std::fopen(argv[1], "rb");
  if (!fin) {
    std::perror(argv[1]);
    return 2;
  }
  float *inp1 = static_cast<float *>(std::malloc(sizeof(float) * total));
  float *inp2 = static_cast<float *>(std::malloc(sizeof(float) * total));
  float *out = static_cast<float *>(std::malloc(sizeof(float) * total));
  if (!inp1 || !inp2 || !out) {
    std::fprintf(stderr, "host malloc failed\n");
    return 2;
  }
  if (std::fread(inp1, sizeof(float), total, fin) != total) {
    std::fprintf(stderr, "input too short at inp1\n");
    return 2;
  }
  if (std::fread(inp2, sizeof(float), total, fin) != total) {
    std::fprintf(stderr, "input too short at inp2\n");
    return 2;
  }
  std::fclose(fin);
  fin = nullptr;

  FILE *fout = std::fopen(argv[2], "wb");
  if (!fout) {
    std::perror(argv[2]);
    return 2;
  }

  sycl::queue q{sycl::default_selector_v};
  float *d_inp1 = sycl::malloc_device<float>(total, q);
  float *d_inp2 = sycl::malloc_device<float>(total, q);
  float *d_out = sycl::malloc_device<float>(total, q);
  if (total > 0 && (!d_inp1 || !d_inp2 || !d_out)) {
    std::fprintf(stderr, "device malloc failed\n");
    return 2;
  }
  if (total > 0) {
    q.memcpy(d_inp1, inp1, sizeof(float) * total).wait();
    q.memcpy(d_inp2, inp2, sizeof(float) * total).wait();
  }

  const int variants[] = {1, 2};
  for (int vi = 0; vi < 2; ++vi) {
    const int k = variants[vi];
    if (total > 0) {
      q.memset(d_out, 0, sizeof(float) * total).wait();
    }
    residual_forward(q, k, d_out, d_inp1, d_inp2,
                     static_cast<int>(B * T * C));
    q.wait();
    if (total > 0) {
      q.memcpy(out, d_out, sizeof(float) * total).wait();
    }
    std::fwrite(out, sizeof(float), total, fout);
  }

  std::fclose(fout);
  sycl::free(d_inp1, q);
  sycl::free(d_inp2, q);
  sycl::free(d_out, q);
  std::free(inp1);
  std::free(inp2);
  std::free(out);
  return 0;
}
