// Residual forward pass: out = inp1 + inp2 (fp32 build).
// Idiomatic SYCL 2020 port of llm.c dev/cuda/residual_forward.cu.
// Kernel variants: 1 = scalar, 2 = 128-bit vectorized.
#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using floatX = float;

// 128-bit packed vector of 4 fp32 values (forces 128-bit loads/stores).
struct alignas(16) F128 {
  static constexpr int size = 4;
  float v[size];
  float &operator[](int i) { return v[i]; }
  const float &operator[](int i) const { return v[i]; }
};

inline F128 load128cs(const float *p) {
  F128 r;
  for (int i = 0; i < F128::size; ++i) {
    r[i] = p[i];
  }
  return r;
}

inline void store128(float *p, const F128 &v) {
  for (int i = 0; i < F128::size; ++i) {
    p[i] = v[i];
  }
}

template <typename T>
inline T ceil_div(T dividend, T divisor) {
  return (dividend + divisor - 1) / divisor;
}

class ResidualForwardK1;
class ResidualForwardK2;

constexpr std::size_t kBlockSize = 256;
constexpr int kPackSize = F128::size;

void residual_forward1(sycl::queue &q, floatX *out, const floatX *inp1,
                       const floatX *inp2, std::size_t N) {
  if (N == 0) return;
  const std::size_t local = kBlockSize;
  const std::size_t nblocks = ceil_div(N, local);
  q.parallel_for<ResidualForwardK1>(
       sycl::nd_range<1>(nblocks * local, local),
       [=](sycl::nd_item<1> it) {
         const std::size_t idx = it.get_global_id(0);
         if (idx < N) {
           out[idx] = inp1[idx] + inp2[idx];
         }
       })
      .wait();
}

void residual_forward2(sycl::queue &q, floatX *out, const floatX *inp1,
                       const floatX *inp2, std::size_t N) {
  if (N == 0) return;
  const std::size_t local = kBlockSize;
  const std::size_t packs = ceil_div(N, static_cast<std::size_t>(kPackSize));
  const std::size_t nblocks = ceil_div(packs, local);
  q.parallel_for<ResidualForwardK2>(
       sycl::nd_range<1>(nblocks * local, local),
       [=](sycl::nd_item<1> it) {
         const std::size_t p = it.get_global_id(0);
         if (p >= packs) return;
         const std::size_t idx = p * static_cast<std::size_t>(kPackSize);
         if (idx + kPackSize <= N) {
           F128 a = load128cs(inp1 + idx);
           F128 b = load128cs(inp2 + idx);
           F128 c;
           for (int k = 0; k < F128::size; ++k) {
             c[k] = a[k] + b[k];
           }
           store128(out + idx, c);
         } else {
           for (std::size_t k = 0; k < N - idx; ++k) {
             out[idx + k] = inp1[idx + k] + inp2[idx + k];
           }
         }
       })
      .wait();
}

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
      std::printf("Invalid kernel number\n");
      std::exit(1);
  }
}

int main(int argc, char **argv) {
  std::srand(0);
  if (argc != 6) {
    std::fprintf(stderr, "usage: %s <input> <output> B T C\n", argv[0]);
    return 2;
  }
  const long B = std::atol(argv[3]);
  const long T = std::atol(argv[4]);
  const long C = std::atol(argv[5]);
  const std::size_t N = static_cast<std::size_t>(B) *
                        static_cast<std::size_t>(T) *
                        static_cast<std::size_t>(C);
  const std::size_t bytes = N * sizeof(float);

  FILE *fin = std::fopen(argv[1], "rb");
  if (!fin) {
    std::perror(argv[1]);
    return 2;
  }
  float *inp1 = static_cast<float *>(std::malloc(bytes == 0 ? 1 : bytes));
  if (std::fread(inp1, sizeof(float), N, fin) != N) {
    std::fprintf(stderr, "input too short at inp1\n");
    return 2;
  }
  float *inp2 = static_cast<float *>(std::malloc(bytes == 0 ? 1 : bytes));
  if (std::fread(inp2, sizeof(float), N, fin) != N) {
    std::fprintf(stderr, "input too short at inp2\n");
    return 2;
  }
  std::fclose(fin);
  FILE *fout = std::fopen(argv[2], "wb");
  if (!fout) {
    std::perror(argv[2]);
    return 2;
  }

  sycl::queue q;
  float *d_inp1 = sycl::malloc_device<float>(N == 0 ? 1 : N, q);
  float *d_inp2 = sycl::malloc_device<float>(N == 0 ? 1 : N, q);
  float *d_out = sycl::malloc_device<float>(N == 0 ? 1 : N, q);
  float *out = static_cast<float *>(std::malloc(bytes == 0 ? 1 : bytes));

  if (N > 0) {
    q.memcpy(d_inp1, inp1, bytes).wait();
    q.memcpy(d_inp2, inp2, bytes).wait();
  }

  const int variants[] = {1, 2};
  for (int vi = 0; vi < 2; ++vi) {
    const int k = variants[vi];
    if (N > 0) {
      q.memset(d_out, 0, bytes).wait();
    }
    residual_forward(k, q, d_out, d_inp1, d_inp2, N);
    q.wait();
    if (N > 0) {
      q.memcpy(out, d_out, bytes).wait();
    }
    std::fwrite(out, sizeof(float), N, fout);
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
