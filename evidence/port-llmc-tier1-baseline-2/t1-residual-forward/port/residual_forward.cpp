/*
Kernels for residual forward pass (idiomatic SYCL 2020 port, fp32 build).

version 1 is naive elementwise add.
version 2 packs input into 128-bit (float4) memory accesses.
*/

#include <sycl/sycl.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstddef>

using floatX = float;
constexpr int kPack128 = 4; // floats per 128-bit packed access

template <typename T>
inline T ceil_div(T dividend, T divisor) {
  return (dividend + divisor - 1) / divisor;
}

class ResidualForwardKernel1;
class ResidualForwardKernel2;

void residual_forward1(sycl::queue &q, floatX *out, const floatX *inp1,
                       const floatX *inp2, int N, int block_size) {
  if (N <= 0) {
    return;
  }
  const std::size_t local = static_cast<std::size_t>(block_size);
  const std::size_t grid =
      static_cast<std::size_t>(ceil_div(N, block_size));
  const std::size_t global = grid * local;
  q.parallel_for<ResidualForwardKernel1>(
      sycl::nd_range<1>{sycl::range<1>{global}, sycl::range<1>{local}},
      [=](sycl::nd_item<1> it) {
        const std::size_t idx = it.get_global_id(0);
        if (idx < static_cast<std::size_t>(N)) {
          out[idx] = inp1[idx] + inp2[idx];
        }
      });
}

void residual_forward2(sycl::queue &q, floatX *out, const floatX *inp1,
                       const floatX *inp2, int N, int block_size) {
  if (N <= 0) {
    return;
  }
  constexpr int kPack = kPack128;
  const int n_packs = ceil_div(N, kPack);
  const std::size_t local = static_cast<std::size_t>(block_size);
  const std::size_t grid =
      static_cast<std::size_t>(ceil_div(n_packs, block_size));
  const std::size_t global = grid * local;
  q.parallel_for<ResidualForwardKernel2>(
      sycl::nd_range<1>{sycl::range<1>{global}, sycl::range<1>{local}},
      [=](sycl::nd_item<1> it) {
        const std::size_t pack_id = it.get_global_id(0);
        const std::size_t base = pack_id * static_cast<std::size_t>(kPack);
        if (base >= static_cast<std::size_t>(N)) {
          return;
        }
        if (base + kPack <= static_cast<std::size_t>(N)) {
          // 128-bit packed path.
          const auto *p1 =
              reinterpret_cast<const sycl::vec<float, kPack> *>(inp1);
          const auto *p2 =
              reinterpret_cast<const sycl::vec<float, kPack> *>(inp2);
          auto *po = reinterpret_cast<sycl::vec<float, kPack> *>(out);
          po[pack_id] = p1[pack_id] + p2[pack_id];
        } else {
          // Scalar tail for N not divisible by pack size.
          for (int k = 0; k < kPack; ++k) {
            const std::size_t i = base + static_cast<std::size_t>(k);
            if (i < static_cast<std::size_t>(N)) {
              out[i] = inp1[i] + inp2[i];
            }
          }
        }
      });
}

void residual_forward(sycl::queue &q, int kernel_num, floatX *out,
                      const floatX *inp1, const floatX *inp2, int N,
                      int block_size) {
  switch (kernel_num) {
    case 1:
      residual_forward1(q, out, inp1, inp2, N, block_size);
      break;
    case 2:
      residual_forward2(q, out, inp1, inp2, N, block_size);
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
  const std::size_t count = static_cast<std::size_t>(B) *
                            static_cast<std::size_t>(T) *
                            static_cast<std::size_t>(C);
  const std::size_t bytes = count * sizeof(float);

  FILE *fin = std::fopen(argv[1], "rb");
  if (!fin) {
    std::perror(argv[1]);
    return 2;
  }
  float *inp1 = (float *)std::malloc(bytes == 0 ? 1 : bytes);
  float *inp2 = (float *)std::malloc(bytes == 0 ? 1 : bytes);
  float *out = (float *)std::malloc(bytes == 0 ? 1 : bytes);
  if (!inp1 || !inp2 || !out) {
    std::fprintf(stderr, "host malloc failed\n");
    return 2;
  }
  if (count > 0) {
    if (std::fread(inp1, sizeof(float), count, fin) != count) {
      std::fprintf(stderr, "input too short at inp1\n");
      return 2;
    }
    if (std::fread(inp2, sizeof(float), count, fin) != count) {
      std::fprintf(stderr, "input too short at inp2\n");
      return 2;
    }
  }
  std::fclose(fin);
  FILE *fout = std::fopen(argv[2], "wb");
  if (!fout) {
    std::perror(argv[2]);
    return 2;
  }

  sycl::queue q;
  floatX *d_inp1 = nullptr;
  floatX *d_inp2 = nullptr;
  floatX *d_out = nullptr;
  if (count > 0) {
    d_inp1 = sycl::malloc_device<floatX>(count, q);
    d_inp2 = sycl::malloc_device<floatX>(count, q);
    d_out = sycl::malloc_device<floatX>(count, q);
    if (!d_inp1 || !d_inp2 || !d_out) {
      std::fprintf(stderr, "device malloc failed\n");
      return 2;
    }
    q.memcpy(d_inp1, inp1, bytes).wait();
    q.memcpy(d_inp2, inp2, bytes).wait();
  }

  const int variants[] = {1, 2};
  const int n_variants = 2;
  const int N = B * T * C;
  for (int vi = 0; vi < n_variants; ++vi) {
    const int k = variants[vi];
    if (count > 0) {
      q.memset(d_out, 0, bytes).wait();
      residual_forward(q, k, d_out, d_inp1, d_inp2, N, 256);
      q.wait();
      q.memcpy(out, d_out, bytes).wait();
      std::fwrite(out, sizeof(float), count, fout);
    }
  }

  if (d_out != nullptr) {
    sycl::free(d_out, q);
  }
  if (d_inp1 != nullptr) {
    sycl::free(d_inp1, q);
  }
  if (d_inp2 != nullptr) {
    sycl::free(d_inp2, q);
  }
  std::fclose(fout);
  std::free(inp1);
  std::free(inp2);
  std::free(out);
  return 0;
}
