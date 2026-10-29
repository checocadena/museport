#include <sycl/sycl.hpp>

#include <cstdio>
#include <cstdlib>

// fp32 build
using floatX = float;

// Each kernel variant is its own SYCL kernel (named types below).

class ResidualForwardK1;
class ResidualForwardK2;

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

void residual_forward1(sycl::queue &q, floatX *out, const floatX *inp1,
                       const floatX *inp2, int N, int block_size) {
  const int grid_size = ceil_div(N, block_size);
  const std::size_t total = static_cast<std::size_t>(grid_size) *
                            static_cast<std::size_t>(block_size);
  q.parallel_for<ResidualForwardK1>(
       sycl::nd_range<1>(sycl::range<1>(total),
                         sycl::range<1>(static_cast<std::size_t>(block_size))),
       [=](sycl::nd_item<1> it) {
         const int idx = static_cast<int>(it.get_global_id(0));
         if (idx < N) {
           out[idx] = inp1[idx] + inp2[idx];
         }
       })
      .wait();
}

void residual_forward2(sycl::queue &q, floatX *out, const floatX *inp1,
                       const floatX *inp2, int N, int block_size) {
  constexpr int pack = 4; // 128-bit / 32-bit
  const int packed_N = ceil_div(N, pack);
  const int grid_size = ceil_div(packed_N, block_size);
  const std::size_t total = static_cast<std::size_t>(grid_size) *
                            static_cast<std::size_t>(block_size);
  q.parallel_for<ResidualForwardK2>(
       sycl::nd_range<1>(sycl::range<1>(total),
                         sycl::range<1>(static_cast<std::size_t>(block_size))),
       [=](sycl::nd_item<1> it) {
         const int p = static_cast<int>(it.get_global_id(0));
         if (p >= packed_N) {
           return;
         }
         const int idx = p * pack;
         // 128-bit vector load / store.
         sycl::vec<float, 4> a =
             reinterpret_cast<const sycl::vec<float, 4> *>(inp1)[p];
         sycl::vec<float, 4> b =
             reinterpret_cast<const sycl::vec<float, 4> *>(inp2)[p];
         sycl::vec<float, 4> c(a[0] + b[0], a[1] + b[1], a[2] + b[2],
                                a[3] + b[3]);
         if (idx + pack <= N) {
           reinterpret_cast<sycl::vec<float, 4> *>(out)[p] = c;
         } else {
           for (int k = 0; k < pack; ++k) {
             const int i = idx + k;
             if (i < N) {
               out[i] = c[k];
             }
           }
         }
       })
      .wait();
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
  const std::size_t N = static_cast<std::size_t>(B) *
                        static_cast<std::size_t>(T) *
                        static_cast<std::size_t>(C);

  FILE *fin = std::fopen(argv[1], "rb");
  if (!fin) {
    std::perror(argv[1]);
    return 2;
  }
  float *inp1 = static_cast<float *>(std::malloc(sizeof(float) * N));
  float *inp2 = static_cast<float *>(std::malloc(sizeof(float) * N));
  float *out = static_cast<float *>(std::malloc(sizeof(float) * N));
  if (!inp1 || !inp2 || !out) {
    std::fprintf(stderr, "host malloc failed\n");
    return 2;
  }
  if (std::fread(inp1, sizeof(float), N, fin) != N) {
    std::fprintf(stderr, "input too short at inp1\n");
    return 2;
  }
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

  try {
    sycl::queue q{sycl::default_selector_v};

    float *d_inp1 = sycl::malloc_device<float>(N, q);
    float *d_inp2 = sycl::malloc_device<float>(N, q);
    float *d_out = sycl::malloc_device<float>(N, q);
    if (!d_inp1 || !d_inp2 || !d_out) {
      std::fprintf(stderr, "device malloc failed\n");
      return 2;
    }

    q.memcpy(d_inp1, inp1, sizeof(float) * N).wait();
    q.memcpy(d_inp2, inp2, sizeof(float) * N).wait();

    const int variants[] = {1, 2};
    for (int vi = 0; vi < 2; ++vi) {
      const int k = variants[vi];
      q.fill(d_out, 0.0f, N).wait();
      residual_forward(q, k, d_out, d_inp1, d_inp2, static_cast<int>(N), 256);
      q.wait();
      q.memcpy(out, d_out, sizeof(float) * N).wait();
      std::fwrite(out, sizeof(float), N, fout);
    }

    sycl::free(d_inp1, q);
    sycl::free(d_inp2, q);
    sycl::free(d_out, q);
  } catch (sycl::exception const &e) {
    std::fprintf(stderr, "SYCL exception: %s\n", e.what());
    return 1;
  }

  std::fclose(fout);
  std::free(inp1);
  std::free(inp2);
  std::free(out);
  return 0;
}
