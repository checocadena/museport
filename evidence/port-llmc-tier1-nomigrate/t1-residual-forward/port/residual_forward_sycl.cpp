#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <sycl/sycl.hpp>

class ResidualForwardKernel1;
class ResidualForwardKernel2;

static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

void residual_forward1(sycl::queue &q, float *out, const float *inp1,
                       const float *inp2, int N, int block_size) {
  const int nblocks = ceil_div(N, block_size);
  const std::size_t global = static_cast<std::size_t>(nblocks) *
                             static_cast<std::size_t>(block_size);
  q.parallel_for<ResidualForwardKernel1>(
      sycl::nd_range<1>(sycl::range<1>(global),
                        sycl::range<1>(static_cast<std::size_t>(block_size))),
      [=](sycl::nd_item<1> it) {
        const int idx = static_cast<int>(it.get_global_id(0));
        if (idx < N) {
          out[idx] = inp1[idx] + inp2[idx];
        }
      });
}

void residual_forward2(sycl::queue &q, float *out, const float *inp1,
                       const float *inp2, int N, int block_size) {
  constexpr int vec = 4;
  const int nvec = ceil_div(N, vec);
  const int nblocks = ceil_div(nvec, block_size);
  const std::size_t global = static_cast<std::size_t>(nblocks) *
                             static_cast<std::size_t>(block_size);
  using vec_t = sycl::vec<float, 4>;
  q.parallel_for<ResidualForwardKernel2>(
      sycl::nd_range<1>(sycl::range<1>(global),
                        sycl::range<1>(static_cast<std::size_t>(block_size))),
      [=](sycl::nd_item<1> it) {
        const int vidx = static_cast<int>(it.get_global_id(0));
        const int idx = vidx * vec;
        if (idx + vec <= N) {
          const vec_t a = reinterpret_cast<const vec_t *>(inp1)[vidx];
          const vec_t b = reinterpret_cast<const vec_t *>(inp2)[vidx];
          vec_t c{a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]};
          reinterpret_cast<vec_t *>(out)[vidx] = c;
        } else {
          for (int k = 0; k < vec; ++k) {
            const int i = idx + k;
            if (i < N) {
              out[i] = inp1[i] + inp2[i];
            }
          }
        }
      });
}

void residual_forward(sycl::queue &q, int kernel_num, float *out,
                      const float *inp1, const float *inp2, int N,
                      int block_size) {
  switch (kernel_num) {
    case 1:
      residual_forward1(q, out, inp1, inp2, N, block_size);
      break;
    case 2:
      residual_forward2(q, out, inp1, inp2, N, block_size);
      break;
    default:
      std::fprintf(stderr, "Invalid kernel number\n");
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
    q.memset(d_out, 0, sizeof(float) * N).wait();
    residual_forward(q, k, d_out, d_inp1, d_inp2, static_cast<int>(N), 256);
    q.wait();
    q.memcpy(out, d_out, sizeof(float) * N).wait();
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
