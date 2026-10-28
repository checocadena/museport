#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstddef>

namespace {

template <typename T>
T ceil_div(T dividend, T divisor) {
  return (dividend + divisor - 1) / divisor;
}

// Variant 1: naive elementwise add, one element per work-item (1-D range).
void residual_forward1(sycl::queue &q, float *out, const float *inp1,
                       const float *inp2, std::size_t n) {
  if (n == 0) {
    return;
  }
  constexpr std::size_t wg = 256;
  const std::size_t n_groups = ceil_div(n, wg);
  q.parallel_for(sycl::nd_range<1>{sycl::range<1>{n_groups * wg}, sycl::range<1>{wg}},
                 [=](sycl::nd_item<1> it) {
                   const std::size_t idx = it.get_global_id(0);
                   if (idx < n) {
                     out[idx] = inp1[idx] + inp2[idx];
                   }
                 })
      .wait();
}

// Variant 2: 128-bit vectorized add, one float4 per work-item (1-D range).
// Falls back to scalar for a non-multiple-of-4 tail so any N is exact.
void residual_forward2(sycl::queue &q, float *out, const float *inp1,
                       const float *inp2, std::size_t n) {
  if (n == 0) {
    return;
  }
  constexpr std::size_t vec = 4;
  const std::size_t n_vec = ceil_div(n, vec);
  constexpr std::size_t wg = 256;
  const std::size_t n_groups = ceil_div(n_vec, wg);
  q.parallel_for(
         sycl::nd_range<1>{sycl::range<1>{n_groups * wg}, sycl::range<1>{wg}},
         [=](sycl::nd_item<1> it) {
           const std::size_t vid = it.get_global_id(0);
           if (vid >= n_vec) {
             return;
           }
           const std::size_t base = vid * vec;
           if (base + vec <= n) {
             const auto *p_in1 =
                 reinterpret_cast<const sycl::vec<float, 4> *>(inp1);
             const auto *p_in2 =
                 reinterpret_cast<const sycl::vec<float, 4> *>(inp2);
             auto *p_out = reinterpret_cast<sycl::vec<float, 4> *>(out);
             p_out[vid] = p_in1[vid] + p_in2[vid];
           } else {
             for (std::size_t k = 0; k < vec && base + k < n; ++k) {
               out[base + k] = inp1[base + k] + inp2[base + k];
             }
           }
         })
      .wait();
}

void residual_forward(int kernel_num, sycl::queue &q, float *out,
                      const float *inp1, const float *inp2, std::size_t n) {
  constexpr std::size_t block_size = 256;
  (void)block_size;
  switch (kernel_num) {
  case 1:
    residual_forward1(q, out, inp1, inp2, n);
    break;
  case 2:
    residual_forward2(q, out, inp1, inp2, n);
    break;
  default:
    std::fprintf(stderr, "Invalid kernel number\n");
    std::exit(1);
  }
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 6) {
    std::fprintf(stderr, "usage: %s <input> <output> B T C\n", argv[0]);
    return 2;
  }
  const std::size_t b = static_cast<std::size_t>(std::atoi(argv[3]));
  const std::size_t t = static_cast<std::size_t>(std::atoi(argv[4]));
  const std::size_t c = static_cast<std::size_t>(std::atoi(argv[5]));
  const std::size_t n = b * t * c;

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

  sycl::queue q;
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
    q.memcpy(out, d_out, sizeof(float) * n).wait();
    std::fwrite(out, sizeof(float), n, fout);
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
