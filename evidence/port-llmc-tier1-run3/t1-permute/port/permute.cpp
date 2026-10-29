#include <sycl/sycl.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <vector>

// Variant 1: permute (D1,D2,D3,D4) -> (D4,D3,D1,D2), fp32.
class PermuteKernel1;

int main(int argc, char **argv) {
  if (argc != 7) {
    std::fprintf(stderr, "usage: %s <input> <output> D1 D2 D3 D4\n", argv[0]);
    return 2;
  }
  const int D1 = std::atoi(argv[3]);
  const int D2 = std::atoi(argv[4]);
  const int D3 = std::atoi(argv[5]);
  const int D4 = std::atoi(argv[6]);
  const std::size_t n =
      static_cast<std::size_t>(D1) * static_cast<std::size_t>(D2) *
      static_cast<std::size_t>(D3) * static_cast<std::size_t>(D4);

  std::vector<float> host_in(n);
  FILE *fin = std::fopen(argv[1], "rb");
  if (!fin) {
    std::perror(argv[1]);
    return 2;
  }
  std::size_t got =
      n == 0 ? 0 : std::fread(host_in.data(), sizeof(float), n, fin);
  std::fclose(fin);
  if (got != n) {
    std::fprintf(stderr, "input too short at matrix\n");
    return 2;
  }

  FILE *fout = std::fopen(argv[2], "wb");
  if (!fout) {
    std::perror(argv[2]);
    return 2;
  }

  sycl::queue q{sycl::default_selector_v};

  float *d_in = sycl::malloc_device<float>(n == 0 ? 1 : n, q);
  float *d_out = sycl::malloc_device<float>(n == 0 ? 1 : n, q);
  if (!d_in || !d_out) {
    std::fprintf(stderr, "device allocation failed\n");
    std::fclose(fout);
    return 1;
  }

  if (n > 0) {
    q.memcpy(d_in, host_in.data(), n * sizeof(float)).wait();
  }

  std::vector<float> host_out(n);
  const int variants[] = {1};
  for (int vi = 0; vi < 1; ++vi) {
    const int k = variants[vi];
    if (k == 1) {
      if (n > 0) {
        const std::size_t s23 = static_cast<std::size_t>(D2) *
                                static_cast<std::size_t>(D3) *
                                static_cast<std::size_t>(D4);
        const std::size_t s3 = static_cast<std::size_t>(D3) *
                               static_cast<std::size_t>(D4);
        const std::size_t d4 = static_cast<std::size_t>(D4);
        const std::size_t uD1 = static_cast<std::size_t>(D1);
        const std::size_t uD2 = static_cast<std::size_t>(D2);
        const std::size_t uD3 = static_cast<std::size_t>(D3);
        q.parallel_for<PermuteKernel1>(sycl::range<1>(n),
                                       [=](sycl::id<1> id) {
                                         std::size_t idx = id[0];
                                         std::size_t i1 = (idx / s23) % uD1;
                                         std::size_t i2 = (idx / s3) % uD2;
                                         std::size_t i3 = (idx / d4) % uD3;
                                         std::size_t i4 = idx % d4;
                                         std::size_t permuted =
                                             i4 * (uD3 * uD1 * uD2) +
                                             i3 * (uD1 * uD2) + i1 * uD2 + i2;
                                         d_out[permuted] = d_in[idx];
                                       })
            .wait();
        q.memcpy(host_out.data(), d_out, n * sizeof(float)).wait();
      }
      std::fwrite(host_out.data(), sizeof(float), n, fout);
    }
  }

  sycl::free(d_in, q);
  sycl::free(d_out, q);
  std::fclose(fout);
  return 0;
}
