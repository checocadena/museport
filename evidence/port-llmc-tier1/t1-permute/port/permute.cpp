// Idiomatic SYCL 2020 port of permute_harness.cu (fp32).
// Permutes 4D tensor (D1,D2,D3,D4) -> (D4,D3,D1,D2).
#include <sycl/sycl.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstddef>

// Variant 1 kernel name: distinct SYCL kernel per variant k in [1].
class PermuteV1Kernel;

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

  float *matrix = nullptr;
  float *out = nullptr;
  if (n > 0) {
    matrix = static_cast<float *>(std::malloc(sizeof(float) * n));
    out = static_cast<float *>(std::malloc(sizeof(float) * n));
    if (matrix == nullptr || out == nullptr) {
      std::fprintf(stderr, "host malloc failed\n");
      std::free(matrix);
      std::free(out);
      return 2;
    }
  }

  FILE *fin = std::fopen(argv[1], "rb");
  if (fin == nullptr) {
    std::perror(argv[1]);
    std::free(matrix);
    std::free(out);
    return 2;
  }
  if (n > 0) {
    if (std::fread(matrix, sizeof(float), n, fin) != n) {
      std::fprintf(stderr, "input too short at matrix\n");
      std::fclose(fin);
      std::free(matrix);
      std::free(out);
      return 2;
    }
  }
  std::fclose(fin);

  FILE *fout = std::fopen(argv[2], "wb");
  if (fout == nullptr) {
    std::perror(argv[2]);
    std::free(matrix);
    std::free(out);
    return 2;
  }

  try {
    sycl::queue q{sycl::default_selector_v};

    float *d_matrix =
        (n == 0) ? nullptr : sycl::malloc_device<float>(n, q);
    float *d_out = (n == 0) ? nullptr : sycl::malloc_device<float>(n, q);
    if (n > 0 && (d_matrix == nullptr || d_out == nullptr)) {
      std::fprintf(stderr, "device malloc failed\n");
      if (d_matrix != nullptr) sycl::free(d_matrix, q);
      if (d_out != nullptr) sycl::free(d_out, q);
      std::fclose(fout);
      std::free(matrix);
      std::free(out);
      return 1;
    }

    const int variants[] = {1};
    for (int vi = 0; vi < 1; ++vi) {
      const int k = variants[vi];
      if (n == 0) {
        continue;
      }
      if (k == 1) {
        q.memcpy(d_matrix, matrix, sizeof(float) * n).wait();
        q.memset(d_out, 0, sizeof(float) * n).wait();

        const std::size_t sD1 = static_cast<std::size_t>(D1);
        const std::size_t sD2 = static_cast<std::size_t>(D2);
        const std::size_t sD3 = static_cast<std::size_t>(D3);
        const std::size_t sD4 = static_cast<std::size_t>(D4);
        const std::size_t stride_i1 = sD2 * sD3 * sD4;
        const std::size_t stride_i2 = sD3 * sD4;
        const std::size_t block_v1 = sD3 * sD1 * sD2;
        const std::size_t block_v1_inner = sD1 * sD2;

        q.parallel_for<PermuteV1Kernel>(
            sycl::range<1>(n), [=](sycl::id<1> id) {
              const std::size_t idx = id[0];
              const std::size_t i1 = (idx / stride_i1) % sD1;
              const std::size_t i2 = (idx / stride_i2) % sD2;
              const std::size_t i3 = (idx / sD4) % sD3;
              const std::size_t i4 = idx % sD4;
              const std::size_t permuted_idx =
                  i4 * block_v1 + i3 * block_v1_inner + i1 * sD2 + i2;
              d_out[permuted_idx] = d_matrix[idx];
            }).wait();

        q.memcpy(out, d_out, sizeof(float) * n).wait();
        std::fwrite(out, sizeof(float), n, fout);
      }
    }

    if (d_matrix != nullptr) sycl::free(d_matrix, q);
    if (d_out != nullptr) sycl::free(d_out, q);
  } catch (const sycl::exception &e) {
    std::fprintf(stderr, "SYCL error: %s\n", e.what());
    std::fclose(fout);
    std::free(matrix);
    std::free(out);
    return 1;
  }

  std::fclose(fout);
  std::free(matrix);
  std::free(out);
  return 0;
}
