#include <sycl/sycl.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstddef>

// Kernel variant k=1: permute (D1,D2,D3,D4) -> (D4,D3,D1,D2).
// Each work-item handles one linear element.
class permute_kernel_1;

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

  float *matrix = static_cast<float *>(std::malloc(sizeof(float) * (n ? n : 1)));
  float *out = static_cast<float *>(std::malloc(sizeof(float) * (n ? n : 1)));
  if (matrix == nullptr || out == nullptr) {
    std::fprintf(stderr, "host malloc failed\n");
    std::free(matrix);
    std::free(out);
    return 2;
  }

  FILE *fin = std::fopen(argv[1], "rb");
  if (!fin) {
    std::perror(argv[1]);
    std::free(matrix);
    std::free(out);
    return 2;
  }
  if (n != 0) {
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
  if (!fout) {
    std::perror(argv[2]);
    std::free(matrix);
    std::free(out);
    return 2;
  }

  sycl::queue q{sycl::default_selector_v};

  float *d_matrix = n ? sycl::malloc_device<float>(n, q) : nullptr;
  float *d_out = n ? sycl::malloc_device<float>(n, q) : nullptr;
  if (n != 0 && (d_matrix == nullptr || d_out == nullptr)) {
    std::fprintf(stderr, "device malloc failed\n");
    sycl::free(d_matrix, q);
    sycl::free(d_out, q);
    std::fclose(fout);
    std::free(matrix);
    std::free(out);
    return 2;
  }

  const int variants[] = {1};
  for (int vi = 0; vi < 1; ++vi) {
    const int k = variants[vi];
    if (k == 1 && n != 0) {
      q.memcpy(d_matrix, matrix, sizeof(float) * n).wait();
      q.memset(d_out, 0, sizeof(float) * n).wait();
      const std::size_t sD1 = static_cast<std::size_t>(D1);
      const std::size_t sD2 = static_cast<std::size_t>(D2);
      const std::size_t sD3 = static_cast<std::size_t>(D3);
      const std::size_t sD4 = static_cast<std::size_t>(D4);
      q.parallel_for<permute_kernel_1>(
          sycl::range<1>{n}, [=](sycl::item<1> it) {
            std::size_t idx = it.get_id(0);
            std::size_t i1 = (idx / (sD2 * sD3 * sD4)) % sD1;
            std::size_t i2 = (idx / (sD3 * sD4)) % sD2;
            std::size_t i3 = (idx / sD4) % sD3;
            std::size_t i4 = idx % sD4;
            std::size_t permuted_idx =
                i4 * (sD3 * sD1 * sD2) + i3 * (sD1 * sD2) + i1 * sD2 + i2;
            d_out[permuted_idx] = d_matrix[idx];
          }).wait();
      q.memcpy(out, d_out, sizeof(float) * n).wait();
    }
    if (n != 0) {
      std::fwrite(out, sizeof(float), n, fout);
    }
  }

  sycl::free(d_matrix, q);
  sycl::free(d_out, q);
  std::fclose(fout);
  std::free(matrix);
  std::free(out);
  return 0;
}
