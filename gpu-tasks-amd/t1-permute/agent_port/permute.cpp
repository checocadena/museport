#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstddef>

// Variant 1: permute 4D matrix (dim1, dim2, dim3, dim4) -> (dim4, dim3, dim1, dim2).
class PermuteKernel1;

int main(int argc, char **argv) {
  if (argc != 7) {
    std::fprintf(stderr, "usage: %s <input> <output> D1 D2 D3 D4\n", argv[0]);
    return 2;
  }
  const long D1 = std::atol(argv[3]);
  const long D2 = std::atol(argv[4]);
  const long D3 = std::atol(argv[5]);
  const long D4 = std::atol(argv[6]);
  const std::size_t n =
      static_cast<std::size_t>(D1) * static_cast<std::size_t>(D2) *
      static_cast<std::size_t>(D3) * static_cast<std::size_t>(D4);

  std::FILE *fin = std::fopen(argv[1], "rb");
  if (!fin) {
    std::perror(argv[1]);
    return 2;
  }
  float *matrix = static_cast<float *>(std::malloc(sizeof(float) * n));
  if (!matrix) {
    std::fclose(fin);
    return 2;
  }
  if (std::fread(matrix, sizeof(float), n, fin) != n) {
    std::fprintf(stderr, "input too short at matrix\n");
    std::fclose(fin);
    std::free(matrix);
    return 2;
  }
  std::fclose(fin);

  std::FILE *fout = std::fopen(argv[2], "wb");
  if (!fout) {
    std::perror(argv[2]);
    std::free(matrix);
    return 2;
  }
  float *out = static_cast<float *>(std::malloc(sizeof(float) * n));
  if (!out) {
    std::fclose(fout);
    std::free(matrix);
    return 2;
  }

  try {
    sycl::queue q{sycl::default_selector_v};

    float *d_matrix = sycl::malloc_device<float>(n, q);
    float *d_out = sycl::malloc_device<float>(n, q);
    if (!d_matrix || !d_out) {
      std::fprintf(stderr, "device allocation failed\n");
      sycl::free(d_matrix, q);
      sycl::free(d_out, q);
      std::fclose(fout);
      std::free(matrix);
      std::free(out);
      return 2;
    }

    q.memcpy(d_matrix, matrix, sizeof(float) * n).wait();

    const int variants[] = {1};
    for (int vi = 0; vi < 1; ++vi) {
      const int k = variants[vi];
      if (k == 1) {
        q.memset(d_out, 0, sizeof(float) * n).wait();
        const int dim1 = static_cast<int>(D1);
        const int dim2 = static_cast<int>(D2);
        const int dim3 = static_cast<int>(D3);
        const int dim4 = static_cast<int>(D4);
        q.parallel_for<PermuteKernel1>(sycl::range<1>(n),
                                       [=](sycl::id<1> id) {
                                         const std::size_t idx = id[0];
                                         const long s23 = (long)dim2 * dim3 * dim4;
                                         const long s3 = (long)dim3 * dim4;
                                         const int i1 = (int)((idx / (std::size_t)s23) % (std::size_t)dim1);
                                         const int i2 = (int)((idx / (std::size_t)s3) % (std::size_t)dim2);
                                         const int i3 = (int)((idx / (std::size_t)dim4) % (std::size_t)dim3);
                                         const int i4 = (int)(idx % (std::size_t)dim4);
                                         const std::size_t permuted_idx =
                                             (std::size_t)i4 * (std::size_t)(dim3 * dim1 * dim2) +
                                             (std::size_t)i3 * (std::size_t)(dim1 * dim2) +
                                             (std::size_t)i1 * (std::size_t)dim2 +
                                             (std::size_t)i2;
                                         d_out[permuted_idx] = d_matrix[idx];
                                       })
            .wait();
      }
    }

    q.memcpy(out, d_out, sizeof(float) * n).wait();
    sycl::free(d_matrix, q);
    sycl::free(d_out, q);
  } catch (const sycl::exception &e) {
    std::fprintf(stderr, "SYCL error: %s\n", e.what());
    std::fclose(fout);
    std::free(matrix);
    std::free(out);
    return 1;
  }

  std::fwrite(out, sizeof(float), n, fout);
  std::fclose(fout);
  std::free(matrix);
  std::free(out);
  return 0;
}
