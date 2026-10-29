#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

// Kernel name tag for variant 1 so it is its own SYCL kernel.
class PermuteKernel1;

static void permute_variant1(sycl::queue &q, const float *d_in, float *d_out,
                             int dim1, int dim2, int dim3, int dim4,
                             std::size_t n) {
  q.fill(d_out, 0.0f, n).wait();
  q.parallel_for<PermuteKernel1>(sycl::range<1>(n),
                                 [=](sycl::id<1> id) {
                                   std::size_t idx = id[0];
                                   int d2d3d4 = dim2 * dim3 * dim4;
                                   int d3d4 = dim3 * dim4;
                                   int i1 = static_cast<int>((idx / (std::size_t)d2d3d4) % (std::size_t)dim1);
                                   int i2 = static_cast<int>((idx / (std::size_t)d3d4) % (std::size_t)dim2);
                                   int i3 = static_cast<int>((idx / (std::size_t)dim4) % (std::size_t)dim3);
                                   int i4 = static_cast<int>(idx % (std::size_t)dim4);
                                   std::size_t permuted_idx =
                                       (std::size_t)i4 * (std::size_t)(dim3 * dim1 * dim2) +
                                       (std::size_t)i3 * (std::size_t)(dim1 * dim2) +
                                       (std::size_t)i1 * (std::size_t)dim2 +
                                       (std::size_t)i2;
                                   d_out[permuted_idx] = d_in[idx];
                                 })
      .wait();
}

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

  FILE *fin = std::fopen(argv[1], "rb");
  if (!fin) {
    std::perror(argv[1]);
    return 2;
  }
  float *matrix = static_cast<float *>(std::malloc(sizeof(float) * n));
  if (n > 0 &&
      std::fread(matrix, sizeof(float), n, fin) != n) {
    std::fprintf(stderr, "input too short at matrix\n");
    std::fclose(fin);
    return 2;
  }
  std::fclose(fin);

  FILE *fout = std::fopen(argv[2], "wb");
  if (!fout) {
    std::perror(argv[2]);
    return 2;
  }

  sycl::queue q{sycl::default_selector_v};
  float *d_matrix = sycl::malloc_device<float>(n, q);
  float *d_out = sycl::malloc_device<float>(n, q);
  float *out = static_cast<float *>(std::malloc(sizeof(float) * n));

  q.memcpy(d_matrix, matrix, sizeof(float) * n).wait();

  const int variants[] = {1};
  for (int vi = 0; vi < 1; ++vi) {
    const int k = variants[vi];
    if (k == 1) {
      permute_variant1(q, d_matrix, d_out, D1, D2, D3, D4, n);
    }
    q.memcpy(out, d_out, sizeof(float) * n).wait();
    std::fwrite(out, sizeof(float), n, fout);
  }

  std::fclose(fout);
  sycl::free(d_matrix, q);
  sycl::free(d_out, q);
  std::free(matrix);
  std::free(out);
  return 0;
}
