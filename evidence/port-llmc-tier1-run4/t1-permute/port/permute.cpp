#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstddef>

struct PermuteKernelV1;

int main(int argc, char **argv) {
  if (argc != 7) {
    std::fprintf(stderr, "usage: %s <input> <output> D1 D2 D3 D4\n", argv[0]);
    return 2;
  }
  const int D1 = std::atoi(argv[3]);
  const int D2 = std::atoi(argv[4]);
  const int D3 = std::atoi(argv[5]);
  const int D4 = std::atoi(argv[6]);
  const std::size_t N =
      static_cast<std::size_t>(D1) * static_cast<std::size_t>(D2) *
      static_cast<std::size_t>(D3) * static_cast<std::size_t>(D4);

  FILE *fin = std::fopen(argv[1], "rb");
  if (!fin) {
    std::perror(argv[1]);
    return 2;
  }
  float *matrix = static_cast<float *>(std::malloc(sizeof(float) * N));
  if (!matrix) {
    std::fclose(fin);
    return 2;
  }
  if (std::fread(matrix, sizeof(float), N, fin) != N) {
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
  float *out = static_cast<float *>(std::malloc(sizeof(float) * N));
  if (!out) {
    std::fclose(fout);
    return 2;
  }

  try {
    sycl::queue q{sycl::default_selector_v};

    float *d_matrix = sycl::malloc_device<float>(N, q);
    float *d_out = sycl::malloc_device<float>(N, q);
    if (!d_matrix || !d_out) {
      std::fprintf(stderr, "device allocation failed\n");
      return 2;
    }

    q.memcpy(d_matrix, matrix, sizeof(float) * N).wait();

    constexpr std::size_t WG = 256;
    const std::size_t n_groups = (N + WG - 1) / WG;
    const sycl::nd_range<1> launch_range(sycl::range<1>(n_groups * WG),
                                         sycl::range<1>(WG));

    const int variants[] = {1};
    for (int vi = 0; vi < 1; ++vi) {
      const int k = variants[vi];
      q.memset(d_out, 0, sizeof(float) * N).wait();
      if (k == 1) {
        q.parallel_for<PermuteKernelV1>(
            launch_range, [=](sycl::nd_item<1> it) {
              const std::size_t idx = it.get_global_id(0);
              if (idx < N) {
                const int s234 = D2 * D3 * D4;
                const int s34 = D3 * D4;
                const int i1 =
                    static_cast<int>((idx / static_cast<std::size_t>(s234)) %
                                     static_cast<std::size_t>(D1));
                const int i2 =
                    static_cast<int>((idx / static_cast<std::size_t>(s34)) %
                                     static_cast<std::size_t>(D2));
                const int i3 =
                    static_cast<int>((idx / static_cast<std::size_t>(D4)) %
                                     static_cast<std::size_t>(D3));
                const int i4 =
                    static_cast<int>(idx % static_cast<std::size_t>(D4));
                const std::size_t permuted_idx =
                    static_cast<std::size_t>(i4) *
                        (static_cast<std::size_t>(D3) *
                         static_cast<std::size_t>(D1) *
                         static_cast<std::size_t>(D2)) +
                    static_cast<std::size_t>(i3) *
                        (static_cast<std::size_t>(D1) *
                         static_cast<std::size_t>(D2)) +
                    static_cast<std::size_t>(i1) * static_cast<std::size_t>(D2) +
                    static_cast<std::size_t>(i2);
                d_out[permuted_idx] = d_matrix[idx];
              }
            });
        q.wait();
      }
      q.memcpy(out, d_out, sizeof(float) * N).wait();
      std::fwrite(out, sizeof(float), N, fout);
    }

    sycl::free(d_matrix, q);
    sycl::free(d_out, q);
  } catch (sycl::exception const &e) {
    std::fprintf(stderr, "SYCL exception: %s\n", e.what());
    std::fclose(fout);
    return 1;
  }

  std::fclose(fout);
  std::free(matrix);
  std::free(out);
  return 0;
}
