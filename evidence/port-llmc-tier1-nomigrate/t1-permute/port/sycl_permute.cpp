#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

// SYCL kernel to permute a 4D matrix from (dim1, dim2, dim3, dim4)
// to (dim4, dim3, dim1, dim2).
class permute_kernel_1;

int main(int argc, char **argv) {
    if (argc != 7) {
        std::fprintf(stderr, "usage: %s <input> <output> D1 D2 D3 D4\n", argv[0]);
        return 2;
    }
    const long D1 = std::atol(argv[3]);
    const long D2 = std::atol(argv[4]);
    const long D3 = std::atol(argv[5]);
    const long D4 = std::atol(argv[6]);
    const size_t N = static_cast<size_t>(D1) * static_cast<size_t>(D2) *
                     static_cast<size_t>(D3) * static_cast<size_t>(D4);

    FILE *fin = std::fopen(argv[1], "rb");
    if (!fin) { std::perror(argv[1]); return 2; }
    float *matrix = static_cast<float *>(std::malloc(sizeof(float) * N));
    if (!matrix) { std::fprintf(stderr, "host malloc failed\n"); std::fclose(fin); return 2; }
    if (N > 0 && std::fread(matrix, sizeof(float), N, fin) != N) {
        std::fprintf(stderr, "input too short at matrix\n");
        std::fclose(fin);
        std::free(matrix);
        return 2;
    }
    std::fclose(fin);

    FILE *fout = std::fopen(argv[2], "wb");
    if (!fout) { std::perror(argv[2]); std::free(matrix); return 2; }

    float *out = static_cast<float *>(std::malloc(sizeof(float) * N));
    if (!out) { std::fprintf(stderr, "host malloc failed\n"); std::fclose(fout); std::free(matrix); return 2; }

    sycl::queue q{sycl::default_selector_v};

    float *d_matrix = sycl::malloc_device<float>(N > 0 ? N : 1, q);
    float *d_out = sycl::malloc_device<float>(N > 0 ? N : 1, q);
    if (!d_matrix || !d_out) {
        std::fprintf(stderr, "device malloc failed\n");
        std::fclose(fout);
        std::free(matrix);
        std::free(out);
        return 2;
    }

    if (N > 0) {
        q.memcpy(d_matrix, matrix, sizeof(float) * N).wait();
    }

    // Variant 1: direct port of permute_kernel, one work-item per element.
    const int dim1 = static_cast<int>(D1);
    const int dim2 = static_cast<int>(D2);
    const int dim3 = static_cast<int>(D3);
    const int dim4 = static_cast<int>(D4);
    const size_t n = N;

    if (n > 0) {
        q.memset(d_out, 0, sizeof(float) * n).wait();
        q.submit([&](sycl::handler &h) {
            h.parallel_for<permute_kernel_1>(sycl::range<1>(n), [=](sycl::id<1> wid) {
                const size_t idx = wid[0];
                const size_t s23 = static_cast<size_t>(dim2) * static_cast<size_t>(dim3) *
                                   static_cast<size_t>(dim4);
                const size_t s3 = static_cast<size_t>(dim3) * static_cast<size_t>(dim4);
                const size_t d4 = static_cast<size_t>(dim4);
                const int i1 = static_cast<int>((idx / s23) % static_cast<size_t>(dim1));
                const int i2 = static_cast<int>((idx / s3) % static_cast<size_t>(dim2));
                const int i3 = static_cast<int>((idx / d4) % static_cast<size_t>(dim3));
                const int i4 = static_cast<int>(idx % d4);
                const size_t permuted_idx =
                    static_cast<size_t>(i4) * (static_cast<size_t>(dim3) * static_cast<size_t>(dim1) * static_cast<size_t>(dim2)) +
                    static_cast<size_t>(i3) * (static_cast<size_t>(dim1) * static_cast<size_t>(dim2)) +
                    static_cast<size_t>(i1) * static_cast<size_t>(dim2) +
                    static_cast<size_t>(i2);
                d_out[permuted_idx] = d_matrix[idx];
            });
        }).wait();
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
