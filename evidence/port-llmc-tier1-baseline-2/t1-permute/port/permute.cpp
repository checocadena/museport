#include <sycl/sycl.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstddef>

// Variant 1: permute (D1,D2,D3,D4) -> (D4,D3,D1,D2)
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
    const std::size_t N =
        static_cast<std::size_t>(D1) * static_cast<std::size_t>(D2) *
        static_cast<std::size_t>(D3) * static_cast<std::size_t>(D4);

    float *matrix = static_cast<float *>(std::malloc(sizeof(float) * (N ? N : 1)));
    float *out = static_cast<float *>(std::malloc(sizeof(float) * (N ? N : 1)));
    if (!matrix || !out) {
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
    if (N > 0 &&
        std::fread(matrix, sizeof(float), N, fin) != N) {
        std::fprintf(stderr, "input too short at matrix\n");
        std::fclose(fin);
        std::free(matrix);
        std::free(out);
        return 2;
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

    float *d_matrix = sycl::malloc_device<float>(N ? N : 1, q);
    float *d_out = sycl::malloc_device<float>(N ? N : 1, q);
    if (!d_matrix || !d_out) {
        std::fprintf(stderr, "device malloc failed\n");
        std::fclose(fout);
        sycl::free(d_matrix, q);
        sycl::free(d_out, q);
        std::free(matrix);
        std::free(out);
        return 2;
    }

    if (N > 0) {
        q.memcpy(d_matrix, matrix, sizeof(float) * N).wait();
    }

    const int variants[] = {1};
    for (int vi = 0; vi < 1; ++vi) {
        const int k = variants[vi];
        if (k == 1) {
            if (N > 0) {
                q.fill(d_out, 0.0f, N).wait();
                q.submit([&](sycl::handler &h) {
                    h.parallel_for<PermuteKernel1>(
                        sycl::range<1>(N), [=](sycl::id<1> id) {
                            std::size_t idx = id[0];
                            int i1 = static_cast<int>((idx / (static_cast<std::size_t>(D2) * static_cast<std::size_t>(D3) * static_cast<std::size_t>(D4))) % static_cast<std::size_t>(D1));
                            int i2 = static_cast<int>((idx / (static_cast<std::size_t>(D3) * static_cast<std::size_t>(D4))) % static_cast<std::size_t>(D2));
                            int i3 = static_cast<int>((idx / static_cast<std::size_t>(D4)) % static_cast<std::size_t>(D3));
                            int i4 = static_cast<int>(idx % static_cast<std::size_t>(D4));
                            std::size_t permuted_idx =
                                static_cast<std::size_t>(i4) *
                                    (static_cast<std::size_t>(D3) *
                                     static_cast<std::size_t>(D1) *
                                     static_cast<std::size_t>(D2)) +
                                static_cast<std::size_t>(i3) *
                                    (static_cast<std::size_t>(D1) *
                                     static_cast<std::size_t>(D2)) +
                                static_cast<std::size_t>(i1) *
                                    static_cast<std::size_t>(D2) +
                                static_cast<std::size_t>(i2);
                            d_out[permuted_idx] = d_matrix[idx];
                        });
                }).wait();
                q.memcpy(out, d_out, sizeof(float) * N).wait();
            }
            std::fwrite(out, sizeof(float), N, fout);
        }
    }

    std::fclose(fout);
    sycl::free(d_matrix, q);
    sycl::free(d_out, q);
    std::free(matrix);
    std::free(out);
    return 0;
}
