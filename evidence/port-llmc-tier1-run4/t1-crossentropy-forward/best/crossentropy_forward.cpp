#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

namespace {

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

class CrossEntropyForwardKernel1;

void crossentropy_forward1(sycl::queue &q, float *losses, const float *probs,
                           const int *targets, int B, int T, int V) {
    const int N = B * T;
    if (N <= 0) {
        return;
    }
    constexpr int block_size = 256;
    const int grid_size = ceil_div(N, block_size);
    q.submit([&](sycl::handler &h) {
        h.parallel_for<CrossEntropyForwardKernel1>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(grid_size * block_size)),
                              sycl::range<1>(static_cast<size_t>(block_size))),
            [=](sycl::nd_item<1> item) {
                const int i = static_cast<int>(item.get_global_id(0));
                if (i < N) {
                    const int b = i / T;
                    const int t = i % T;
                    const float *probs_bt = probs + static_cast<size_t>(b) * T * V + static_cast<size_t>(t) * V;
                    const int ix = targets[static_cast<size_t>(b) * T + t];
                    losses[static_cast<size_t>(b) * T + t] = -sycl::log(probs_bt[ix]);
                }
            });
    });
}

void crossentropy_forward(int kernel_num, sycl::queue &q, float *losses,
                          const float *probs, const int *targets, int B, int T,
                          int V) {
    switch (kernel_num) {
        case 1:
            crossentropy_forward1(q, losses, probs, targets, B, T, V);
            break;
        default:
            std::printf("Invalid kernel number\n");
            std::exit(1);
    }
}

}  // namespace

int main(int argc, char **argv) {
    std::srand(0);
    if (argc != 6) {
        std::fprintf(stderr, "usage: %s <input> <output> B T V\n", argv[0]);
        return 2;
    }
    const int B = std::atoi(argv[3]);
    const int T = std::atoi(argv[4]);
    const int V = std::atoi(argv[5]);

    FILE *fin = std::fopen(argv[1], "rb");
    if (!fin) {
        std::perror(argv[1]);
        return 2;
    }
    float *probs = static_cast<float *>(std::malloc(sizeof(float) * static_cast<size_t>(B) * T * V));
    if (std::fread(probs, sizeof(float), static_cast<size_t>(B) * T * V, fin) != static_cast<size_t>(B) * T * V) {
        std::fprintf(stderr, "input too short at probs\n");
        return 2;
    }
    int *targets = static_cast<int *>(std::malloc(sizeof(int) * static_cast<size_t>(B) * T));
    if (std::fread(targets, sizeof(int), static_cast<size_t>(B) * T, fin) != static_cast<size_t>(B) * T) {
        std::fprintf(stderr, "input too short at targets\n");
        return 2;
    }
    std::fclose(fin);
    FILE *fout = std::fopen(argv[2], "wb");
    if (!fout) {
        std::perror(argv[2]);
        return 2;
    }

    sycl::queue q{sycl::default_selector_v};
    float *d_probs = sycl::malloc_device<float>(static_cast<size_t>(B) * T * V, q);
    q.memcpy(d_probs, probs, sizeof(float) * static_cast<size_t>(B) * T * V).wait();
    int *d_targets = sycl::malloc_device<int>(static_cast<size_t>(B) * T, q);
    q.memcpy(d_targets, targets, sizeof(int) * static_cast<size_t>(B) * T).wait();
    float *out = static_cast<float *>(std::malloc(sizeof(float) * static_cast<size_t>(B) * T));
    float *d_out = sycl::malloc_device<float>(static_cast<size_t>(B) * T, q);

    const int variants[] = {1};
    for (int vi = 0; vi < 1; ++vi) {
        const int k = variants[vi];
        q.memset(d_out, 0, sizeof(float) * static_cast<size_t>(B) * T).wait();
        crossentropy_forward(k, q, d_out, d_probs, d_targets, B, T, V);
        q.wait();
        q.memcpy(out, d_out, sizeof(float) * static_cast<size_t>(B) * T).wait();
        std::fwrite(out, sizeof(float), static_cast<size_t>(B) * T, fout);
    }
    std::fclose(fout);

    sycl::free(d_probs, q);
    sycl::free(d_targets, q);
    sycl::free(d_out, q);
    std::free(probs);
    std::free(targets);
    std::free(out);
    return 0;
}
