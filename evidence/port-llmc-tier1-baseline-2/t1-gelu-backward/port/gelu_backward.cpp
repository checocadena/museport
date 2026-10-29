#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>

#include <sycl/sycl.hpp>

using floatX = float;

// sqrt(2/pi), the tanh-approximation GELU scaling factor.
inline constexpr float kGeluScalingFactor = 0.7978845608028654f;
inline constexpr int kPackSize = 4;
inline constexpr int kBlockSize = 256;

template <typename T>
T ceil_div(T dividend, T divisor) {
    return (dividend + divisor - 1) / divisor;
}

inline float gelu_local_grad(float x) {
    const float cube = 0.044715f * x * x * x;
    const float tanh_arg = kGeluScalingFactor * (x + cube);
    const float tanh_out = sycl::tanh(tanh_arg);
    const float coshf_out = sycl::cosh(tanh_arg);
    const float sech_out = 1.0f / (coshf_out * coshf_out);
    return 0.5f * (1.0f + tanh_out) +
           x * 0.5f * sech_out * kGeluScalingFactor *
               (1.0f + 3.0f * 0.044715f * x * x);
}

class GeluBackward1Kernel;
class GeluBackward2Kernel;

void gelu_backward1(sycl::queue &q, floatX *dinp, const floatX *inp,
                    const floatX *dout, std::size_t N) {
    if (N == 0) {
        return;
    }
    const std::size_t local = static_cast<std::size_t>(kBlockSize);
    const std::size_t global = ceil_div(N, local) * local;
    q.submit([&](sycl::handler &h) {
        h.parallel_for<GeluBackward1Kernel>(
            sycl::nd_range<1>{sycl::range<1>{global}, sycl::range<1>{local}},
            [=](sycl::nd_item<1> it) {
                const std::size_t i = it.get_global_id(0);
                if (i < N) {
                    const float x = static_cast<float>(inp[i]);
                    const float local_grad = gelu_local_grad(x);
                    dinp[i] =
                        static_cast<floatX>(local_grad * static_cast<float>(dout[i]));
                }
            });
    });
}

void gelu_backward2(sycl::queue &q, floatX *dinp, const floatX *inp,
                    const floatX *dout, std::size_t N) {
    if (N == 0) {
        return;
    }
    const std::size_t num_packs =
        ceil_div(N, static_cast<std::size_t>(kPackSize));
    const std::size_t local = static_cast<std::size_t>(kBlockSize);
    const std::size_t global = ceil_div(num_packs, local) * local;
    q.submit([&](sycl::handler &h) {
        h.parallel_for<GeluBackward2Kernel>(
            sycl::nd_range<1>{sycl::range<1>{global}, sycl::range<1>{local}},
            [=](sycl::nd_item<1> it) {
                const std::size_t pack = it.get_global_id(0);
                if (pack >= num_packs) {
                    return;
                }
                // 128-bit wide work unit: 4 contiguous fp32 elements per
                // work-item. Coalesced scalar accesses let the compiler emit
                // 128-bit load/store transactions, mirroring the CUDA
                // Packed128 (LDG.128/STS.128) intent. The per-lane bound
                // check keeps tail packs (N % 4 != 0) in bounds.
                const std::size_t base =
                    pack * static_cast<std::size_t>(kPackSize);
                float xs[kPackSize];
                float dys[kPackSize];
                for (int k = 0; k < kPackSize; ++k) {
                    const std::size_t idx = base + static_cast<std::size_t>(k);
                    xs[k] = (idx < N) ? static_cast<float>(inp[idx]) : 0.0f;
                    dys[k] = (idx < N) ? static_cast<float>(dout[idx]) : 0.0f;
                }
                for (int k = 0; k < kPackSize; ++k) {
                    const std::size_t idx = base + static_cast<std::size_t>(k);
                    if (idx < N) {
                        const float local_grad = gelu_local_grad(xs[k]);
                        dinp[idx] =
                            static_cast<floatX>(local_grad * dys[k]);
                    }
                }
            });
    });
}

void gelu_backward(sycl::queue &q, int kernel_num, floatX *dinp,
                   const floatX *inp, const floatX *dout, std::size_t N) {
    switch (kernel_num) {
        case 1:
            gelu_backward1(q, dinp, inp, dout, N);
            break;
        case 2:
            gelu_backward2(q, dinp, inp, dout, N);
            break;
        default:
            std::printf("Invalid kernel number\n");
            std::exit(1);
    }
}

int main(int argc, char **argv) {
    std::srand(0);
    if (argc != 6) {
        std::fprintf(stderr, "usage: %s <input> <output> B T C\n", argv[0]);
        return 2;
    }
    const int B = std::atoi(argv[3]);
    const int T = std::atoi(argv[4]);
    const int C = std::atoi(argv[5]);
    const std::size_t N = static_cast<std::size_t>(B) *
                          static_cast<std::size_t>(T) *
                          static_cast<std::size_t>(C);

    FILE *fin = std::fopen(argv[1], "rb");
    if (!fin) {
        std::perror(argv[1]);
        return 2;
    }
    float *inp = static_cast<float *>(std::malloc(sizeof(float) * N));
    float *dout_host = static_cast<float *>(std::malloc(sizeof(float) * N));
    if (N > 0) {
        if (std::fread(inp, sizeof(float), N, fin) != N) {
            std::fprintf(stderr, "input too short at inp\n");
            return 2;
        }
        if (std::fread(dout_host, sizeof(float), N, fin) != N) {
            std::fprintf(stderr, "input too short at dout\n");
            return 2;
        }
    }
    std::fclose(fin);

    FILE *fout = std::fopen(argv[2], "wb");
    if (!fout) {
        std::perror(argv[2]);
        return 2;
    }

    sycl::queue q{sycl::default_selector_v};

    float *d_inp = sycl::malloc_device<float>(N, q);
    float *d_dout = sycl::malloc_device<float>(N, q);
    float *d_dinp = sycl::malloc_device<float>(N, q);
    float *dinp = static_cast<float *>(std::malloc(sizeof(float) * N));

    if (N > 0) {
        q.memcpy(d_inp, inp, sizeof(float) * N).wait();
        q.memcpy(d_dout, dout_host, sizeof(float) * N).wait();
    }

    const int variants[] = {1, 2};
    for (int vi = 0; vi < 2; ++vi) {
        const int k = variants[vi];
        if (N > 0) {
            q.fill(d_dinp, 0.0f, N).wait();
        }
        gelu_backward(q, k, d_dinp, d_inp, d_dout, N);
        q.wait_and_throw();
        if (N > 0) {
            q.memcpy(dinp, d_dinp, sizeof(float) * N).wait();
            std::fwrite(dinp, sizeof(float), N, fout);
        }
    }

    std::fclose(fout);
    sycl::free(d_inp, q);
    sycl::free(d_dout, q);
    sycl::free(d_dinp, q);
    std::free(inp);
    std::free(dout_host);
    std::free(dinp);
    return 0;
}
