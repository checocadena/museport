#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <sycl/sycl.hpp>

using floatX = float;

namespace {

constexpr float kGeluScaling = 0.7978845608028654f; // sqrtf(2/pi)
constexpr int kBlockSize = 256;
constexpr int kVecSize = 4; // sizeof(int4)/sizeof(float)

template <typename T>
constexpr T ceil_div(T dividend, T divisor) {
    return (dividend + divisor - 1) / divisor;
}

inline float gelu_local_grad(float x) {
    float cube = 0.044715f * x * x * x;
    float tanh_arg = kGeluScaling * (x + cube);
    float tanh_out = sycl::tanh(tanh_arg);
    float cosh_out = sycl::cosh(tanh_arg);
    float sech_out = 1.0f / (cosh_out * cosh_out);
    return 0.5f * (1.0f + tanh_out) +
           x * 0.5f * sech_out * kGeluScaling * (1.0f + 3.0f * 0.044715f * x * x);
}

class GeluBackward1Kernel;
class GeluBackward2Kernel;

void launch_variant1(sycl::queue &q, float *dinp, const float *inp,
                     const float *dout, std::size_t n) {
    if (n == 0) {
        return;
    }
    const std::size_t local = static_cast<std::size_t>(kBlockSize);
    const std::size_t groups = ceil_div(n, local);
    const std::size_t global = groups * local;
    q.submit([&](sycl::handler &h) {
        h.parallel_for<GeluBackward1Kernel>(
            sycl::nd_range<1>{sycl::range<1>{global}, sycl::range<1>{local}},
            [=](sycl::nd_item<1> it) {
                std::size_t i = it.get_global_id(0);
                if (i < n) {
                    float x = inp[i];
                    dinp[i] = gelu_local_grad(x) * dout[i];
                }
            });
    });
}

void launch_variant2(sycl::queue &q, float *dinp, const float *inp,
                     const float *dout, std::size_t n) {
    if (n == 0) {
        return;
    }
    // One work-item handles a 128-bit (4x fp32) packet, mirroring x128.
    const std::size_t packets = ceil_div(n, static_cast<std::size_t>(kVecSize));
    const std::size_t local = static_cast<std::size_t>(kBlockSize);
    const std::size_t groups = ceil_div(packets, local);
    const std::size_t global = groups * local;
    q.submit([&](sycl::handler &h) {
        h.parallel_for<GeluBackward2Kernel>(
            sycl::nd_range<1>{sycl::range<1>{global}, sycl::range<1>{local}},
            [=](sycl::nd_item<1> it) {
                std::size_t base = it.get_global_id(0) * kVecSize;
                if (base >= n) {
                    return;
                }
                // Vector-width load/compute/store of one packet.
                if (base + kVecSize <= n) {
                    const auto *vin =
                        reinterpret_cast<const sycl::vec<float, 4> *>(inp);
                    const auto *vdout4 =
                        reinterpret_cast<const sycl::vec<float, 4> *>(dout);
                    auto *vdinp4 = reinterpret_cast<sycl::vec<float, 4> *>(dinp);
                    sycl::vec<float, 4> a = vin[base / kVecSize];
                    sycl::vec<float, 4> b = vdout4[base / kVecSize];
                    sycl::vec<float, 4> c;
                    for (int k = 0; k < kVecSize; ++k) {
                        c[k] = gelu_local_grad(a[k]) * b[k];
                    }
                    vdinp4[base / kVecSize] = c;
                } else {
                    for (int k = 0; k < kVecSize; ++k) {
                        std::size_t i = base + static_cast<std::size_t>(k);
                        if (i < n) {
                            dinp[i] = gelu_local_grad(inp[i]) * dout[i];
                        }
                    }
                }
            });
    });
}

} // namespace

int main(int argc, char **argv) {
    if (argc != 6) {
        std::fprintf(stderr, "usage: %s <input> <output> B T C\n", argv[0]);
        return 2;
    }
    const long B = std::atol(argv[3]);
    const long T = std::atol(argv[4]);
    const long C = std::atol(argv[5]);
    const std::size_t n = static_cast<std::size_t>(B) * static_cast<std::size_t>(T) *
                          static_cast<std::size_t>(C);

    FILE *fin = std::fopen(argv[1], "rb");
    if (!fin) {
        std::perror(argv[1]);
        return 2;
    }
    float *inp = static_cast<float *>(std::malloc(sizeof(float) * n));
    float *dout_h = static_cast<float *>(std::malloc(sizeof(float) * n));
    float *dinp_h = static_cast<float *>(std::malloc(sizeof(float) * n));
    if (!inp || !dout_h || !dinp_h) {
        std::fprintf(stderr, "host malloc failed\n");
        return 2;
    }
    if (std::fread(inp, sizeof(float), n, fin) != n) {
        std::fprintf(stderr, "input too short at inp\n");
        return 2;
    }
    if (std::fread(dout_h, sizeof(float), n, fin) != n) {
        std::fprintf(stderr, "input too short at dout\n");
        return 2;
    }
    std::fclose(fin);

    FILE *fout = std::fopen(argv[2], "wb");
    if (!fout) {
        std::perror(argv[2]);
        return 2;
    }

    try {
        sycl::queue q{sycl::default_selector_v};

        float *d_inp = sycl::malloc_device<float>(n, q);
        float *d_dout = sycl::malloc_device<float>(n, q);
        float *d_dinp = sycl::malloc_device<float>(n, q);
        if (!d_inp || !d_dout || !d_dinp) {
            std::fprintf(stderr, "device malloc failed\n");
            return 2;
        }

        q.memcpy(d_inp, inp, sizeof(float) * n).wait();
        q.memcpy(d_dout, dout_h, sizeof(float) * n).wait();

        const int variants[] = {1, 2};
        for (int vi = 0; vi < 2; ++vi) {
            q.memset(d_dinp, 0, sizeof(float) * n).wait();
            if (variants[vi] == 1) {
                launch_variant1(q, d_dinp, d_inp, d_dout, n);
            } else {
                launch_variant2(q, d_dinp, d_inp, d_dout, n);
            }
            q.wait();
            q.memcpy(dinp_h, d_dinp, sizeof(float) * n).wait();
            std::fwrite(dinp_h, sizeof(float), n, fout);
        }

        sycl::free(d_inp, q);
        sycl::free(d_dout, q);
        sycl::free(d_dinp, q);
    } catch (const sycl::exception &e) {
        std::fprintf(stderr, "SYCL error: %s\n", e.what());
        return 1;
    }

    std::fclose(fout);
    std::free(inp);
    std::free(dout_h);
    std::free(dinp_h);
    return 0;
}
