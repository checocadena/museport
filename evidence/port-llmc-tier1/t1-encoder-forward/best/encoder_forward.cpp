// Idiomatic SYCL 2020 port of encoder_forward harness (fp32).
// Three distinct SYCL kernels, one per variant, computed on the device.
#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstddef>

using floatX = float;

namespace {

template <typename T>
T ceil_div(T dividend, T divisor) {
  return (dividend + divisor - 1) / divisor;
}

constexpr int kBlockSize = 256;

struct EncoderK1 {};
struct EncoderK2 {};
struct EncoderK3Vec {};
struct EncoderK3Scalar {};

void encoder_forward1(sycl::queue &q, floatX *out, const int *inp,
                      const floatX *wte, const floatX *wpe, int B, int T,
                      int C) {
  const std::size_t n = static_cast<std::size_t>(B) * static_cast<std::size_t>(T);
  if (n == 0) {
    return;
  }
  const std::size_t local = static_cast<std::size_t>(kBlockSize);
  const std::size_t n_groups = ceil_div(n, local);
  const std::size_t global = n_groups * local;
  q.submit([&](sycl::handler &h) {
    h.parallel_for<EncoderK1>(
        sycl::nd_range<1>{sycl::range<1>{global}, sycl::range<1>{local}},
        [=](sycl::nd_item<1> it) {
          const std::size_t idx = it.get_global_id(0);
          if (idx < n) {
            const std::size_t b = idx / static_cast<std::size_t>(T);
            const std::size_t t = idx % static_cast<std::size_t>(T);
            floatX *out_bt =
                out + b * static_cast<std::size_t>(T) * static_cast<std::size_t>(C) +
                t * static_cast<std::size_t>(C);
            const int ix =
                inp[b * static_cast<std::size_t>(T) + t];
            const floatX *wte_ix =
                wte + static_cast<std::size_t>(ix) * static_cast<std::size_t>(C);
            const floatX *wpe_t =
                wpe + t * static_cast<std::size_t>(C);
            for (int i = 0; i < C; ++i) {
              out_bt[i] = wte_ix[i] + wpe_t[i];
            }
          }
        });
  }).wait();
}

void encoder_forward2(sycl::queue &q, floatX *out, const int *inp,
                      const floatX *wte, const floatX *wpe, int B, int T,
                      int C) {
  const std::size_t n = static_cast<std::size_t>(B) *
                        static_cast<std::size_t>(T) *
                        static_cast<std::size_t>(C);
  if (n == 0) {
    return;
  }
  const std::size_t local = static_cast<std::size_t>(kBlockSize);
  const std::size_t n_groups = ceil_div(n, local);
  const std::size_t global = n_groups * local;
  q.submit([&](sycl::handler &h) {
    h.parallel_for<EncoderK2>(
        sycl::nd_range<1>{sycl::range<1>{global}, sycl::range<1>{local}},
        [=](sycl::nd_item<1> it) {
          const std::size_t idx = it.get_global_id(0);
          if (idx < n) {
            const std::size_t uuC = static_cast<std::size_t>(C);
            const std::size_t uuT = static_cast<std::size_t>(T);
            const std::size_t bt = idx / uuC;
            const std::size_t b = bt / uuT;
            const std::size_t t = bt % uuT;
            const std::size_t c = idx % uuC;
            const int ix = inp[b * uuT + t];
            out[b * uuT * uuC + t * uuC + c] =
                wte[static_cast<std::size_t>(ix) * uuC + c] +
                wpe[t * uuC + c];
          }
        });
  }).wait();
}

void encoder_forward3(sycl::queue &q, floatX *out, const int *inp,
                      const floatX *wte, const floatX *wpe, int B, int T,
                      int C) {
  const std::size_t uuB = static_cast<std::size_t>(B);
  const std::size_t uuT = static_cast<std::size_t>(T);
  const std::size_t uuC = static_cast<std::size_t>(C);
  const std::size_t n = uuB * uuT * uuC;
  if (n == 0) {
    return;
  }
  // Vector width for 128-bit float4 traffic.
  constexpr std::size_t kWidth = 4;
  if (uuC % kWidth == 0) {
    const std::size_t nvec = n / kWidth;
    const std::size_t local = static_cast<std::size_t>(kBlockSize);
    const std::size_t n_groups = ceil_div(nvec, local);
    const std::size_t global = n_groups * local;
    using vec4 = sycl::vec<float, 4>;
    const vec4 *wte_vec = reinterpret_cast<const vec4 *>(wte);
    const vec4 *wpe_vec = reinterpret_cast<const vec4 *>(wpe);
    vec4 *out_vec = reinterpret_cast<vec4 *>(out);
    q.submit([&](sycl::handler &h) {
      h.parallel_for<EncoderK3Vec>(
          sycl::nd_range<1>{sycl::range<1>{global}, sycl::range<1>{local}},
          [=](sycl::nd_item<1> it) {
            const std::size_t gid = it.get_global_id(0);
            if (gid < nvec) {
              const std::size_t base = gid * kWidth;
              const std::size_t bt = base / uuC;
              const std::size_t b = bt / uuT;
              const std::size_t t = bt % uuT;
              const std::size_t c = base % uuC;
              const int ix = inp[b * uuT + t];
              const std::size_t wte_vi =
                  (static_cast<std::size_t>(ix) * uuC + c) / kWidth;
              const std::size_t wpe_vi = (t * uuC + c) / kWidth;
              out_vec[base / kWidth] = wte_vec[wte_vi] + wpe_vec[wpe_vi];
            }
          });
    }).wait();
  } else {
    // C not divisible by 4: scalar fallback under its own kernel name.
    const std::size_t local = static_cast<std::size_t>(kBlockSize);
    const std::size_t n_groups = ceil_div(n, local);
    const std::size_t global = n_groups * local;
    q.submit([&](sycl::handler &h) {
      h.parallel_for<EncoderK3Scalar>(
          sycl::nd_range<1>{sycl::range<1>{global}, sycl::range<1>{local}},
          [=](sycl::nd_item<1> it) {
            const std::size_t idx = it.get_global_id(0);
            if (idx < n) {
              const std::size_t bt = idx / uuC;
              const std::size_t b = bt / uuT;
              const std::size_t t = bt % uuT;
              const std::size_t c = idx % uuC;
              const int ix = inp[b * uuT + t];
              out[b * uuT * uuC + t * uuC + c] =
                  wte[static_cast<std::size_t>(ix) * uuC + c] +
                  wpe[t * uuC + c];
            }
          });
    }).wait();
  }
}

void encoder_forward(sycl::queue &q, int kernel_num, floatX *out,
                     const int *inp, const floatX *wte, const floatX *wpe,
                     int B, int T, int C) {
  switch (kernel_num) {
    case 1:
      encoder_forward1(q, out, inp, wte, wpe, B, T, C);
      break;
    case 2:
      encoder_forward2(q, out, inp, wte, wpe, B, T, C);
      break;
    case 3:
      encoder_forward3(q, out, inp, wte, wpe, B, T, C);
      break;
    default:
      std::fprintf(stderr, "Invalid kernel number\n");
      std::exit(1);
  }
}

}  // namespace

int main(int argc, char **argv) {
  std::srand(0);
  if (argc != 7) {
    std::fprintf(stderr, "usage: %s <input> <output> B T C V\n", argv[0]);
    return 2;
  }
  const int B = std::atoi(argv[3]);
  const int T = std::atoi(argv[4]);
  const int C = std::atoi(argv[5]);
  const int V = std::atoi(argv[6]);

  const std::size_t n_inp = static_cast<std::size_t>(B) * static_cast<std::size_t>(T);
  const std::size_t n_wte = static_cast<std::size_t>(V) * static_cast<std::size_t>(C);
  const std::size_t n_wpe = static_cast<std::size_t>(T) * static_cast<std::size_t>(C);
  const std::size_t n_out = static_cast<std::size_t>(B) * static_cast<std::size_t>(T) *
                            static_cast<std::size_t>(C);

  FILE *fin = std::fopen(argv[1], "rb");
  if (!fin) {
    std::perror(argv[1]);
    return 2;
  }
  int *inp = static_cast<int *>(std::malloc(sizeof(int) * n_inp));
  float *wte = static_cast<float *>(std::malloc(sizeof(float) * n_wte));
  float *wpe = static_cast<float *>(std::malloc(sizeof(float) * n_wpe));
  if (!inp || !wte || !wpe) {
    std::fprintf(stderr, "host malloc failed\n");
    return 2;
  }
  if (std::fread(inp, sizeof(int), n_inp, fin) != n_inp) {
    std::fprintf(stderr, "input too short at inp\n");
    return 2;
  }
  if (std::fread(wte, sizeof(float), n_wte, fin) != n_wte) {
    std::fprintf(stderr, "input too short at wte\n");
    return 2;
  }
  if (std::fread(wpe, sizeof(float), n_wpe, fin) != n_wpe) {
    std::fprintf(stderr, "input too short at wpe\n");
    return 2;
  }
  std::fclose(fin);

  FILE *fout = std::fopen(argv[2], "wb");
  if (!fout) {
    std::perror(argv[2]);
    return 2;
  }

  sycl::queue q{sycl::default_selector_v};
  int *d_inp = sycl::malloc_device<int>(n_inp, q);
  float *d_wte = sycl::malloc_device<float>(n_wte, q);
  float *d_wpe = sycl::malloc_device<float>(n_wpe, q);
  float *d_out = sycl::malloc_device<float>(n_out, q);
  if (!d_inp || !d_wte || !d_wpe || !d_out) {
    std::fprintf(stderr, "device malloc failed\n");
    return 2;
  }
  q.memcpy(d_inp, inp, sizeof(int) * n_inp).wait();
  q.memcpy(d_wte, wte, sizeof(float) * n_wte).wait();
  q.memcpy(d_wpe, wpe, sizeof(float) * n_wpe).wait();

  float *out = static_cast<float *>(std::malloc(sizeof(float) * n_out));
  if (!out) {
    std::fprintf(stderr, "host malloc failed\n");
    return 2;
  }

  const int variants[] = {1, 2, 3};
  for (int vi = 0; vi < 3; ++vi) {
    const int k = variants[vi];
    q.memset(d_out, 0, sizeof(float) * n_out).wait();
    encoder_forward(q, k, d_out, d_inp, d_wte, d_wpe, B, T, C);
    q.memcpy(out, d_out, sizeof(float) * n_out).wait();
    std::fwrite(out, sizeof(float), n_out, fout);
  }

  std::fclose(fout);
  sycl::free(d_inp, q);
  sycl::free(d_wte, q);
  sycl::free(d_wpe, q);
  sycl::free(d_out, q);
  std::free(inp);
  std::free(wte);
  std::free(wpe);
  std::free(out);
  return 0;
}
