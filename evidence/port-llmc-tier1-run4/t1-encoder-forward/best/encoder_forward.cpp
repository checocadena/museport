#include <sycl/sycl.hpp>

#include <cstdio>
#include <cstdlib>

// fp32 build
using floatX = float;
constexpr int kVecWidth = 4;
using vec4f = sycl::vec<float, 4>;

// ----------------------------------------------------------------------------
// SYCL kernels: one distinct kernel per variant.

// Variant 1: parallelize over B*T, loop over C.
class EncoderForwardK1;
void encoder_forward1(sycl::queue &q, floatX *out, const int *inp,
                      const floatX *wte, const floatX *wpe, int B, int T,
                      int C) {
  const int N = B * T;
  constexpr int wg_size = 256;
  const int n_wg = (N + wg_size - 1) / wg_size;
  if (n_wg == 0) return;
  q.submit([&](sycl::handler &h) {
    h.parallel_for<EncoderForwardK1>(
        sycl::nd_range<1>(sycl::range<1>(n_wg * wg_size),
                          sycl::range<1>(wg_size)),
        [=](sycl::nd_item<1> it) {
          const int idx = static_cast<int>(it.get_global_id(0));
          if (idx < N) {
            const int b = idx / T;
            const int t = idx % T;
            floatX *out_bt = out + static_cast<size_t>(b) * T * C + t * C;
            const int ix = inp[static_cast<size_t>(b) * T + t];
            const floatX *wte_ix = wte + static_cast<size_t>(ix) * C;
            const floatX *wpe_t = wpe + static_cast<size_t>(t) * C;
            for (int i = 0; i < C; ++i) {
              out_bt[i] = wte_ix[i] + wpe_t[i];
            }
          }
        });
  });
  q.wait();
}

// Variant 2: parallelize over all of B*T*C.
class EncoderForwardK2;
void encoder_forward2(sycl::queue &q, floatX *out, const int *inp,
                      const floatX *wte, const floatX *wpe, int B, int T,
                      int C) {
  const int N = B * T * C;
  constexpr int wg_size = 256;
  const int n_wg = (N + wg_size - 1) / wg_size;
  if (n_wg == 0) return;
  q.submit([&](sycl::handler &h) {
    h.parallel_for<EncoderForwardK2>(
        sycl::nd_range<1>(sycl::range<1>(n_wg * wg_size),
                          sycl::range<1>(wg_size)),
        [=](sycl::nd_item<1> it) {
          const int idx = static_cast<int>(it.get_global_id(0));
          if (idx < N) {
            const int bt = idx / C;
            const int b = bt / T;
            const int t = bt % T;
            const int c = idx % C;
            const int ix = inp[static_cast<size_t>(b) * T + t];
            out[static_cast<size_t>(b) * T * C + t * C + c] =
                wte[static_cast<size_t>(ix) * C + c] + wpe[static_cast<size_t>(t) * C + c];
          }
        });
  });
  q.wait();
}

// Variant 3: like variant 2 but with vector-width (float4) reads/writes.
class EncoderForwardK3;
void encoder_forward3(sycl::queue &q, floatX *out, const int *inp,
                      const floatX *wte, const floatX *wpe, int B, int T,
                      int C) {
  const int N = B * T * C;
  const int n_packs = (N + kVecWidth - 1) / kVecWidth;
  constexpr int wg_size = 256;
  const int n_wg = (n_packs + wg_size - 1) / wg_size;
  if (n_wg == 0) return;
  q.submit([&](sycl::handler &h) {
    h.parallel_for<EncoderForwardK3>(
        sycl::nd_range<1>(sycl::range<1>(n_wg * wg_size),
                          sycl::range<1>(wg_size)),
        [=](sycl::nd_item<1> it) {
          const int pack = static_cast<int>(it.get_global_id(0));
          if (pack < n_packs) {
            const int idx = pack * kVecWidth;
            const int bt = idx / C;
            const int b = bt / T;
            const int t = bt % T;
            const int c = idx % C;
            const int ix = inp[static_cast<size_t>(b) * T + t];
            floatX *out_btc =
                out + static_cast<size_t>(b) * T * C + t * C + c;
            const floatX *wte_ix =
                wte + static_cast<size_t>(ix) * C + c;
            const floatX *wpe_tc =
                wpe + static_cast<size_t>(t) * C + c;
            if (idx + kVecWidth <= N &&
                static_cast<size_t>(c) + kVecWidth <= static_cast<size_t>(C)) {
              const vec4f wte_v =
                  reinterpret_cast<const vec4f *>(wte_ix)[0];
              const vec4f wpe_v =
                  reinterpret_cast<const vec4f *>(wpe_tc)[0];
              reinterpret_cast<vec4f *>(out_btc)[0] = wte_v + wpe_v;
            } else {
              // Scalar tail (also covers any C not divisible by 4).
              for (int k = 0; k < kVecWidth; ++k) {
                if (idx + k < N &&
                    static_cast<size_t>(c) + k < static_cast<size_t>(C)) {
                  out_btc[k] = wte_ix[k] + wpe_tc[k];
                }
              }
            }
          }
        });
  });
  q.wait();
}

void encoder_forward(int kernel_num, sycl::queue &q, floatX *out,
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
      std::printf("Invalid kernel number\n");
      std::exit(1);
  }
}

// ----------------------------------------------------------------------------
// Contract main: <input> <output> B T C V; runs every kernel variant and
// writes each variant's outputs, in order, as raw float32.
int main(int argc, char **argv) {
  if (argc != 7) {
    std::fprintf(stderr, "usage: %s <input> <output> B T C V\n", argv[0]);
    return 2;
  }
  const int B = std::atoi(argv[3]);
  const int T = std::atoi(argv[4]);
  const int C = std::atoi(argv[5]);
  const int V = std::atoi(argv[6]);

  FILE *fin = std::fopen(argv[1], "rb");
  if (!fin) {
    std::perror(argv[1]);
    return 2;
  }
  int *inp = static_cast<int *>(std::malloc(sizeof(int) * (size_t)(B * T)));
  if (std::fread(inp, sizeof(int), (size_t)(B * T), fin) != (size_t)(B * T)) {
    std::fprintf(stderr, "input too short at inp\n");
    return 2;
  }
  float *wte =
      static_cast<float *>(std::malloc(sizeof(float) * (size_t)(V * C)));
  if (std::fread(wte, sizeof(float), (size_t)(V * C), fin) !=
      (size_t)(V * C)) {
    std::fprintf(stderr, "input too short at wte\n");
    return 2;
  }
  float *wpe =
      static_cast<float *>(std::malloc(sizeof(float) * (size_t)(T * C)));
  if (std::fread(wpe, sizeof(float), (size_t)(T * C), fin) !=
      (size_t)(T * C)) {
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

  int *d_inp = sycl::malloc_device<int>((size_t)(B * T), q);
  float *d_wte = sycl::malloc_device<float>((size_t)(V * C), q);
  float *d_wpe = sycl::malloc_device<float>((size_t)(T * C), q);
  float *d_out = sycl::malloc_device<float>((size_t)(B * T * C), q);
  float *out =
      static_cast<float *>(std::malloc(sizeof(float) * (size_t)(B * T * C)));

  q.memcpy(d_inp, inp, sizeof(int) * (size_t)(B * T)).wait();
  q.memcpy(d_wte, wte, sizeof(float) * (size_t)(V * C)).wait();
  q.memcpy(d_wpe, wpe, sizeof(float) * (size_t)(T * C)).wait();

  const int variants[] = {1, 2, 3};
  for (int vi = 0; vi < 3; ++vi) {
    const int k = variants[vi];
    q.memset(d_out, 0, sizeof(float) * (size_t)(B * T * C)).wait();
    encoder_forward(k, q, d_out, d_inp, d_wte, d_wpe, B, T, C);
    q.memcpy(out, d_out, sizeof(float) * (size_t)(B * T * C)).wait();
    std::fwrite(out, sizeof(float), (size_t)(B * T * C), fout);
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
