#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sycl/sycl.hpp>

using floatX = float;

// ----------------------------------------------------------------------------
// SYCL kernels: one named kernel type per variant
// ----------------------------------------------------------------------------

class EncoderForwardKernel1;
class EncoderForwardKernel2;
class EncoderForwardKernel3;

inline std::size_t ceil_div(std::size_t a, std::size_t b) { return (a + b - 1) / b; }

void encoder_forward1(sycl::queue &q, floatX *out, const int *inp,
                      const floatX *wte, const floatX *wpe, int B, int T, int C) {
  const std::size_t block_size = 256;
  const std::size_t N = static_cast<std::size_t>(B) * static_cast<std::size_t>(T);
  if (N == 0) {
    return;
  }
  const std::size_t global = ceil_div(N, block_size) * block_size;
  q.parallel_for<EncoderForwardKernel1>(
       sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(block_size)),
       [=](sycl::nd_item<1> it) {
         std::size_t idx = it.get_global_id(0);
         if (idx < N) {
           int b = static_cast<int>(idx / static_cast<std::size_t>(T));
           int t = static_cast<int>(idx % static_cast<std::size_t>(T));
           floatX *out_bt = out + static_cast<std::size_t>(b) * T * C + static_cast<std::size_t>(t) * C;
           int ix = inp[static_cast<std::size_t>(b) * T + static_cast<std::size_t>(t)];
           const floatX *wte_ix = wte + static_cast<std::size_t>(ix) * C;
           const floatX *wpe_t = wpe + static_cast<std::size_t>(t) * C;
           for (int i = 0; i < C; ++i) {
             out_bt[i] = wte_ix[i] + wpe_t[i];
           }
         }
       });
}

void encoder_forward2(sycl::queue &q, floatX *out, const int *inp,
                      const floatX *wte, const floatX *wpe, int B, int T, int C) {
  const std::size_t block_size = 256;
  const std::size_t N = static_cast<std::size_t>(B) * static_cast<std::size_t>(T) *
                        static_cast<std::size_t>(C);
  if (N == 0) {
    return;
  }
  const std::size_t global = ceil_div(N, block_size) * block_size;
  const std::size_t Cs = static_cast<std::size_t>(C);
  const std::size_t Ts = static_cast<std::size_t>(T);
  q.parallel_for<EncoderForwardKernel2>(
       sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(block_size)),
       [=](sycl::nd_item<1> it) {
         std::size_t idx = it.get_global_id(0);
         if (idx < N) {
           std::size_t bt = idx / Cs;
           std::size_t b = bt / Ts;
           std::size_t t = bt % Ts;
           std::size_t c = idx % Cs;
           int ix = inp[b * Ts + t];
           out[b * Ts * Cs + t * Cs + c] =
               wte[static_cast<std::size_t>(ix) * Cs + c] + wpe[t * Cs + c];
         }
       });
}

void encoder_forward3(sycl::queue &q, floatX *out, const int *inp,
                      const floatX *wte, const floatX *wpe, int B, int T, int C) {
  constexpr int VEC = 4;
  const std::size_t block_size = 256;
  const std::size_t N = static_cast<std::size_t>(B) * static_cast<std::size_t>(T) *
                        static_cast<std::size_t>(C);
  if (N == 0) {
    return;
  }
  const std::size_t Nvec = ceil_div(N, static_cast<std::size_t>(VEC));
  const std::size_t global = ceil_div(Nvec, block_size) * block_size;
  const std::size_t Cs = static_cast<std::size_t>(C);
  const std::size_t Ts = static_cast<std::size_t>(T);
  q.parallel_for<EncoderForwardKernel3>(
       sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(block_size)),
       [=](sycl::nd_item<1> it) {
         std::size_t vec_id = it.get_global_id(0);
         std::size_t idx = vec_id * VEC;
         if (idx >= N) {
           return;
         }
         std::size_t bt = idx / Cs;
         std::size_t b = bt / Ts;
         std::size_t t = bt % Ts;
         std::size_t c = idx % Cs;
         // Fast path: 4-wide aligned segment fully inside one (b,t) row.
         // Matches CUDA kernel3 when C is a multiple of 4.
         bool aligned = (c + VEC <= Cs) && (c % VEC == 0);
         if (aligned) {
           int ix = inp[b * Ts + t];
           std::size_t out_base = b * Ts * Cs + t * Cs + c;
           std::size_t wte_base = static_cast<std::size_t>(ix) * Cs + c;
           std::size_t wpe_base = t * Cs + c;
           const sycl::vec<float, VEC> *wte_p =
               reinterpret_cast<const sycl::vec<float, VEC> *>(wte + wte_base);
           const sycl::vec<float, VEC> *wpe_p =
               reinterpret_cast<const sycl::vec<float, VEC> *>(wpe + wpe_base);
           sycl::vec<float, VEC> a = *wte_p;
           sycl::vec<float, VEC> p = *wpe_p;
           sycl::vec<float, VEC> o = a + p;
           sycl::vec<float, VEC> *out_p =
               reinterpret_cast<sycl::vec<float, VEC> *>(out + out_base);
           *out_p = o;
         } else {
           // Scalar tail / unaligned fallback preserves exact numerics.
           for (int k = 0; k < VEC; ++k) {
             std::size_t flat = idx + static_cast<std::size_t>(k);
             if (flat >= N) {
               break;
             }
             std::size_t lbt = flat / Cs;
             std::size_t lb = lbt / Ts;
             std::size_t lt = lbt % Ts;
             std::size_t lc = flat % Cs;
             int ix = inp[lb * Ts + lt];
             out[lb * Ts * Cs + lt * Cs + lc] =
                 wte[static_cast<std::size_t>(ix) * Cs + lc] + wpe[lt * Cs + lc];
           }
         }
       });
}

void encoder_forward(sycl::queue &q, int kernel_num, floatX *out, const int *inp,
                     const floatX *wte, const floatX *wpe, int B, int T, int C) {
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

// ----------------------------------------------------------------------------
// Contract main: <input> <output> B T C V; runs every variant, writes outputs
// in order as raw float32.
// ----------------------------------------------------------------------------
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
  const std::size_t n_inp = static_cast<std::size_t>(B) * static_cast<std::size_t>(T);
  const std::size_t n_wte = static_cast<std::size_t>(V) * static_cast<std::size_t>(C);
  const std::size_t n_wpe = static_cast<std::size_t>(T) * static_cast<std::size_t>(C);
  const std::size_t n_out = static_cast<std::size_t>(B) * static_cast<std::size_t>(T) *
                            static_cast<std::size_t>(C);

  int *inp = static_cast<int *>(std::malloc(sizeof(int) * (n_inp ? n_inp : 1)));
  float *wte = static_cast<float *>(std::malloc(sizeof(float) * (n_wte ? n_wte : 1)));
  float *wpe = static_cast<float *>(std::malloc(sizeof(float) * (n_wpe ? n_wpe : 1)));
  float *out = static_cast<float *>(std::malloc(sizeof(float) * (n_out ? n_out : 1)));
  if (!inp || !wte || !wpe || !out) {
    std::fprintf(stderr, "host allocation failed\n");
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
  int *d_inp = sycl::malloc_device<int>(n_inp ? n_inp : 1, q);
  float *d_wte = sycl::malloc_device<float>(n_wte ? n_wte : 1, q);
  float *d_wpe = sycl::malloc_device<float>(n_wpe ? n_wpe : 1, q);
  float *d_out = sycl::malloc_device<float>(n_out ? n_out : 1, q);
  if (!d_inp || !d_wte || !d_wpe || !d_out) {
    std::fprintf(stderr, "device allocation failed\n");
    return 2;
  }
  q.memcpy(d_inp, inp, sizeof(int) * n_inp).wait();
  q.memcpy(d_wte, wte, sizeof(float) * n_wte).wait();
  q.memcpy(d_wpe, wpe, sizeof(float) * n_wpe).wait();

  const int variants[] = {1, 2, 3};
  for (int vi = 0; vi < 3; ++vi) {
    const int k = variants[vi];
    if (n_out) {
      q.memset(d_out, 0, sizeof(float) * n_out).wait();
    }
    encoder_forward(q, k, d_out, d_inp, d_wte, d_wpe, B, T, C);
    q.wait();
    if (n_out) {
      q.memcpy(out, d_out, sizeof(float) * n_out).wait();
      std::fwrite(out, sizeof(float), n_out, fout);
    }
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
