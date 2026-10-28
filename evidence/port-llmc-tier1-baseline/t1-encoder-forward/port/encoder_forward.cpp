#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

using floatX = float;

// 128-bit packed vector of 4 fp32 values (fp32 build).
struct x128 {
  static constexpr int size = 4;
  float payload[size];
  float &operator[](int i) { return payload[i]; }
  const float &operator[](int i) const { return payload[i]; }
};

inline x128 load128cs(const float *p) {
  x128 r;
#pragma unroll
  for (int k = 0; k < x128::size; ++k) {
    r.payload[k] = p[k];
  }
  return r;
}

inline void store128(float *p, x128 v) {
#pragma unroll
  for (int k = 0; k < x128::size; ++k) {
    p[k] = v.payload[k];
  }
}

template <typename T> inline T ceil_div(T dividend, T divisor) {
  return (dividend + divisor - 1) / divisor;
}

class EncoderKernel1;
class EncoderKernel2;
class EncoderKernel3;

void encoder_forward1(sycl::queue &q, floatX *out, const int *inp,
                      const floatX *wte, const floatX *wpe, int B, int T,
                      int C) {
  constexpr long WG = 256;
  const long N = static_cast<long>(B) * T;
  if (N <= 0 || C <= 0) {
    return;
  }
  const long G = ceil_div(N, WG) * WG;
  q.parallel_for<EncoderKernel1>(sycl::nd_range<1>(sycl::range<1>(G), sycl::range<1>(WG)),
                                 [=](sycl::nd_item<1> it) {
                                   long idx = static_cast<long>(it.get_global_id(0));
                                   if (idx < N) {
                                     long b = idx / T;
                                     long t = idx % T;
                                     floatX *out_bt = out + b * T * C + t * C;
                                     long ix = inp[b * T + t];
                                     const floatX *wte_ix = wte + ix * C;
                                     const floatX *wpe_t = wpe + t * C;
                                     for (int i = 0; i < C; ++i) {
                                       out_bt[i] = wte_ix[i] + wpe_t[i];
                                     }
                                   }
                                 })
      .wait();
}

void encoder_forward2(sycl::queue &q, floatX *out, const int *inp,
                      const floatX *wte, const floatX *wpe, int B, int T,
                      int C) {
  constexpr long WG = 256;
  const long N = static_cast<long>(B) * T * C;
  if (N <= 0) {
    return;
  }
  const long G = ceil_div(N, WG) * WG;
  q.parallel_for<EncoderKernel2>(sycl::nd_range<1>(sycl::range<1>(G), sycl::range<1>(WG)),
                                 [=](sycl::nd_item<1> it) {
                                   long idx = static_cast<long>(it.get_global_id(0));
                                   if (idx < N) {
                                     long bt = idx / C;
                                     long b = bt / T;
                                     long t = bt % T;
                                     long c = idx % C;
                                     long ix = inp[b * T + t];
                                     out[b * T * C + t * C + c] =
                                         wte[ix * C + c] + wpe[t * C + c];
                                   }
                                 })
      .wait();
}

void encoder_forward3(sycl::queue &q, floatX *out, const int *inp,
                      const floatX *wte, const floatX *wpe, int B, int T,
                      int C) {
  constexpr long WG = 256;
  const long N = static_cast<long>(B) * T * C;
  if (N <= 0) {
    return;
  }
  constexpr long VEC = x128::size;
  const long NV = ceil_div(N, VEC);
  const long G = ceil_div(NV, WG) * WG;
  q.parallel_for<EncoderKernel3>(sycl::nd_range<1>(sycl::range<1>(G), sycl::range<1>(WG)),
                                 [=](sycl::nd_item<1> it) {
                                   long vid = static_cast<long>(it.get_global_id(0));
                                   if (vid >= NV) {
                                     return;
                                   }
                                   long idx = vid * VEC;
                                   // Fast path: whole vector inside bounds and
                                   // inside a single (b,t) row, exactly like
                                   // the CUDA kernel.
                                   long c = idx % C;
                                   if (idx + VEC <= N && c + VEC <= C) {
                                     long bt = idx / C;
                                     long b = bt / T;
                                     long t = bt % T;
                                     long ix = inp[b * T + t];
                                     floatX *out_btc = out + b * T * C + t * C + c;
                                     const floatX *wte_ix = wte + ix * C + c;
                                     const floatX *wpe_tc = wpe + t * C + c;
                                     x128 packed_out;
                                     x128 a = load128cs(wte_ix);
                                     x128 b2 = load128cs(wpe_tc);
#pragma unroll
                                     for (int k = 0; k < x128::size; ++k) {
                                       packed_out[k] = a[k] + b2[k];
                                     }
                                     store128(out_btc, packed_out);
                                   } else {
                                     // Tail / row-crossing fallback: scalar,
                                     // element-correct per lane.
                                     for (long k = 0; k < VEC; ++k) {
                                       long lin = idx + k;
                                       if (lin >= N) {
                                         break;
                                       }
                                       long bt = lin / C;
                                       long b = bt / T;
                                       long t = bt % T;
                                       long cc = lin % C;
                                       long ix = inp[b * T + t];
                                       out[lin] = wte[ix * C + cc] + wpe[t * C + cc];
                                     }
                                   }
                                 })
      .wait();
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
    std::printf("Invalid kernel number\n");
    std::exit(1);
  }
}

int main(int argc, char **argv) {
  if (argc != 7) {
    std::fprintf(stderr, "usage: %s <input> <output> B T C V\n", argv[0]);
    return 2;
  }
  const int B = std::atoi(argv[3]);
  const int T = std::atoi(argv[4]);
  const int C = std::atoi(argv[5]);
  const int V = std::atoi(argv[6]);
  const size_t n_inp = (size_t)B * (size_t)T;
  const size_t n_wte = (size_t)V * (size_t)C;
  const size_t n_wpe = (size_t)T * (size_t)C;
  const size_t n_out = (size_t)B * (size_t)T * (size_t)C;

  FILE *fin = std::fopen(argv[1], "rb");
  if (!fin) {
    std::perror(argv[1]);
    return 2;
  }
  int *inp = (int *)std::malloc(sizeof(int) * (n_inp ? n_inp : 1));
  float *wte = (float *)std::malloc(sizeof(float) * (n_wte ? n_wte : 1));
  float *wpe = (float *)std::malloc(sizeof(float) * (n_wpe ? n_wpe : 1));
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

  int *d_inp = sycl::malloc_device<int>(n_inp ? n_inp : 1, q);
  float *d_wte = sycl::malloc_device<float>(n_wte ? n_wte : 1, q);
  float *d_wpe = sycl::malloc_device<float>(n_wpe ? n_wpe : 1, q);
  float *d_out = sycl::malloc_device<float>(n_out ? n_out : 1, q);
  if (!d_inp || !d_wte || !d_wpe || !d_out) {
    std::fprintf(stderr, "device malloc failed\n");
    return 2;
  }
  q.memcpy(d_inp, inp, sizeof(int) * n_inp).wait();
  q.memcpy(d_wte, wte, sizeof(float) * n_wte).wait();
  q.memcpy(d_wpe, wpe, sizeof(float) * n_wpe).wait();

  float *out = (float *)std::malloc(sizeof(float) * (n_out ? n_out : 1));
  if (!out) {
    std::fprintf(stderr, "host malloc failed\n");
    return 2;
  }

  const int variants[] = {1, 2, 3};
  for (int vi = 0; vi < 3; ++vi) {
    const int k = variants[vi];
    q.memset(d_out, 0, sizeof(float) * n_out).wait();
    encoder_forward(q, k, d_out, d_inp, d_wte, d_wpe, B, T, C);
    q.wait();
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
