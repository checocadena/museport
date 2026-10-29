// GPT-2 positional encoder forward pass - idiomatic SYCL 2020 port (fp32).
// Three kernel variants, each its own SYCL kernel, computed on the device.
// Contract: build/candidate <input> <output> B T C V
// reads inp[B*T] int32, wte[V*C] float32, wpe[T*C] float32 and writes
// variant 1,2,3 outputs (each B*T*C float32) in order as raw float32.
#include <sycl/sycl.hpp>

#include <cstdio>
#include <cstdlib>

using floatX = float;

namespace {

constexpr int kVecWidth = 4;
using float4_t = sycl::vec<float, kVecWidth>;

inline std::size_t ceil_div_sz(std::size_t n, std::size_t d) {
  return (n + d - 1) / d;
}

// Named kernel tags so each variant is its own SYCL kernel.
class EncoderKernel1;
class EncoderKernel2;
class EncoderKernel3;

void encoder_forward1(sycl::queue &q, floatX *out, const int *inp,
                      const floatX *wte, const floatX *wpe, int B, int T,
                      int C) {
  const std::size_t n = static_cast<std::size_t>(B) * static_cast<std::size_t>(T);
  if (n == 0) return;
  q.parallel_for<class EncoderKernel1>(sycl::range<1>(n), [=](sycl::item<1> it) {
    const std::size_t idx = it.get_id(0);
    const std::size_t t_sz = static_cast<std::size_t>(T);
    const std::size_t c_sz = static_cast<std::size_t>(C);
    const std::size_t b = idx / t_sz;
    const std::size_t t = idx % t_sz;
    floatX *out_bt = out + (b * t_sz + t) * c_sz;
    const int ix = inp[b * t_sz + t];
    const floatX *wte_ix = wte + static_cast<std::size_t>(ix) * c_sz;
    const floatX *wpe_t = wpe + t * c_sz;
    for (std::size_t i = 0; i < c_sz; ++i) {
      out_bt[i] = wte_ix[i] + wpe_t[i];
    }
  });
}

void encoder_forward2(sycl::queue &q, floatX *out, const int *inp,
                      const floatX *wte, const floatX *wpe, int B, int T,
                      int C) {
  const std::size_t n = static_cast<std::size_t>(B) *
                        static_cast<std::size_t>(T) *
                        static_cast<std::size_t>(C);
  if (n == 0) return;
  q.parallel_for<class EncoderKernel2>(sycl::range<1>(n), [=](sycl::item<1> it) {
    const std::size_t idx = it.get_id(0);
    const std::size_t t_sz = static_cast<std::size_t>(T);
    const std::size_t c_sz = static_cast<std::size_t>(C);
    const std::size_t bt = idx / c_sz;
    const std::size_t b = bt / t_sz;
    const std::size_t t = bt % t_sz;
    const std::size_t c = idx % c_sz;
    const int ix = inp[b * t_sz + t];
    out[(b * t_sz + t) * c_sz + c] =
        wte[static_cast<std::size_t>(ix) * c_sz + c] + wpe[t * c_sz + c];
  });
}

void encoder_forward3(sycl::queue &q, floatX *out, const int *inp,
                      const floatX *wte, const floatX *wpe, int B, int T,
                      int C) {
  const std::size_t n = static_cast<std::size_t>(B) *
                        static_cast<std::size_t>(T) *
                        static_cast<std::size_t>(C);
  if (n == 0) return;
  const std::size_t nvec = ceil_div_sz(n, static_cast<std::size_t>(kVecWidth));
  q.parallel_for<class EncoderKernel3>(sycl::range<1>(nvec), [=](sycl::item<1> it) {
    const std::size_t idx = it.get_id(0) * static_cast<std::size_t>(kVecWidth);
    if (idx >= n) return;
    const std::size_t t_sz = static_cast<std::size_t>(T);
    const std::size_t c_sz = static_cast<std::size_t>(C);
    const std::size_t bt = idx / c_sz;
    const std::size_t b = bt / t_sz;
    const std::size_t t = bt % t_sz;
    const std::size_t c = idx % c_sz;
    // Fast vector path when the 4-wide chunk lies inside one (b,t) row.
    if (c + kVecWidth <= c_sz && idx + kVecWidth <= n) {
      const int ix = inp[b * t_sz + t];
      const floatX *wte_p = wte + static_cast<std::size_t>(ix) * c_sz + c;
      const floatX *wpe_p = wpe + t * c_sz + c;
      floatX *out_p = out + (b * t_sz + t) * c_sz + c;
      float4_t vwte(wte_p[0], wte_p[1], wte_p[2], wte_p[3]);
      float4_t vwpe(wpe_p[0], wpe_p[1], wpe_p[2], wpe_p[3]);
      float4_t vout = vwte + vwpe;
      out_p[0] = vout[0];
      out_p[1] = vout[1];
      out_p[2] = vout[2];
      out_p[3] = vout[3];
    } else {
      // Tail / row-straddling chunk: replicate the CUDA kernel3 semantics
      // exactly (one token/position index per 4-wide chunk, contiguous
      // 4-wide access), guarding only the out-of-bounds tail writes.
      const int ix = inp[b * t_sz + t];
      const std::size_t wte_base = static_cast<std::size_t>(ix) * c_sz + c;
      const std::size_t wpe_base = t * c_sz + c;
      for (int k = 0; k < kVecWidth; ++k) {
        const std::size_t pos = idx + static_cast<std::size_t>(k);
        if (pos >= n) break;
        out[pos] = wte[wte_base + static_cast<std::size_t>(k)] +
                   wpe[wpe_base + static_cast<std::size_t>(k)];
      }
    }
  });
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
  int *inp = static_cast<int *>(std::malloc(sizeof(int) * (n_inp ? n_inp : 1)));
  float *wte = static_cast<float *>(std::malloc(sizeof(float) * (n_wte ? n_wte : 1)));
  float *wpe = static_cast<float *>(std::malloc(sizeof(float) * (n_wpe ? n_wpe : 1)));
  float *out = static_cast<float *>(std::malloc(sizeof(float) * (n_out ? n_out : 1)));
  if (!inp || !wte || !wpe || !out) {
    std::fprintf(stderr, "host malloc failed\n");
    return 2;
  }
  if (n_inp && std::fread(inp, sizeof(int), n_inp, fin) != n_inp) {
    std::fprintf(stderr, "input too short at inp\n");
    return 2;
  }
  if (n_wte && std::fread(wte, sizeof(float), n_wte, fin) != n_wte) {
    std::fprintf(stderr, "input too short at wte\n");
    return 2;
  }
  if (n_wpe && std::fread(wpe, sizeof(float), n_wpe, fin) != n_wpe) {
    std::fprintf(stderr, "input too short at wpe\n");
    return 2;
  }
  std::fclose(fin);

  FILE *fout = std::fopen(argv[2], "wb");
  if (!fout) {
    std::perror(argv[2]);
    return 2;
  }

  sycl::queue q;
  int *d_inp = sycl::malloc_device<int>(n_inp ? n_inp : 1, q);
  float *d_wte = sycl::malloc_device<float>(n_wte ? n_wte : 1, q);
  float *d_wpe = sycl::malloc_device<float>(n_wpe ? n_wpe : 1, q);
  float *d_out = sycl::malloc_device<float>(n_out ? n_out : 1, q);
  if (!d_inp || !d_wte || !d_wpe || !d_out) {
    std::fprintf(stderr, "device malloc failed\n");
    return 2;
  }
  if (n_inp) q.memcpy(d_inp, inp, sizeof(int) * n_inp).wait();
  if (n_wte) q.memcpy(d_wte, wte, sizeof(float) * n_wte).wait();
  if (n_wpe) q.memcpy(d_wpe, wpe, sizeof(float) * n_wpe).wait();

  const int variants[] = {1, 2, 3};
  for (int vi = 0; vi < 3; ++vi) {
    const int k = variants[vi];
    if (n_out) q.memset(d_out, 0, sizeof(float) * n_out).wait();
    encoder_forward(q, k, d_out, d_inp, d_wte, d_wpe, B, T, C);
    q.wait();
    if (n_out) q.memcpy(out, d_out, sizeof(float) * n_out).wait();
    if (n_out) std::fwrite(out, sizeof(float), n_out, fout);
  }

  sycl::free(d_inp, q);
  sycl::free(d_wte, q);
  sycl::free(d_wpe, q);
  sycl::free(d_out, q);
  std::fclose(fout);
  std::free(inp);
  std::free(wte);
  std::free(wpe);
  std::free(out);
  return 0;
}
