#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

using floatX = float;

// Kernel 1: parallel over B*T, loop over C
class EncoderKernel1;
class EncoderKernel2;
class EncoderKernel3;

static void encoder_forward1(sycl::queue &q, floatX *out, const int *inp,
                             const floatX *wte, const floatX *wpe,
                             int B, int T, int C) {
  const std::size_t N = static_cast<std::size_t>(B) * static_cast<std::size_t>(T);
  if (N == 0)
    return;
  q.parallel_for<EncoderKernel1>(sycl::range<1>(N), [=](sycl::id<1> id) {
    const std::size_t idx = id[0];
    const std::size_t t = idx % static_cast<std::size_t>(T);
    const std::size_t b = idx / static_cast<std::size_t>(T);
    const std::size_t bt = b * static_cast<std::size_t>(T) + t;
    floatX *out_bt = out + bt * static_cast<std::size_t>(C);
    const int ix = inp[bt];
    const floatX *wte_ix =
        wte + static_cast<std::size_t>(ix) * static_cast<std::size_t>(C);
    const floatX *wpe_t = wpe + t * static_cast<std::size_t>(C);
    for (int i = 0; i < C; ++i) {
      out_bt[i] = wte_ix[i] + wpe_t[i];
    }
  });
  q.wait();
}

// Kernel 2: parallel over all of B*T*C
static void encoder_forward2(sycl::queue &q, floatX *out, const int *inp,
                             const floatX *wte, const floatX *wpe,
                             int B, int T, int C) {
  const std::size_t N = static_cast<std::size_t>(B) *
                        static_cast<std::size_t>(T) *
                        static_cast<std::size_t>(C);
  if (N == 0)
    return;
  const std::size_t Tsz = static_cast<std::size_t>(T);
  const std::size_t Csz = static_cast<std::size_t>(C);
  q.parallel_for<EncoderKernel2>(sycl::range<1>(N), [=](sycl::id<1> id) {
    const std::size_t idx = id[0];
    const std::size_t c = idx % Csz;
    const std::size_t bt = idx / Csz;
    const std::size_t t = bt % Tsz;
    const std::size_t b = bt / Tsz;
    const int ix = inp[bt];
    out[b * Tsz * Csz + t * Csz + c] =
        wte[static_cast<std::size_t>(ix) * Csz + c] + wpe[t * Csz + c];
  });
  q.wait();
}

// Kernel 3: vectorized with float4 (sycl::vec<float,4>), parallel over B*T rows
static void encoder_forward3(sycl::queue &q, floatX *out, const int *inp,
                             const floatX *wte, const floatX *wpe,
                             int B, int T, int C) {
  const std::size_t N = static_cast<std::size_t>(B) * static_cast<std::size_t>(T);
  if (N == 0)
    return;
  const std::size_t Tsz = static_cast<std::size_t>(T);
  const std::size_t Csz = static_cast<std::size_t>(C);
  using float4 = sycl::vec<float, 4>;
  q.parallel_for<EncoderKernel3>(sycl::range<1>(N), [=](sycl::id<1> id) {
    const std::size_t idx = id[0];
    const std::size_t t = idx % Tsz;
    const std::size_t b = idx / Tsz;
    const std::size_t bt = b * Tsz + t;
    const int ix = inp[bt];
    floatX *out_bt = out + bt * Csz;
    const floatX *wte_ix =
        wte + static_cast<std::size_t>(ix) * Csz;
    const floatX *wpe_t = wpe + t * Csz;
    const std::size_t nvec = Csz / 4;
    const auto *wte4 = reinterpret_cast<const float4 *>(wte_ix);
    const auto *wpe4 = reinterpret_cast<const float4 *>(wpe_t);
    auto *out4 = reinterpret_cast<float4 *>(out_bt);
    for (std::size_t i = 0; i < nvec; ++i) {
      out4[i] = wte4[i] + wpe4[i];
    }
    for (std::size_t i = nvec * 4; i < Csz; ++i) {
      out_bt[i] = wte_ix[i] + wpe_t[i];
    }
  });
  q.wait();
}

static void encoder_forward(sycl::queue &q, int kernel_num, floatX *out,
                            const int *inp, const floatX *wte,
                            const floatX *wpe, int B, int T, int C) {
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

  try {
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

    const int variants[] = {1, 2, 3};
    for (int vi = 0; vi < 3; ++vi) {
      const int k = variants[vi];
      q.fill(d_out, 0.0f, n_out).wait();
      encoder_forward(q, k, d_out, d_inp, d_wte, d_wpe, B, T, C);
      q.memcpy(out, d_out, sizeof(float) * n_out).wait();
      std::fwrite(out, sizeof(float), n_out, fout);
    }

    sycl::free(d_inp, q);
    sycl::free(d_wte, q);
    sycl::free(d_wpe, q);
    sycl::free(d_out, q);
  } catch (const sycl::exception &e) {
    std::fprintf(stderr, "SYCL error: %s\n", e.what());
    return 1;
  }

  std::fclose(fout);
  std::free(inp);
  std::free(wte);
  std::free(wpe);
  std::free(out);
  return 0;
}
