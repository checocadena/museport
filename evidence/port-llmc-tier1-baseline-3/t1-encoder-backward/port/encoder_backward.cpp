// GPT-2 positional encoder backward pass, idiomatic SYCL 2020 (fp32).
// Variant 1: parallel over B*T*C with atomics.
// Variant 2: parallel over C, sequential loop over B*T without atomics.
#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstddef>

// Distinct SYCL kernel names, one per variant.
class EncoderBackwardKernel1;
class EncoderBackwardKernel2;

void encoder_backward1(sycl::queue &q, float *dwte, float *dwpe,
                       const float *dout, const int *inp, int B, int T, int C,
                       [[maybe_unused]] int block_size) {
  const std::size_t N = static_cast<std::size_t>(B) * static_cast<std::size_t>(T) *
                        static_cast<std::size_t>(C);
  if (N == 0) {
    return;
  }
  q.parallel_for<EncoderBackwardKernel1>(
      sycl::range<1>(N), [=](sycl::id<1> id) {
        std::size_t idx = id[0];
        int c = static_cast<int>(idx % static_cast<std::size_t>(C));
        std::size_t bt = idx / static_cast<std::size_t>(C);
        int b = static_cast<int>(bt / static_cast<std::size_t>(T));
        int t = static_cast<int>(bt % static_cast<std::size_t>(T));
        int ix = inp[static_cast<std::size_t>(b) * static_cast<std::size_t>(T) +
                     static_cast<std::size_t>(t)];
        float val = dout[idx];
        {
          sycl::atomic_ref<float, sycl::memory_order::relaxed,
                           sycl::memory_scope::device>
              acc(dwte[static_cast<std::size_t>(ix) *
                       static_cast<std::size_t>(C) +
                       static_cast<std::size_t>(c)]);
          acc.fetch_add(val);
        }
        {
          sycl::atomic_ref<float, sycl::memory_order::relaxed,
                           sycl::memory_scope::device>
              acc(dwpe[static_cast<std::size_t>(t) *
                       static_cast<std::size_t>(C) +
                       static_cast<std::size_t>(c)]);
          acc.fetch_add(val);
        }
      });
}

void encoder_backward2(sycl::queue &q, float *dwte, float *dwpe,
                       const float *dout, const int *inp, int B, int T, int C,
                       [[maybe_unused]] int block_size) {
  if (C <= 0) {
    return;
  }
  const std::size_t BT =
      static_cast<std::size_t>(B) * static_cast<std::size_t>(T);
  const std::size_t Csz = static_cast<std::size_t>(C);
  const std::size_t Tsz = static_cast<std::size_t>(T);
  q.parallel_for<EncoderBackwardKernel2>(
      sycl::range<1>(Csz), [=](sycl::id<1> id) {
        std::size_t c = id[0];
        for (std::size_t i = 0; i < BT; ++i) {
          int t = static_cast<int>(i % Tsz);
          int ix = inp[i];
          float val = dout[i * Csz + c];
          dwte[static_cast<std::size_t>(ix) * Csz + c] += val;
          dwpe[static_cast<std::size_t>(t) * Csz + c] += val;
        }
      });
}

void encoder_backward(sycl::queue &q, int kernel_num, float *dwte, float *dwpe,
                      const float *dout, const int *inp, int B, int T, int C,
                      int block_size) {
  switch (kernel_num) {
  case 1:
    encoder_backward1(q, dwte, dwpe, dout, inp, B, T, C, block_size);
    break;
  case 2:
    encoder_backward2(q, dwte, dwpe, dout, inp, B, T, C, block_size);
    break;
  default:
    std::printf("Invalid kernel number\n");
    std::exit(1);
  }
}

// museport contract main: <input> <output> sizes...; runs every kernel variant
// and writes each variant's outputs, in order, as raw float32.
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
  const std::size_t n_dout =
      static_cast<std::size_t>(B) * static_cast<std::size_t>(T) *
      static_cast<std::size_t>(C);
  const std::size_t n_inp =
      static_cast<std::size_t>(B) * static_cast<std::size_t>(T);
  const std::size_t n_wte =
      static_cast<std::size_t>(V) * static_cast<std::size_t>(C);
  const std::size_t n_wpe =
      static_cast<std::size_t>(T) * static_cast<std::size_t>(C);

  FILE *fin = std::fopen(argv[1], "rb");
  if (!fin) {
    std::perror(argv[1]);
    return 2;
  }
  float *dout = static_cast<float *>(std::malloc(sizeof(float) * n_dout));
  int *inp = static_cast<int *>(std::malloc(sizeof(int) * n_inp));
  if ((n_dout > 0 &&
       std::fread(dout, sizeof(float), n_dout, fin) != n_dout) ||
      (n_inp > 0 && std::fread(inp, sizeof(int), n_inp, fin) != n_inp)) {
    std::fprintf(stderr, "input too short\n");
    std::fclose(fin);
    return 2;
  }
  std::fclose(fin);

  FILE *fout = std::fopen(argv[2], "wb");
  if (!fout) {
    std::perror(argv[2]);
    return 2;
  }

  sycl::queue q{sycl::default_selector_v};

  float *d_dout = sycl::malloc_device<float>(n_dout == 0 ? 1 : n_dout, q);
  int *d_inp = sycl::malloc_device<int>(n_inp == 0 ? 1 : n_inp, q);
  float *d_dwte = sycl::malloc_device<float>(n_wte == 0 ? 1 : n_wte, q);
  float *d_dwpe = sycl::malloc_device<float>(n_wpe == 0 ? 1 : n_wpe, q);
  float *dwte = static_cast<float *>(std::malloc(sizeof(float) * (n_wte == 0 ? 1 : n_wte)));
  float *dwpe = static_cast<float *>(std::malloc(sizeof(float) * (n_wpe == 0 ? 1 : n_wpe)));

  if (n_dout > 0) {
    q.memcpy(d_dout, dout, sizeof(float) * n_dout).wait();
  }
  if (n_inp > 0) {
    q.memcpy(d_inp, inp, sizeof(int) * n_inp).wait();
  }

  const int variants[] = {1, 2};
  for (int vi = 0; vi < 2; ++vi) {
    const int k = variants[vi];
    if (n_wte > 0) {
      q.memset(d_dwte, 0, sizeof(float) * n_wte).wait();
    }
    if (n_wpe > 0) {
      q.memset(d_dwpe, 0, sizeof(float) * n_wpe).wait();
    }
    encoder_backward(q, k, d_dwte, d_dwpe, d_dout, d_inp, B, T, C, 256);
    q.wait();
    if (n_wte > 0) {
      q.memcpy(dwte, d_dwte, sizeof(float) * n_wte).wait();
      std::fwrite(dwte, sizeof(float), n_wte, fout);
    }
    if (n_wpe > 0) {
      q.memcpy(dwpe, d_dwpe, sizeof(float) * n_wpe).wait();
      std::fwrite(dwpe, sizeof(float), n_wpe, fout);
    }
  }
  std::fclose(fout);

  sycl::free(d_dout, q);
  sycl::free(d_inp, q);
  sycl::free(d_dwte, q);
  sycl::free(d_dwpe, q);
  std::free(dout);
  std::free(inp);
  std::free(dwte);
  std::free(dwpe);
  return 0;
}
