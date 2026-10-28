#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <vector>
#include <sycl/sycl.hpp>

struct EncoderBackwardK1Tag {};
struct EncoderBackwardK2Tag {};

static void encoder_backward1(sycl::queue &q, float *dwte, float *dwpe,
                              const float *dout, const int *inp, int B, int T,
                              int C) {
  const long N = (long)B * (long)T * (long)C;
  if (N <= 0) {
    return;
  }
  q.parallel_for<EncoderBackwardK1Tag>(
      sycl::range<1>((std::size_t)N), [=](sycl::id<1> id) {
        const long idx = (long)id[0];
        const long bt = idx / (long)C;
        const long b = bt / (long)T;
        const long t = bt % (long)T;
        const long c = idx % (long)C;
        const int ix = inp[(std::size_t)b * (std::size_t)T + (std::size_t)t];
        const float v =
            dout[(std::size_t)b * (std::size_t)T * (std::size_t)C +
                 (std::size_t)t * (std::size_t)C + (std::size_t)c];
        {
          sycl::atomic_ref<float, sycl::memory_order::relaxed,
                           sycl::memory_scope::device,
                           sycl::access::address_space::global_space>
              atom(dwte[(std::size_t)ix * (std::size_t)C + (std::size_t)c]);
          atom.fetch_add(v);
        }
        {
          sycl::atomic_ref<float, sycl::memory_order::relaxed,
                           sycl::memory_scope::device,
                           sycl::access::address_space::global_space>
              atom(dwpe[(std::size_t)t * (std::size_t)C + (std::size_t)c]);
          atom.fetch_add(v);
        }
      });
}

static void encoder_backward2(sycl::queue &q, float *dwte, float *dwpe,
                              const float *dout, const int *inp, int B, int T,
                              int C) {
  if (C <= 0) {
    return;
  }
  const long BT = (long)B * (long)T;
  q.parallel_for<EncoderBackwardK2Tag>(
      sycl::range<1>((std::size_t)C), [=](sycl::id<1> id) {
        const long c = (long)id[0];
        for (long i = 0; i < BT; ++i) {
          const long t = (T == 0) ? 0 : (i % (long)T);
          const int ix = inp[(std::size_t)i];
          const float v = dout[(std::size_t)i * (std::size_t)C + (std::size_t)c];
          dwte[(std::size_t)ix * (std::size_t)C + (std::size_t)c] += v;
          dwpe[(std::size_t)t * (std::size_t)C + (std::size_t)c] += v;
        }
      });
}

static void encoder_backward(sycl::queue &q, int kernel_num, float *dwte,
                             float *dwpe, const float *dout, const int *inp,
                             int B, int T, int C) {
  switch (kernel_num) {
  case 1:
    encoder_backward1(q, dwte, dwpe, dout, inp, B, T, C);
    break;
  case 2:
    encoder_backward2(q, dwte, dwpe, dout, inp, B, T, C);
    break;
  default:
    std::fprintf(stderr, "Invalid kernel number\n");
    std::exit(1);
  }
  q.wait();
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

  const std::size_t n_dout = (std::size_t)B * (std::size_t)T * (std::size_t)C;
  const std::size_t n_inp = (std::size_t)B * (std::size_t)T;
  const std::size_t n_wte = (std::size_t)V * (std::size_t)C;
  const std::size_t n_wpe = (std::size_t)T * (std::size_t)C;

  FILE *fin = std::fopen(argv[1], "rb");
  if (!fin) {
    std::perror(argv[1]);
    return 2;
  }
  std::vector<float> dout(n_dout);
  if (n_dout > 0) {
    if (std::fread(dout.data(), sizeof(float), n_dout, fin) != n_dout) {
      std::fprintf(stderr, "input too short at dout\n");
      return 2;
    }
  }
  std::vector<int> inp(n_inp);
  if (n_inp > 0) {
    if (std::fread(inp.data(), sizeof(int), n_inp, fin) != n_inp) {
      std::fprintf(stderr, "input too short at inp\n");
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
  float *d_dout = sycl::malloc_device<float>(n_dout == 0 ? 1 : n_dout, q);
  int *d_inp = sycl::malloc_device<int>(n_inp == 0 ? 1 : n_inp, q);
  float *d_dwte = sycl::malloc_device<float>(n_wte == 0 ? 1 : n_wte, q);
  float *d_dwpe = sycl::malloc_device<float>(n_wpe == 0 ? 1 : n_wpe, q);
  if (!d_dout || !d_inp || !d_dwte || !d_dwpe) {
    std::fprintf(stderr, "device allocation failed\n");
    return 1;
  }
  if (n_dout > 0) {
    q.memcpy(d_dout, dout.data(), n_dout * sizeof(float)).wait();
  }
  if (n_inp > 0) {
    q.memcpy(d_inp, inp.data(), n_inp * sizeof(int)).wait();
  }

  std::vector<float> dwte(n_wte);
  std::vector<float> dwpe(n_wpe);
  const int variants[] = {1, 2};
  for (int vi = 0; vi < 2; ++vi) {
    const int k = variants[vi];
    if (n_wte > 0) {
      q.fill(d_dwte, 0.0f, n_wte).wait();
    }
    if (n_wpe > 0) {
      q.fill(d_dwpe, 0.0f, n_wpe).wait();
    }
    encoder_backward(q, k, d_dwte, d_dwpe, d_dout, d_inp, B, T, C);
    if (n_wte > 0) {
      q.memcpy(dwte.data(), d_dwte, n_wte * sizeof(float)).wait();
      std::fwrite(dwte.data(), sizeof(float), n_wte, fout);
    }
    if (n_wpe > 0) {
      q.memcpy(dwpe.data(), d_dwpe, n_wpe * sizeof(float)).wait();
      std::fwrite(dwpe.data(), sizeof(float), n_wpe, fout);
    }
  }
  std::fclose(fout);

  sycl::free(d_dout, q);
  sycl::free(d_inp, q);
  sycl::free(d_dwte, q);
  sycl::free(d_dwpe, q);
  return 0;
}
