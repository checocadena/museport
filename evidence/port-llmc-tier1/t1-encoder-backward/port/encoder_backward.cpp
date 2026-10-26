#include <sycl/sycl.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstddef>

// Atomics over B*T*C (variant 1).
class EncoderBackwardKernel1;
 // Parallel over C, sequential over B*T without atomics (variant 2).
class EncoderBackwardKernel2;

namespace {

inline std::size_t round_up(std::size_t n, std::size_t wg) {
  return ((n + wg - 1u) / wg) * wg;
}

void encoder_backward1(sycl::queue &q, float *dwte, float *dwpe,
                       const float *dout, const int *inp, int B, int T,
                       int C) {
  const std::size_t n = static_cast<std::size_t>(B) * static_cast<std::size_t>(T) *
                        static_cast<std::size_t>(C);
  if (n == 0) {
    return;
  }
  constexpr std::size_t wg = 256;
  const std::size_t global = round_up(n, wg);
  const std::size_t bt_stride = static_cast<std::size_t>(T) * static_cast<std::size_t>(C);
  const std::size_t c_sz = static_cast<std::size_t>(C);
  const std::size_t t_sz = static_cast<std::size_t>(T);
  q.parallel_for<EncoderBackwardKernel1>(
      sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(wg)),
      [=](sycl::nd_item<1> it) {
        const std::size_t idx = it.get_global_id(0);
        if (idx >= n) {
          return;
        }
        const std::size_t bt = idx / c_sz;
        const std::size_t b = bt / t_sz;
        const std::size_t t = bt % t_sz;
        const std::size_t c = idx % c_sz;
        const int ix = inp[b * t_sz + t];
        const float v = dout[b * bt_stride + t * c_sz + c];
        sycl::atomic_ref<float, sycl::memory_order::relaxed,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            a_wte(dwte[static_cast<std::size_t>(ix) * c_sz + c]);
        a_wte.fetch_add(v);
        sycl::atomic_ref<float, sycl::memory_order::relaxed,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            a_wpe(dwpe[t * c_sz + c]);
        a_wpe.fetch_add(v);
      });
}

void encoder_backward2(sycl::queue &q, float *dwte, float *dwpe,
                       const float *dout, const int *inp, int B, int T,
                       int C) {
  if (C <= 0) {
    return;
  }
  const std::size_t c_n = static_cast<std::size_t>(C);
  const std::size_t bt = static_cast<std::size_t>(B) * static_cast<std::size_t>(T);
  const std::size_t t_sz = static_cast<std::size_t>(T);
  constexpr std::size_t wg = 256;
  const std::size_t global = round_up(c_n, wg);
  q.parallel_for<EncoderBackwardKernel2>(
      sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(wg)),
      [=](sycl::nd_item<1> it) {
        const std::size_t c = it.get_global_id(0);
        if (c >= c_n) {
          return;
        }
        for (std::size_t i = 0; i < bt; ++i) {
          const std::size_t t = i % t_sz;
          const int ix = inp[i];
          const float v = dout[i * c_n + c];
          // Distinct c per work-item: no races across items, plain update.
          dwte[static_cast<std::size_t>(ix) * c_n + c] += v;
          dwpe[t * c_n + c] += v;
        }
      });
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 7) {
    std::fprintf(stderr, "usage: %s <input> <output> B T C V\n", argv[0]);
    return 2;
  }
  const int B = std::atoi(argv[3]);
  const int T = std::atoi(argv[4]);
  const int C = std::atoi(argv[5]);
  const int V = std::atoi(argv[6]);

  const std::size_t n_dout =
      static_cast<std::size_t>(B) * static_cast<std::size_t>(T) * static_cast<std::size_t>(C);
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
  if ((n_dout && !dout) || (n_inp && !inp)) {
    std::fprintf(stderr, "host alloc failed\n");
    std::fclose(fin);
    return 2;
  }
  if (std::fread(dout, sizeof(float), n_dout, fin) != n_dout) {
    std::fprintf(stderr, "input too short at dout\n");
    std::fclose(fin);
    return 2;
  }
  if (std::fread(inp, sizeof(int), n_inp, fin) != n_inp) {
    std::fprintf(stderr, "input too short at inp\n");
    std::fclose(fin);
    return 2;
  }
  std::fclose(fin);

  FILE *fout = std::fopen(argv[2], "wb");
  if (!fout) {
    std::perror(argv[2]);
    return 2;
  }

  float *dwte = static_cast<float *>(std::malloc(sizeof(float) * n_wte));
  float *dwpe = static_cast<float *>(std::malloc(sizeof(float) * n_wpe));
  if ((n_wte && !dwte) || (n_wpe && !dwpe)) {
    std::fprintf(stderr, "host alloc failed\n");
    std::fclose(fout);
    return 2;
  }

  sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order{}};

  float *d_dout = sycl::malloc_device<float>(n_dout, q);
  int *d_inp = sycl::malloc_device<int>(n_inp, q);
  float *d_dwte = sycl::malloc_device<float>(n_wte, q);
  float *d_dwpe = sycl::malloc_device<float>(n_wpe, q);
  if ((n_dout && !d_dout) || (n_inp && !d_inp) || (n_wte && !d_dwte) ||
      (n_wpe && !d_dwpe)) {
    std::fprintf(stderr, "device alloc failed\n");
    return 2;
  }

  if (n_dout) {
    q.memcpy(d_dout, dout, sizeof(float) * n_dout).wait();
  }
  if (n_inp) {
    q.memcpy(d_inp, inp, sizeof(int) * n_inp).wait();
  }

  const int variants[] = {1, 2};
  for (int vi = 0; vi < 2; ++vi) {
    const int k = variants[vi];
    if (n_wte) {
      q.memset(d_dwte, 0, sizeof(float) * n_wte).wait();
    }
    if (n_wpe) {
      q.memset(d_dwpe, 0, sizeof(float) * n_wpe).wait();
    }
    if (k == 1) {
      encoder_backward1(q, d_dwte, d_dwpe, d_dout, d_inp, B, T, C);
    } else {
      encoder_backward2(q, d_dwte, d_dwpe, d_dout, d_inp, B, T, C);
    }
    q.wait();
    if (n_wte) {
      q.memcpy(dwte, d_dwte, sizeof(float) * n_wte).wait();
      std::fwrite(dwte, sizeof(float), n_wte, fout);
    }
    if (n_wpe) {
      q.memcpy(dwpe, d_dwpe, sizeof(float) * n_wpe).wait();
      std::fwrite(dwpe, sizeof(float), n_wpe, fout);
    }
  }

  sycl::free(d_dout, q);
  sycl::free(d_inp, q);
  sycl::free(d_dwte, q);
  sycl::free(d_dwpe, q);
  std::free(dout);
  std::free(inp);
  std::free(dwte);
  std::free(dwpe);
  std::fclose(fout);
  return 0;
}
