#include <sycl/sycl.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cstdint>

// ----------------------------------------------------------------------------
// SYCL kernels: GPT-2 positional encoder backward pass (fp32)

// Variant 1: parallelize over B*T*C, accumulate with global atomics.
void encoder_backward1(sycl::queue &q, float *dwte, float *dwpe,
                       const float *dout, const int *inp, int B, int T,
                       int C) {
  const std::size_t N = static_cast<std::size_t>(B) * static_cast<std::size_t>(T) *
                        static_cast<std::size_t>(C);
  if (N == 0) {
    return;
  }
  q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
    const std::size_t i = idx[0];
    const std::size_t c_sz = static_cast<std::size_t>(C);
    const std::size_t t_sz = static_cast<std::size_t>(T);
    const std::size_t bt = i / c_sz;
    const std::size_t b = bt / t_sz;
    const std::size_t t = bt % t_sz;
    const std::size_t c = i % c_sz;

    const int ix = inp[b * t_sz + t];
    const float grad = dout[b * t_sz * c_sz + t * c_sz + c];

    float *dwte_ptr = dwte + static_cast<std::size_t>(ix) * c_sz + c;
    float *dwpe_ptr = dwpe + t * c_sz + c;
    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space>
        a_wte(*dwte_ptr);
    a_wte.fetch_add(grad);
    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space>
        a_wpe(*dwpe_ptr);
    a_wpe.fetch_add(grad);
  });
}

// Variant 2: parallelize over C, sequential loop over B*T (no atomics needed:
// each work-item owns a distinct channel c).
void encoder_backward2(sycl::queue &q, float *dwte, float *dwpe,
                       const float *dout, const int *inp, int B, int T,
                       int C) {
  if (C <= 0) {
    return;
  }
  const std::size_t uC = static_cast<std::size_t>(C);
  const std::size_t uT = static_cast<std::size_t>(T);
  const std::size_t BT = static_cast<std::size_t>(B) * uT;
  q.parallel_for(sycl::range<1>(uC), [=](sycl::id<1> id) {
    const std::size_t c = id[0];
    for (std::size_t i = 0; i < BT; ++i) {
      const std::size_t t = i % uT;
      const int ix = inp[i];
      const float grad = dout[i * uC + c];
      dwte[static_cast<std::size_t>(ix) * uC + c] += grad;
      dwpe[t * uC + c] += grad;
    }
  });
}

void encoder_backward(int kernel_num, sycl::queue &q, float *dwte, float *dwpe,
                      const float *dout, const int *inp, int B, int T, int C) {
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

  const std::size_t n_dout = static_cast<std::size_t>(B) *
                             static_cast<std::size_t>(T) *
                             static_cast<std::size_t>(C);
  const std::size_t n_inp =
      static_cast<std::size_t>(B) * static_cast<std::size_t>(T);
  const std::size_t n_dwte =
      static_cast<std::size_t>(V) * static_cast<std::size_t>(C);
  const std::size_t n_dwpe =
      static_cast<std::size_t>(T) * static_cast<std::size_t>(C);

  FILE *fin = std::fopen(argv[1], "rb");
  if (!fin) {
    std::perror(argv[1]);
    return 2;
  }
  float *dout = static_cast<float *>(std::malloc(sizeof(float) * n_dout));
  int *inp = static_cast<int *>(std::malloc(sizeof(int) * n_inp));
  float *dwte = static_cast<float *>(std::malloc(sizeof(float) * n_dwte));
  float *dwpe = static_cast<float *>(std::malloc(sizeof(float) * n_dwpe));
  if ((n_dout && !dout) || (n_inp && !inp) || (n_dwte && !dwte) ||
      (n_dwpe && !dwpe)) {
    std::fprintf(stderr, "host allocation failed\n");
    return 2;
  }
  if (std::fread(dout, sizeof(float), n_dout, fin) != n_dout) {
    std::fprintf(stderr, "input too short at dout\n");
    return 2;
  }
  if (std::fread(inp, sizeof(int), n_inp, fin) != n_inp) {
    std::fprintf(stderr, "input too short at inp\n");
    return 2;
  }
  std::fclose(fin);

  FILE *fout = std::fopen(argv[2], "wb");
  if (!fout) {
    std::perror(argv[2]);
    return 2;
  }

  try {
    sycl::queue q{sycl::property::queue::in_order{}};

    float *d_dout = sycl::malloc_device<float>(n_dout, q);
    int *d_inp = sycl::malloc_device<int>(n_inp, q);
    float *d_dwte = sycl::malloc_device<float>(n_dwte, q);
    float *d_dwpe = sycl::malloc_device<float>(n_dwpe, q);
    if ((n_dout && !d_dout) || (n_inp && !d_inp) || (n_dwte && !d_dwte) ||
        (n_dwpe && !d_dwpe)) {
      std::fprintf(stderr, "device allocation failed\n");
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
      if (n_dwte) {
        q.memset(d_dwte, 0, sizeof(float) * n_dwte).wait();
      }
      if (n_dwpe) {
        q.memset(d_dwpe, 0, sizeof(float) * n_dwpe).wait();
      }
      encoder_backward(k, q, d_dwte, d_dwpe, d_dout, d_inp, B, T, C);
      q.wait();
      if (n_dwte) {
        q.memcpy(dwte, d_dwte, sizeof(float) * n_dwte).wait();
        std::fwrite(dwte, sizeof(float), n_dwte, fout);
      }
      if (n_dwpe) {
        q.memcpy(dwpe, d_dwpe, sizeof(float) * n_dwpe).wait();
        std::fwrite(dwpe, sizeof(float), n_dwpe, fout);
      }
    }

    sycl::free(d_dout, q);
    sycl::free(d_inp, q);
    sycl::free(d_dwte, q);
    sycl::free(d_dwpe, q);
  } catch (const sycl::exception &e) {
    std::fprintf(stderr, "SYCL error: %s\n", e.what());
    return 1;
  }

  std::fclose(fout);
  std::free(dout);
  std::free(inp);
  std::free(dwte);
  std::free(dwpe);
  return 0;
}
