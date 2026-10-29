#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstddef>

// Kernel variant 1: parallelize over B*T*C, atomically accumulate into
// dwte[ix*C+c] and dwpe[t*C+c].
class EncoderBackwardKernel1;
class EncoderBackwardKernel2;

static void encoder_backward1(sycl::queue &q, float *dwte, float *dwpe,
                              const float *dout, const int *inp, int B, int T,
                              int C) {
  const std::size_t N = static_cast<std::size_t>(B) * static_cast<std::size_t>(T) *
                        static_cast<std::size_t>(C);
  if (N == 0) return;
  q.parallel_for<EncoderBackwardKernel1>(sycl::range<1>(N),
                                         [=](sycl::item<1> it) {
    std::size_t idx = it.get_id(0);
    std::size_t uC = static_cast<std::size_t>(C);
    std::size_t uT = static_cast<std::size_t>(T);
    std::size_t bt = idx / uC;
    std::size_t b = bt / uT;
    std::size_t t = bt % uT;
    std::size_t c = idx % uC;
    int ix = inp[b * uT + t];
    float v = dout[idx];
    {
      sycl::atomic_ref<float, sycl::memory_order::relaxed,
                       sycl::memory_scope::device,
                       sycl::access::address_space::global_space>
          a(dwte[static_cast<std::size_t>(ix) * uC + c]);
      a.fetch_add(v);
    }
    {
      sycl::atomic_ref<float, sycl::memory_order::relaxed,
                       sycl::memory_scope::device,
                       sycl::access::address_space::global_space>
          a(dwpe[t * uC + c]);
      a.fetch_add(v);
    }
  });
}

// Kernel variant 2: parallelize over C, loop over B*T serially per channel.
// No atomics needed since each work-item owns a distinct channel c.
static void encoder_backward2(sycl::queue &q, float *dwte, float *dwpe,
                              const float *dout, const int *inp, int B, int T,
                              int C) {
  if (C <= 0) return;
  const std::size_t BT =
      static_cast<std::size_t>(B) * static_cast<std::size_t>(T);
  const std::size_t uC = static_cast<std::size_t>(C);
  const std::size_t uT = static_cast<std::size_t>(T);
  q.parallel_for<EncoderBackwardKernel2>(sycl::range<1>(uC),
                                         [=](sycl::item<1> it) {
    std::size_t c = it.get_id(0);
    for (std::size_t i = 0; i < BT; ++i) {
      std::size_t t = i % uT;
      int ix = inp[i];
      float v = dout[i * uC + c];
      dwte[static_cast<std::size_t>(ix) * uC + c] += v;
      dwpe[t * uC + c] += v;
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
  const std::size_t n_dout =
      static_cast<std::size_t>(B) * static_cast<std::size_t>(T) * static_cast<std::size_t>(C);
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
  if (n_dout > 0) {
    if (std::fread(dout, sizeof(float), n_dout, fin) != n_dout) {
      std::fprintf(stderr, "input too short at dout\n");
      return 2;
    }
  }
  if (n_inp > 0) {
    if (std::fread(inp, sizeof(int), n_inp, fin) != n_inp) {
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

  sycl::queue q{sycl::default_selector_v,
                sycl::property::queue::in_order{}};

  float *d_dout = sycl::malloc_device<float>(n_dout, q);
  int *d_inp = sycl::malloc_device<int>(n_inp, q);
  float *d_dwte = sycl::malloc_device<float>(n_dwte, q);
  float *d_dwpe = sycl::malloc_device<float>(n_dwpe, q);
  float *dwte = static_cast<float *>(std::malloc(sizeof(float) * n_dwte));
  float *dwpe = static_cast<float *>(std::malloc(sizeof(float) * n_dwpe));

  if (n_dout > 0) q.memcpy(d_dout, dout, sizeof(float) * n_dout).wait();
  if (n_inp > 0) q.memcpy(d_inp, inp, sizeof(int) * n_inp).wait();

  const int variants[] = {1, 2};
  for (int vi = 0; vi < 2; ++vi) {
    const int k = variants[vi];
    if (n_dwte > 0) q.memset(d_dwte, 0, sizeof(float) * n_dwte).wait();
    if (n_dwpe > 0) q.memset(d_dwpe, 0, sizeof(float) * n_dwpe).wait();
    encoder_backward(q, k, d_dwte, d_dwpe, d_dout, d_inp, B, T, C);
    q.wait();
    if (n_dwte > 0) q.memcpy(dwte, d_dwte, sizeof(float) * n_dwte).wait();
    std::fwrite(dwte, sizeof(float), n_dwte, fout);
    if (n_dwpe > 0) q.memcpy(dwpe, d_dwpe, sizeof(float) * n_dwpe).wait();
    std::fwrite(dwpe, sizeof(float), n_dwpe, fout);
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
