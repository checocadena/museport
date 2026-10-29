#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>

// Variant 1: parallelize over B*T*C, atomics accumulate into dwte / dwpe.
class EncoderBackwardKernel1;

void encoder_backward1(sycl::queue &q, float *dwte, float *dwpe,
                       const float *dout, const int *inp, int B, int T, int C,
                       int block_size) {
  const long N = static_cast<long>(B) * static_cast<long>(T) * static_cast<long>(C);
  if (N <= 0) {
    return;
  }
  const long local = static_cast<long>(block_size);
  const long n_groups = (N + local - 1) / local;
  const long global = n_groups * local;
  q.submit([&](sycl::handler &h) {
    h.parallel_for<EncoderBackwardKernel1>(
        sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(global)),
                          sycl::range<1>(static_cast<size_t>(local))),
        [=](sycl::nd_item<1> it) {
          long idx = static_cast<long>(it.get_global_id(0));
          if (idx < N) {
            long bt = idx / C;
            long b = bt / T;
            long t = bt % T;
            long c = idx % C;
            int ix = inp[b * T + t];
            float v = dout[b * T * C + t * C + c];
            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                dwte_ref(dwte[ix * C + c]);
            dwte_ref.fetch_add(v);
            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                dwpe_ref(dwpe[t * C + c]);
            dwpe_ref.fetch_add(v);
          }
        });
  });
}

// Variant 2: parallelize over C, loop over B*T serially; no atomics.
class EncoderBackwardKernel2;

void encoder_backward2(sycl::queue &q, float *dwte, float *dwpe,
                       const float *dout, const int *inp, int B, int T, int C,
                       int block_size) {
  if (C <= 0) {
    return;
  }
  const long BT = static_cast<long>(B) * static_cast<long>(T);
  const long local = static_cast<long>(block_size);
  const long n_groups = (static_cast<long>(C) + local - 1) / local;
  const long global = n_groups * local;
  q.submit([&](sycl::handler &h) {
    h.parallel_for<EncoderBackwardKernel2>(
        sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(global)),
                          sycl::range<1>(static_cast<size_t>(local))),
        [=](sycl::nd_item<1> it) {
          long c = static_cast<long>(it.get_global_id(0));
          if (c >= C) {
            return;
          }
          for (long i = 0; i < BT; ++i) {
            long t = i % T;
            int ix = inp[i];
            float v = dout[i * C + c];
            dwte[ix * C + c] += v;
            dwpe[t * C + c] += v;
          }
        });
  });
}

void encoder_backward(int kernel_num, sycl::queue &q, float *dwte, float *dwpe,
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

int main(int argc, char **argv) {
  if (argc != 7) {
    std::fprintf(stderr, "usage: %s <input> <output> B T C V\n", argv[0]);
    return 2;
  }
  const int B = std::atoi(argv[3]);
  const int T = std::atoi(argv[4]);
  const int C = std::atoi(argv[5]);
  const int V = std::atoi(argv[6]);

  const size_t n_dout = static_cast<size_t>(B) * static_cast<size_t>(T) * static_cast<size_t>(C);
  const size_t n_inp = static_cast<size_t>(B) * static_cast<size_t>(T);
  const size_t n_wte = static_cast<size_t>(V) * static_cast<size_t>(C);
  const size_t n_wpe = static_cast<size_t>(T) * static_cast<size_t>(C);

  FILE *fin = std::fopen(argv[1], "rb");
  if (!fin) {
    std::perror(argv[1]);
    return 2;
  }
  float *dout = static_cast<float *>(std::malloc(sizeof(float) * (n_dout ? n_dout : 1)));
  if (n_dout > 0) {
    if (std::fread(dout, sizeof(float), n_dout, fin) != n_dout) {
      std::fprintf(stderr, "input too short at dout\n");
      return 2;
    }
  }
  int *inp = static_cast<int *>(std::malloc(sizeof(int) * (n_inp ? n_inp : 1)));
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

  sycl::queue q;
  float *d_dout = sycl::malloc_device<float>(n_dout ? n_dout : 1, q);
  int *d_inp = sycl::malloc_device<int>(n_inp ? n_inp : 1, q);
  float *d_dwte = sycl::malloc_device<float>(n_wte ? n_wte : 1, q);
  float *d_dwpe = sycl::malloc_device<float>(n_wpe ? n_wpe : 1, q);
  float *dwte = static_cast<float *>(std::malloc(sizeof(float) * (n_wte ? n_wte : 1)));
  float *dwpe = static_cast<float *>(std::malloc(sizeof(float) * (n_wpe ? n_wpe : 1)));

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
      q.fill(d_dwte, 0.0f, n_wte).wait();
    }
    if (n_wpe > 0) {
      q.fill(d_dwpe, 0.0f, n_wpe).wait();
    }
    encoder_backward(k, q, d_dwte, d_dwpe, d_dout, d_inp, B, T, C, 256);
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
