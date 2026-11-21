// Idiomatic SYCL 2020 port of encoder_backward_harness.cu (fp32).
// Two kernel variants, each its own SYCL kernel, computed on the device.
#include <sycl/sycl.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <vector>

namespace {

constexpr int kBlockSize = 256;

inline int ceil_div(int n, int d) { return (n + d - 1) / d; }

class EncoderBackwardKernel1;
class EncoderBackwardKernel2;

void encoder_backward1(sycl::queue &q, float *dwte, float *dwpe,
                       const float *dout, const int *inp, int B, int T, int C) {
  const int N = B * T * C;
  if (N <= 0) return;
  const int groups = ceil_div(N, kBlockSize);
  const std::size_t global = static_cast<std::size_t>(groups) * kBlockSize;
  q.parallel_for<EncoderBackwardKernel1>(
       sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(kBlockSize)),
       [=](sycl::nd_item<1> it) {
         const int idx = static_cast<int>(it.get_global_id(0));
         if (idx >= N) return;
         const int bt = idx / C;
         const int b = bt / T;
         const int t = bt % T;
         const int c = idx % C;
         const int ix = inp[b * T + t];
         const float v = dout[b * T * C + t * C + c];
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
       })
      .wait();
}

void encoder_backward2(sycl::queue &q, float *dwte, float *dwpe,
                       const float *dout, const int *inp, int B, int T, int C) {
  if (C <= 0) return;
  const int BT = B * T;
  const int groups = ceil_div(C, kBlockSize);
  const std::size_t global = static_cast<std::size_t>(groups) * kBlockSize;
  q.parallel_for<EncoderBackwardKernel2>(
       sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(kBlockSize)),
       [=](sycl::nd_item<1> it) {
         const int c = static_cast<int>(it.get_global_id(0));
         if (c >= C) return;
         for (int i = 0; i < BT; ++i) {
           const int t = i % T;
           const int ix = inp[i];
           const float v = dout[i * C + c];
           dwte[ix * C + c] += v;
           dwpe[t * C + c] += v;
         }
       })
      .wait();
}

void encoder_backward(sycl::queue &q, int kernel_num, float *dwte, float *dwpe,
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

  FILE *fin = std::fopen(argv[1], "rb");
  if (!fin) {
    std::perror(argv[1]);
    return 2;
  }
  std::vector<float> dout(static_cast<std::size_t>(B) * T * C);
  if (std::fread(dout.data(), sizeof(float),
                 static_cast<std::size_t>(B) * T * C,
                 fin) != static_cast<std::size_t>(B) * T * C) {
    std::fprintf(stderr, "input too short at dout\n");
    return 2;
  }
  std::vector<int> inp(static_cast<std::size_t>(B) * T);
  if (std::fread(inp.data(), sizeof(int), static_cast<std::size_t>(B) * T,
                 fin) != static_cast<std::size_t>(B) * T) {
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
    sycl::queue q{sycl::default_selector_v,
                  sycl::property::queue::in_order{}};

    const std::size_t n_dout = static_cast<std::size_t>(B) * T * C;
    const std::size_t n_inp = static_cast<std::size_t>(B) * T;
    const std::size_t n_wte = static_cast<std::size_t>(V) * C;
    const std::size_t n_wpe = static_cast<std::size_t>(T) * C;

    float *d_dout = sycl::malloc_device<float>(n_dout, q);
    int *d_inp = sycl::malloc_device<int>(n_inp, q);
    float *d_dwte = sycl::malloc_device<float>(n_wte, q);
    float *d_dwpe = sycl::malloc_device<float>(n_wpe, q);
    if (!d_dout || !d_inp || !d_dwte || !d_dwpe) {
      std::fprintf(stderr, "device allocation failed\n");
      return 2;
    }

    q.memcpy(d_dout, dout.data(), n_dout * sizeof(float)).wait();
    q.memcpy(d_inp, inp.data(), n_inp * sizeof(int)).wait();

    std::vector<float> dwte(n_wte);
    std::vector<float> dwpe(n_wpe);

    const int variants[] = {1, 2};
    for (int vi = 0; vi < 2; ++vi) {
      const int k = variants[vi];
      q.memset(d_dwte, 0, n_wte * sizeof(float)).wait();
      q.memset(d_dwpe, 0, n_wpe * sizeof(float)).wait();
      encoder_backward(q, k, d_dwte, d_dwpe, d_dout, d_inp, B, T, C);
      q.wait();
      q.memcpy(dwte.data(), d_dwte, n_wte * sizeof(float)).wait();
      std::fwrite(dwte.data(), sizeof(float), n_wte, fout);
      q.memcpy(dwpe.data(), d_dwpe, n_wpe * sizeof(float)).wait();
      std::fwrite(dwpe.data(), sizeof(float), n_wpe, fout);
    }

    sycl::free(d_dout, q);
    sycl::free(d_inp, q);
    sycl::free(d_dwte, q);
    sycl::free(d_dwpe, q);
  } catch (const sycl::exception &e) {
    std::fprintf(stderr, "SYCL error: %s\n", e.what());
    return 2;
  }

  std::fclose(fout);
  return 0;
}
