#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>

constexpr int kBlockSize = 256;

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

class EncoderBackwardV1;
class EncoderBackwardV2;

void encoder_backward1(sycl::queue &q, float *dwte, float *dwpe,
                       const float *dout, const int *inp, int B, int T,
                       int C) {
  const int N = B * T * C;
  const int nblocks = ceil_div(N, kBlockSize);
  const std::size_t local = static_cast<std::size_t>(kBlockSize);
  const std::size_t global = static_cast<std::size_t>(nblocks) * local;
  q.parallel_for<EncoderBackwardV1>(
      sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(local)),
      [=](sycl::nd_item<1> it) {
        const int idx = static_cast<int>(it.get_global_id(0));
        if (idx >= N) {
          return;
        }
        const int bt = idx / C;
        const int b = bt / T;
        const int t = bt % T;
        const int c = idx % C;

        const int ix = inp[b * T + t];
        const float v = dout[b * T * C + t * C + c];

        sycl::atomic_ref<float, sycl::memory_order::relaxed,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            awte(dwte[ix * C + c]);
        awte.fetch_add(v);
        sycl::atomic_ref<float, sycl::memory_order::relaxed,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            awpe(dwpe[t * C + c]);
        awpe.fetch_add(v);
      });
}

void encoder_backward2(sycl::queue &q, float *dwte, float *dwpe,
                       const float *dout, const int *inp, int B, int T,
                       int C) {
  const int BT = B * T;
  const int nblocks = ceil_div(C, kBlockSize);
  const std::size_t local = static_cast<std::size_t>(kBlockSize);
  const std::size_t global = static_cast<std::size_t>(nblocks) * local;
  q.parallel_for<EncoderBackwardV2>(
      sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(local)),
      [=](sycl::nd_item<1> it) {
        const int c = static_cast<int>(it.get_global_id(0));
        if (c >= C) {
          return;
        }
        for (int i = 0; i < BT; ++i) {
          const int t = i % T;
          const int ix = inp[i];
          const float v = dout[i * C + c];
          dwte[ix * C + c] += v;
          dwpe[t * C + c] += v;
        }
      });
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

  FILE *fin = std::fopen(argv[1], "rb");
  if (!fin) {
    std::perror(argv[1]);
    return 2;
  }
  float *dout =
      (float *)std::malloc(sizeof(float) * (std::size_t)(B * T * C));
  if (std::fread(dout, sizeof(float), (std::size_t)(B * T * C), fin) !=
      (std::size_t)(B * T * C)) {
    std::fprintf(stderr, "input too short at dout\n");
    return 2;
  }
  int *inp = (int *)std::malloc(sizeof(int) * (std::size_t)(B * T));
  if (std::fread(inp, sizeof(int), (std::size_t)(B * T), fin) !=
      (std::size_t)(B * T)) {
    std::fprintf(stderr, "input too short at inp\n");
    return 2;
  }
  std::fclose(fin);
  FILE *fout = std::fopen(argv[2], "wb");
  if (!fout) {
    std::perror(argv[2]);
    return 2;
  }

  sycl::queue q;
  float *d_dout =
      sycl::malloc_device<float>((std::size_t)(B * T * C), q);
  q.memcpy(d_dout, dout, sizeof(float) * (std::size_t)(B * T * C)).wait();
  int *d_inp = sycl::malloc_device<int>((std::size_t)(B * T), q);
  q.memcpy(d_inp, inp, sizeof(int) * (std::size_t)(B * T)).wait();

  float *dwte = (float *)std::malloc(sizeof(float) * (std::size_t)(V * C));
  float *d_dwte = sycl::malloc_device<float>((std::size_t)(V * C), q);
  float *dwpe = (float *)std::malloc(sizeof(float) * (std::size_t)(T * C));
  float *d_dwpe = sycl::malloc_device<float>((std::size_t)(T * C), q);

  const int variants[] = {1, 2};
  for (int vi = 0; vi < 2; ++vi) {
    const int k = variants[vi];
    q.memset(d_dwte, 0, sizeof(float) * (std::size_t)(V * C)).wait();
    q.memset(d_dwpe, 0, sizeof(float) * (std::size_t)(T * C)).wait();
    encoder_backward(q, k, d_dwte, d_dwpe, d_dout, d_inp, B, T, C);
    q.wait();
    q.memcpy(dwte, d_dwte, sizeof(float) * (std::size_t)(V * C)).wait();
    std::fwrite(dwte, sizeof(float), (std::size_t)(V * C), fout);
    q.memcpy(dwpe, d_dwpe, sizeof(float) * (std::size_t)(T * C)).wait();
    std::fwrite(dwpe, sizeof(float), (std::size_t)(T * C), fout);
  }

  std::fclose(fout);
  sycl::free(d_dout, q);
  sycl::free(d_inp, q);
  sycl::free(d_dwte, q);
  sycl::free(d_dwpe, q);
  std::free(dout);
  std::free(inp);
  return 0;
}
