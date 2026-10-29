#include <sycl/sycl.hpp>

#include <cstdio>
#include <cstdlib>

// fp32 build: floatX is float
using floatX = float;

class EncoderKernel1;
class EncoderKernel2;
class EncoderKernel3;

// Variant 1: parallelize over B*T, loop over C
void encoder_forward1(sycl::queue &q, floatX *out, const int *inp,
                      const floatX *wte, const floatX *wpe, int B, int T,
                      int C) {
  const std::size_t N = static_cast<std::size_t>(B) * static_cast<std::size_t>(T);
  if (N == 0) {
    return;
  }
  const std::size_t UsT = static_cast<std::size_t>(T);
  const std::size_t UsC = static_cast<std::size_t>(C);
  q.parallel_for<EncoderKernel1>(sycl::range<1>(N), [=](sycl::id<1> id) {
     std::size_t idx = id[0];
     std::size_t b = idx / UsT;
     std::size_t t = idx % UsT;
     floatX *out_bt = out + (b * UsT + t) * UsC;
     int ix = inp[b * UsT + t];
     const floatX *wte_ix = wte + static_cast<std::size_t>(ix) * UsC;
     const floatX *wpe_t = wpe + t * UsC;
     for (std::size_t i = 0; i < UsC; ++i) {
       out_bt[i] = wte_ix[i] + wpe_t[i];
     }
   }).wait();
}

// Variant 2: parallelize over all of B*T*C
void encoder_forward2(sycl::queue &q, floatX *out, const int *inp,
                      const floatX *wte, const floatX *wpe, int B, int T,
                      int C) {
  const std::size_t N = static_cast<std::size_t>(B) *
                        static_cast<std::size_t>(T) *
                        static_cast<std::size_t>(C);
  if (N == 0) {
    return;
  }
  const std::size_t UsT = static_cast<std::size_t>(T);
  const std::size_t UsC = static_cast<std::size_t>(C);
  q.parallel_for<EncoderKernel2>(sycl::range<1>(N), [=](sycl::id<1> id) {
     std::size_t idx = id[0];
     std::size_t bt = idx / UsC;
     std::size_t b = bt / UsT;
     std::size_t t = bt % UsT;
     std::size_t c = idx % UsC;
     int ix = inp[b * UsT + t];
     out[idx] = wte[static_cast<std::size_t>(ix) * UsC + c] + wpe[t * UsC + c];
   }).wait();
}

// Variant 3: like variant 2 but uses vector (float4) reads/writes.
// One work-item handles one group of 4 consecutive elements, exactly like the
// CUDA kernel (idx = tid * 4). Full in-row groups use sycl::vec<float, 4>
// arithmetic; a partial tail group (only possible when C % 4 != 0) falls back
// to scalar stores so no out-of-bounds access occurs.
void encoder_forward3(sycl::queue &q, floatX *out, const int *inp,
                      const floatX *wte, const floatX *wpe, int B, int T,
                      int C) {
  const std::size_t UsT = static_cast<std::size_t>(T);
  const std::size_t UsC = static_cast<std::size_t>(C);
  const std::size_t N = static_cast<std::size_t>(B) * UsT * UsC;
  if (N == 0) {
    return;
  }
  const std::size_t Nvec = (N + 3) / 4;
  q.parallel_for<EncoderKernel3>(sycl::range<1>(Nvec), [=](sycl::id<1> id) {
     std::size_t v = id[0];
     std::size_t idx = v * 4;
     std::size_t bt = idx / UsC;
     std::size_t b = bt / UsT;
     std::size_t t = bt % UsT;
     std::size_t c = idx % UsC;
     int ix = inp[b * UsT + t];
     const floatX *wte_row = wte + static_cast<std::size_t>(ix) * UsC;
     const floatX *wpe_row = wpe + t * UsC;
     floatX *out_row = out + bt * UsC;
     if (c + 4 <= UsC) {
       sycl::vec<float, 4> a(wte_row[c], wte_row[c + 1], wte_row[c + 2],
                             wte_row[c + 3]);
       sycl::vec<float, 4> bvec(wpe_row[c], wpe_row[c + 1], wpe_row[c + 2],
                                wpe_row[c + 3]);
       sycl::vec<float, 4> o = a + bvec;
       out_row[c] = o[0];
       out_row[c + 1] = o[1];
       out_row[c + 2] = o[2];
       out_row[c + 3] = o[3];
     } else {
       for (int k = 0; k < 4 && c + (std::size_t)k < UsC; ++k) {
         out_row[c + (std::size_t)k] =
             wte_row[c + (std::size_t)k] + wpe_row[c + (std::size_t)k];
       }
     }
   }).wait();
}

void encoder_forward(int kernel_num, sycl::queue &q, floatX *out,
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
  int *inp = (int *)std::malloc(sizeof(int) * (std::size_t)(B * T));
  if (std::fread(inp, sizeof(int), (std::size_t)(B * T), fin) !=
      (std::size_t)(B * T)) {
    std::fprintf(stderr, "input too short at inp\n");
    return 2;
  }
  float *wte = (float *)std::malloc(sizeof(float) * (std::size_t)(V * C));
  if (std::fread(wte, sizeof(float), (std::size_t)(V * C), fin) !=
      (std::size_t)(V * C)) {
    std::fprintf(stderr, "input too short at wte\n");
    return 2;
  }
  float *wpe = (float *)std::malloc(sizeof(float) * (std::size_t)(T * C));
  if (std::fread(wpe, sizeof(float), (std::size_t)(T * C), fin) !=
      (std::size_t)(T * C)) {
    std::fprintf(stderr, "input too short at wpe\n");
    return 2;
  }
  std::fclose(fin);

  FILE *fout = std::fopen(argv[2], "wb");
  if (!fout) {
    std::perror(argv[2]);
    return 2;
  }

  sycl::queue q{sycl::default_selector_v};

  int *d_inp = sycl::malloc_device<int>((std::size_t)(B * T), q);
  float *d_wte = sycl::malloc_device<float>((std::size_t)(V * C), q);
  float *d_wpe = sycl::malloc_device<float>((std::size_t)(T * C), q);
  float *d_out = sycl::malloc_device<float>((std::size_t)(B * T * C), q);

  q.memcpy(d_inp, inp, sizeof(int) * (std::size_t)(B * T)).wait();
  q.memcpy(d_wte, wte, sizeof(float) * (std::size_t)(V * C)).wait();
  q.memcpy(d_wpe, wpe, sizeof(float) * (std::size_t)(T * C)).wait();

  float *out = (float *)std::malloc(sizeof(float) * (std::size_t)(B * T * C));

  const int variants[] = {1, 2, 3};
  for (int vi = 0; vi < 3; ++vi) {
    const int k = variants[vi];
    q.memset(d_out, 0, sizeof(float) * (std::size_t)(B * T * C)).wait();
    encoder_forward(k, q, d_out, d_inp, d_wte, d_wpe, B, T, C);
    q.wait();
    q.memcpy(out, d_out, sizeof(float) * (std::size_t)(B * T * C)).wait();
    std::fwrite(out, sizeof(float), (std::size_t)(B * T * C), fout);
  }
  std::fclose(fout);

  sycl::free(d_inp, q);
  sycl::free(d_wte, q);
  sycl::free(d_wpe, q);
  sycl::free(d_out, q);
  std::free(inp);
  std::free(wte);
  std::free(wpe);
  std::free(out);
  return 0;
}
