// Idiomatic SYCL 2020 port of encoder_forward_harness.cu (fp32 build).
// Keeps kernel variants 1,2,3 as distinct device kernels.
#include <sycl/sycl.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

constexpr int kBlockSize = 256;

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

class EncoderKernel1;
class EncoderKernel2;
class EncoderKernel3;

void encoder_forward1(sycl::queue &q, float *out, const int *inp,
                      const float *wte, const float *wpe, int B, int T,
                      int C) {
  const int N = B * T;
  if (N <= 0) return;
  const int block = kBlockSize;
  const int64_t n_groups = ceil_div(N, block);
  const sycl::range<1> local(static_cast<std::size_t>(block));
  const sycl::range<1> global(static_cast<std::size_t>(n_groups * block));
  q.parallel_for<EncoderKernel1>(sycl::nd_range<1>(global, local),
                                 [=](sycl::nd_item<1> it) {
    const int idx = static_cast<int>(it.get_global_id(0));
    if (idx < N) {
      const int b = idx / T;
      const int t = idx % T;
      float *out_bt = out + static_cast<std::size_t>(b) * T * C + t * C;
      const int ix = inp[static_cast<std::size_t>(b) * T + t];
      const float *wte_ix = wte + static_cast<std::size_t>(ix) * C;
      const float *wpe_t = wpe + static_cast<std::size_t>(t) * C;
      for (int i = 0; i < C; ++i) {
        out_bt[i] = wte_ix[i] + wpe_t[i];
      }
    }
  });
}

void encoder_forward2(sycl::queue &q, float *out, const int *inp,
                      const float *wte, const float *wpe, int B, int T,
                      int C) {
  const int64_t N = static_cast<int64_t>(B) * T * C;
  if (N <= 0) return;
  const int block = kBlockSize;
  const int64_t n_groups = (N + block - 1) / block;
  const sycl::range<1> local(static_cast<std::size_t>(block));
  const sycl::range<1> global(static_cast<std::size_t>(n_groups * block));
  q.parallel_for<EncoderKernel2>(sycl::nd_range<1>(global, local),
                                 [=](sycl::nd_item<1> it) {
    const int64_t idx = static_cast<int64_t>(it.get_global_id(0));
    if (idx < N) {
      const int64_t bt = idx / C;
      const int b = static_cast<int>(bt / T);
      const int t = static_cast<int>(bt % T);
      const int c = static_cast<int>(idx % C);
      const int ix = inp[static_cast<std::size_t>(b) * T + t];
      out[static_cast<std::size_t>(b) * T * C + t * C + c] =
          wte[static_cast<std::size_t>(ix) * C + c] +
          wpe[static_cast<std::size_t>(t) * C + c];
    }
  });
}

void encoder_forward3(sycl::queue &q, float *out, const int *inp,
                      const float *wte, const float *wpe, int B, int T,
                      int C) {
  const int64_t N = static_cast<int64_t>(B) * T * C;
  if (N <= 0) return;
  // Fall back to scalar path when the embedding dim is not a multiple of 4,
  // so packed float4 accesses never straddle rows.
  if (C % 4 != 0) {
    encoder_forward2(q, out, inp, wte, wpe, B, T, C);
    return;
  }
  constexpr int kVec = 4;
  const int64_t n_vec = (N + kVec - 1) / kVec;
  const int block = kBlockSize;
  const int64_t n_groups = (n_vec + block - 1) / block;
  const sycl::range<1> local(static_cast<std::size_t>(block));
  const sycl::range<1> global(static_cast<std::size_t>(n_groups * block));
  const auto *wte4 = reinterpret_cast<const sycl::float4 *>(wte);
  const auto *wpe4 = reinterpret_cast<const sycl::float4 *>(wpe);
  auto *out4 = reinterpret_cast<sycl::float4 *>(out);
  q.parallel_for<EncoderKernel3>(sycl::nd_range<1>(global, local),
                                 [=](sycl::nd_item<1> it) {
    const int64_t v = static_cast<int64_t>(it.get_global_id(0));
    if (v < n_vec) {
      const int64_t idx = v * kVec;
      // Flat packed index; C % 4 == 0 guarantees the 4 elements share one row.
      const int64_t bt = idx / C;
      const int b = static_cast<int>(bt / T);
      const int t = static_cast<int>(bt % T);
      const int c = static_cast<int>(idx % C);
      const int c4 = c / kVec;
      const int ix = inp[static_cast<std::size_t>(b) * T + t];
      const sycl::float4 a =
          wte4[static_cast<std::size_t>(ix) * (C / kVec) + c4];
      const sycl::float4 b4 =
          wpe4[static_cast<std::size_t>(t) * (C / kVec) + c4];
      out4[static_cast<std::size_t>(b) * (T * (C / kVec)) + t * (C / kVec) +
           c4] = a + b4;
    }
  });
}

void encoder_forward(sycl::queue &q, int kernel_num, float *out,
                     const int *inp, const float *wte, const float *wpe, int B,
                     int T, int C) {
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
  int *inp = static_cast<int *>(std::malloc(sizeof(int) * (std::size_t)B * T));
  float *wte =
      static_cast<float *>(std::malloc(sizeof(float) * (std::size_t)V * C));
  float *wpe =
      static_cast<float *>(std::malloc(sizeof(float) * (std::size_t)T * C));
  if (!inp || !wte || !wpe) {
    std::fprintf(stderr, "host allocation failed\n");
    return 2;
  }
  if (std::fread(inp, sizeof(int), (std::size_t)B * T, fin) !=
      (std::size_t)B * T) {
    std::fprintf(stderr, "input too short at inp\n");
    return 2;
  }
  if (std::fread(wte, sizeof(float), (std::size_t)V * C, fin) !=
      (std::size_t)V * C) {
    std::fprintf(stderr, "input too short at wte\n");
    return 2;
  }
  if (std::fread(wpe, sizeof(float), (std::size_t)T * C, fin) !=
      (std::size_t)T * C) {
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
  int *d_inp = sycl::malloc_device<int>((std::size_t)B * T, q);
  float *d_wte = sycl::malloc_device<float>((std::size_t)V * C, q);
  float *d_wpe = sycl::malloc_device<float>((std::size_t)T * C, q);
  float *d_out =
      sycl::malloc_device<float>((std::size_t)B * T * C, q);
  float *out =
      static_cast<float *>(std::malloc(sizeof(float) * (std::size_t)B * T * C));
  if (!d_inp || !d_wte || !d_wpe || !d_out || !out) {
    std::fprintf(stderr, "allocation failed\n");
    return 2;
  }

  q.memcpy(d_inp, inp, sizeof(int) * (std::size_t)B * T).wait();
  q.memcpy(d_wte, wte, sizeof(float) * (std::size_t)V * C).wait();
  q.memcpy(d_wpe, wpe, sizeof(float) * (std::size_t)T * C).wait();

  const int variants[] = {1, 2, 3};
  for (int k : variants) {
    q.memset(d_out, 0, sizeof(float) * (std::size_t)B * T * C).wait();
    encoder_forward(q, k, d_out, d_inp, d_wte, d_wpe, B, T, C);
    q.wait();
    q.memcpy(out, d_out, sizeof(float) * (std::size_t)B * T * C).wait();
    std::fwrite(out, sizeof(float), (std::size_t)B * T * C, fout);
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
