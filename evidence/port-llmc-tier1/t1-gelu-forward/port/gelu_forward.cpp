// Idiomatic SYCL 2020 port of gelu_forward (fp32 build).
// Two kernel variants, each its own SYCL kernel, computed on the SYCL device.
// Contract: build/candidate <input> <output> B T C reads B*T*C float32 values
// and appends variant 1 then variant 2 outputs as raw float32.
#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstddef>

// sqrt(2/pi) for the tanh approximation of GELU.
inline constexpr float kGeluScalingFactor = 0.7978845608028654f;

class GeluKernel1;
class GeluKernel2;

void gelu_forward1(sycl::queue &q, float *out, const float *inp, std::size_t n) {
  if (n == 0) {
    return;
  }
  q.parallel_for<GeluKernel1>(sycl::range<1>(n), [=](sycl::id<1> idx) {
     std::size_t i = idx[0];
     float xi = inp[i];
     float cube = 0.044715f * xi * xi * xi;
     out[i] = 0.5f * xi * (1.0f + sycl::tanh(kGeluScalingFactor * (xi + cube)));
   }).wait();
}

void gelu_forward2(sycl::queue &q, float *out, const float *inp, std::size_t n) {
  if (n == 0) {
    return;
  }
  constexpr std::size_t kWidth = 4;
  const std::size_t nvec = (n + kWidth - 1) / kWidth;
  q.parallel_for<GeluKernel2>(sycl::range<1>(nvec), [=](sycl::id<1> idx) {
     std::size_t v = idx[0];
     std::size_t base = v * kWidth;
     if (base + kWidth <= n) {
       // Vectorized path: one work-item handles 4 elements via sycl::vec.
       sycl::vec<float, 4> vin(inp[base], inp[base + 1], inp[base + 2],
                               inp[base + 3]);
       float x0 = vin.x();
       float x1 = vin.y();
       float x2 = vin.z();
       float x3 = vin.w();
       float c0 = 0.044715f * x0 * x0 * x0;
       float c1 = 0.044715f * x1 * x1 * x1;
       float c2 = 0.044715f * x2 * x2 * x2;
       float c3 = 0.044715f * x3 * x3 * x3;
       float y0 = 0.5f * x0 * (1.0f + sycl::tanh(kGeluScalingFactor * (x0 + c0)));
       float y1 = 0.5f * x1 * (1.0f + sycl::tanh(kGeluScalingFactor * (x1 + c1)));
       float y2 = 0.5f * x2 * (1.0f + sycl::tanh(kGeluScalingFactor * (x2 + c2)));
       float y3 = 0.5f * x3 * (1.0f + sycl::tanh(kGeluScalingFactor * (x3 + c3)));
       out[base] = y0;
       out[base + 1] = y1;
       out[base + 2] = y2;
       out[base + 3] = y3;
     } else {
       for (std::size_t k = 0; k < kWidth; ++k) {
         std::size_t i = base + k;
         if (i < n) {
           float xi = inp[i];
           float cube = 0.044715f * xi * xi * xi;
           out[i] =
               0.5f * xi * (1.0f + sycl::tanh(kGeluScalingFactor * (xi + cube)));
         }
       }
     }
   }).wait();
}

void gelu_forward(sycl::queue &q, int kernel_num, float *out, const float *inp,
                  std::size_t n) {
  switch (kernel_num) {
    case 1:
      gelu_forward1(q, out, inp, n);
      break;
    case 2:
      gelu_forward2(q, out, inp, n);
      break;
    default:
      std::fprintf(stderr, "Invalid kernel number\n");
      std::exit(1);
  }
}

int main(int argc, char **argv) {
  if (argc != 6) {
    std::fprintf(stderr, "usage: %s <input> <output> B T C\n", argv[0]);
    return 2;
  }
  const std::size_t b = static_cast<std::size_t>(std::atoi(argv[3]));
  const std::size_t t = static_cast<std::size_t>(std::atoi(argv[4]));
  const std::size_t c = static_cast<std::size_t>(std::atoi(argv[5]));
  const std::size_t n = b * t * c;

  FILE *fin = std::fopen(argv[1], "rb");
  if (fin == nullptr) {
    std::perror(argv[1]);
    return 2;
  }
  float *inp = static_cast<float *>(std::malloc(sizeof(float) * n));
  if (inp == nullptr && n != 0) {
    std::fprintf(stderr, "host malloc failed\n");
    std::fclose(fin);
    return 2;
  }
  if (n > 0) {
    if (std::fread(inp, sizeof(float), n, fin) != n) {
      std::fprintf(stderr, "input too short at inp\n");
      std::fclose(fin);
      return 2;
    }
  }
  std::fclose(fin);

  FILE *fout = std::fopen(argv[2], "wb");
  if (fout == nullptr) {
    std::perror(argv[2]);
    return 2;
  }

  sycl::queue q;
  float *d_inp = sycl::malloc_device<float>(n, q);
  float *d_out = sycl::malloc_device<float>(n, q);
  float *out = static_cast<float *>(std::malloc(sizeof(float) * n));
  if ((d_inp == nullptr || d_out == nullptr || out == nullptr) && n != 0) {
    std::fprintf(stderr, "allocation failed\n");
    return 2;
  }

  if (n > 0) {
    q.memcpy(d_inp, inp, sizeof(float) * n).wait();
  }
  const int variants[] = {1, 2};
  for (int vi = 0; vi < 2; ++vi) {
    const int k = variants[vi];
    if (n > 0) {
      q.memset(d_out, 0, sizeof(float) * n).wait();
    }
    gelu_forward(q, k, d_out, d_inp, n);
    q.wait();
    if (n > 0) {
      q.memcpy(out, d_out, sizeof(float) * n).wait();
      std::fwrite(out, sizeof(float), n, fout);
    }
  }

  sycl::free(d_inp, q);
  sycl::free(d_out, q);
  std::free(inp);
  std::free(out);
  std::fclose(fout);
  return 0;
}
