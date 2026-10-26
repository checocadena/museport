/* AdamW optimizer: idiomatic SYCL 2020 port of adamw_harness.cu (fp32).
 * Two kernel variants, each its own SYCL kernel, computed on the SYCL device.
 * Contract: build/candidate <input> <output> N
 */
#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstddef>

// Linear interpolation with two flops: weight*end + (start - weight*start).
inline float lerp(float start, float end, float weight) {
  return sycl::fma(weight, end, sycl::fma(-weight, start, start));
}

class AdamwKernel1;
class AdamwKernel2;

namespace {

constexpr std::size_t kWorkGroupSize = 256;

void adamw_dispatch1(sycl::queue &q, float *params, const float *grads,
                     float *m, float *v, std::size_t n, float lr, float beta1,
                     float beta2, float b1_corr, float b2_corr, float eps,
                     float wd) {
  if (n == 0) return;
  const std::size_t g =
      ((n + kWorkGroupSize - 1) / kWorkGroupSize) * kWorkGroupSize;
  q.parallel_for<AdamwKernel1>(
       sycl::nd_range<1>(sycl::range<1>(g), sycl::range<1>(kWorkGroupSize)),
       [=](sycl::nd_item<1> it) {
         const std::size_t i = it.get_global_id(0);
         if (i >= n) return;
         const float gval = grads[i];
         float mval = beta1 * m[i] + (1.0f - beta1) * gval;
         m[i] = mval;
         float vval = beta2 * v[i] + (1.0f - beta2) * gval * gval;
         v[i] = vval;
         const float m_hat = mval / b1_corr;
         const float v_hat = vval / b2_corr;
         params[i] -= lr * (m_hat / (sycl::sqrt(v_hat) + eps) + wd * params[i]);
       })
      .wait();
}

void adamw_dispatch2(sycl::queue &q, float *params, const float *grads,
                     float *m, float *v, std::size_t n, float lr, float beta1,
                     float beta2, float b1_corr, float b2_corr, float eps,
                     float wd) {
  if (n == 0) return;
  const std::size_t g =
      ((n + kWorkGroupSize - 1) / kWorkGroupSize) * kWorkGroupSize;
  q.parallel_for<AdamwKernel2>(
       sycl::nd_range<1>(sycl::range<1>(g), sycl::range<1>(kWorkGroupSize)),
       [=](sycl::nd_item<1> it) {
         const std::size_t i = it.get_global_id(0);
         if (i >= n) return;
         const float grad = grads[i];
         float mval = m[i];
         float vval = v[i];
         mval = lerp(grad, mval, beta1);
         m[i] = mval;
         vval = lerp(grad * grad, vval, beta2);
         v[i] = vval;
         mval /= b1_corr;
         vval /= b2_corr;
         params[i] -=
             lr * (mval / (sycl::sqrt(vval) + eps) + wd * params[i]);
       })
      .wait();
}

void adamw_run(sycl::queue &q, int kernel_num, float *params,
               const float *grads, float *m, float *v, int t, std::size_t n,
               float lr = 1e-3f, float beta1 = 0.9f, float beta2 = 0.999f,
               float eps = 1e-8f, float wd = 0.0f) {
  const float b1_corr = 1.0f - std::pow(beta1, t);
  const float b2_corr = 1.0f - std::pow(beta2, t);
  switch (kernel_num) {
    case 1:
      adamw_dispatch1(q, params, grads, m, v, n, lr, beta1, beta2, b1_corr,
                      b2_corr, eps, wd);
      break;
    case 2:
      adamw_dispatch2(q, params, grads, m, v, n, lr, beta1, beta2, b1_corr,
                      b2_corr, eps, wd);
      break;
    default:
      std::fprintf(stderr, "Invalid kernel number\n");
      std::exit(1);
  }
}

}  // namespace

int main(int argc, char **argv) {
  std::srand(0);
  if (argc != 4) {
    std::fprintf(stderr, "usage: %s <input> <output> N\n", argv[0]);
    return 2;
  }
  const std::size_t N = static_cast<std::size_t>(std::atoi(argv[3]));

  FILE *fin = std::fopen(argv[1], "rb");
  if (!fin) {
    std::perror(argv[1]);
    return 2;
  }
  float *params = static_cast<float *>(std::malloc(sizeof(float) * N));
  float *grads = static_cast<float *>(std::malloc(sizeof(float) * N));
  float *m = static_cast<float *>(std::malloc(sizeof(float) * N));
  float *v = static_cast<float *>(std::malloc(sizeof(float) * N));
  if (!params || !grads || !m || !v) {
    std::fprintf(stderr, "host alloc failed\n");
    return 2;
  }
  auto read_array = [&](float *dst, const char *what) {
    if (N == 0) return true;
    if (std::fread(dst, sizeof(float), N, fin) != N) {
      std::fprintf(stderr, "input too short at %s\n", what);
      return false;
    }
    return true;
  };
  if (!read_array(params, "params") || !read_array(grads, "grads") ||
      !read_array(m, "m") || !read_array(v, "v")) {
    return 2;
  }
  std::fclose(fin);

  FILE *fout = std::fopen(argv[2], "wb");
  if (!fout) {
    std::perror(argv[2]);
    return 2;
  }

  float *params_out = static_cast<float *>(std::malloc(sizeof(float) * N));
  float *m_out = static_cast<float *>(std::malloc(sizeof(float) * N));
  float *v_out = static_cast<float *>(std::malloc(sizeof(float) * N));
  if (!params_out || !m_out || !v_out) {
    std::fprintf(stderr, "host alloc failed\n");
    return 2;
  }

  try {
    sycl::queue q{sycl::default_selector_v};

    float *d_grads = sycl::malloc_device<float>(N == 0 ? 1 : N, q);
    float *d_params = sycl::malloc_device<float>(N == 0 ? 1 : N, q);
    float *d_m = sycl::malloc_device<float>(N == 0 ? 1 : N, q);
    float *d_v = sycl::malloc_device<float>(N == 0 ? 1 : N, q);
    if (!d_grads || !d_params || !d_m || !d_v) {
      std::fprintf(stderr, "device alloc failed\n");
      return 2;
    }
    if (N > 0) {
      q.memcpy(d_grads, grads, sizeof(float) * N).wait();
    }

    const int variants[] = {1, 2};
    for (int vi = 0; vi < 2; ++vi) {
      const int k = variants[vi];
      if (N > 0) {
        q.memcpy(d_params, params, sizeof(float) * N).wait();
        q.memcpy(d_m, m, sizeof(float) * N).wait();
        q.memcpy(d_v, v, sizeof(float) * N).wait();
      }
      adamw_run(q, k, d_params, d_grads, d_m, d_v, 10, N);
      if (N > 0) {
        q.memcpy(params_out, d_params, sizeof(float) * N).wait();
        std::fwrite(params_out, sizeof(float), N, fout);
        q.memcpy(m_out, d_m, sizeof(float) * N).wait();
        std::fwrite(m_out, sizeof(float), N, fout);
        q.memcpy(v_out, d_v, sizeof(float) * N).wait();
        std::fwrite(v_out, sizeof(float), N, fout);
      }
    }

    sycl::free(d_grads, q);
    sycl::free(d_params, q);
    sycl::free(d_m, q);
    sycl::free(d_v, q);
  } catch (const sycl::exception &e) {
    std::fprintf(stderr, "SYCL error: %s\n", e.what());
    return 1;
  }

  std::fclose(fout);
  std::free(params);
  std::free(grads);
  std::free(m);
  std::free(v);
  std::free(params_out);
  std::free(m_out);
  std::free(v_out);
  return 0;
}
