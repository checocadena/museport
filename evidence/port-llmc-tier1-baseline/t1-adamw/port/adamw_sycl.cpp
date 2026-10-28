#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <sycl/sycl.hpp>

inline float adamw_lerp(float start, float end, float weight) {
  return sycl::fma(weight, end, sycl::fma(-weight, start, start));
}

class AdamwKernel1;
class AdamwKernel2;

static std::size_t ceil_div(std::size_t n, std::size_t d) {
  return (n + d - 1) / d;
}

void adamw_dispatch1(sycl::queue &q, float *params, const float *grads,
                     float *m_mem, float *v_mem, std::size_t n,
                     float learning_rate, float beta1, float beta2,
                     float beta1_correction, float beta2_correction, float eps,
                     float weight_decay) {
  if (n == 0) return;
  constexpr std::size_t block_size = 256;
  const std::size_t num_blocks = ceil_div(n, block_size);
  const std::size_t global = num_blocks * block_size;
  q.parallel_for<AdamwKernel1>(
       sycl::nd_range<1>{sycl::range<1>{global}, sycl::range<1>{block_size}},
       [=](sycl::nd_item<1> item) {
         const std::size_t i = item.get_global_id(0);
         if (i >= n) return;
         m_mem[i] = beta1 * m_mem[i] + (1.0f - beta1) * grads[i];
         v_mem[i] =
             beta2 * v_mem[i] + (1.0f - beta2) * grads[i] * grads[i];
         const float m_hat = m_mem[i] / beta1_correction;
         const float v_hat = v_mem[i] / beta2_correction;
         params[i] -= learning_rate *
                      (m_hat / (sycl::sqrt(v_hat) + eps) +
                       weight_decay * params[i]);
       })
      .wait();
}

void adamw_dispatch2(sycl::queue &q, float *params, const float *grads,
                     float *m_mem, float *v_mem, std::size_t n,
                     float learning_rate, float beta1, float beta2,
                     float beta1_correction, float beta2_correction, float eps,
                     float weight_decay) {
  if (n == 0) return;
  constexpr std::size_t block_size = 256;
  const std::size_t num_blocks = ceil_div(n, block_size);
  const std::size_t global = num_blocks * block_size;
  q.parallel_for<AdamwKernel2>(
       sycl::nd_range<1>{sycl::range<1>{global}, sycl::range<1>{block_size}},
       [=](sycl::nd_item<1> item) {
         const std::size_t i = item.get_global_id(0);
         if (i >= n) return;
         const float grad = grads[i];
         float m = m_mem[i];
         float v = v_mem[i];
         m = adamw_lerp(grad, m, beta1);
         m_mem[i] = m;
         v = adamw_lerp(grad * grad, v, beta2);
         v_mem[i] = v;
         m /= beta1_correction;
         v /= beta2_correction;
         params[i] -=
             learning_rate *
             (m / (sycl::sqrt(v) + eps) + weight_decay * params[i]);
       })
      .wait();
}

void adamw(sycl::queue &q, int kernel_num, float *params, const float *grads,
           float *m_mem, float *v_mem, int t, std::size_t n,
           float learning_rate = 1e-3f, float beta1 = 0.9f,
           float beta2 = 0.999f, float eps = 1e-8f,
           float weight_decay = 0.0f) {
  const float beta1_correction = 1.0f - powf(beta1, (float)t);
  const float beta2_correction = 1.0f - powf(beta2, (float)t);
  switch (kernel_num) {
    case 1:
      adamw_dispatch1(q, params, grads, m_mem, v_mem, n, learning_rate, beta1,
                      beta2, beta1_correction, beta2_correction, eps,
                      weight_decay);
      break;
    case 2:
      adamw_dispatch2(q, params, grads, m_mem, v_mem, n, learning_rate, beta1,
                      beta2, beta1_correction, beta2_correction, eps,
                      weight_decay);
      break;
    default:
      std::printf("Invalid kernel number\n");
      std::exit(1);
  }
}

int main(int argc, char **argv) {
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
  auto read_array = [&](float *dst, const char *name) {
    if (N == 0) return true;
    if (std::fread(dst, sizeof(float), N, fin) != N) {
      std::fprintf(stderr, "input too short at %s\n", name);
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

  sycl::queue q{sycl::default_selector_v};
  float *d_grads = N ? sycl::malloc_device<float>(N, q) : nullptr;
  float *d_params_out = N ? sycl::malloc_device<float>(N, q) : nullptr;
  float *d_m_out = N ? sycl::malloc_device<float>(N, q) : nullptr;
  float *d_v_out = N ? sycl::malloc_device<float>(N, q) : nullptr;
  float *params_out = static_cast<float *>(std::malloc(sizeof(float) * N));
  float *m_out = static_cast<float *>(std::malloc(sizeof(float) * N));
  float *v_out = static_cast<float *>(std::malloc(sizeof(float) * N));

  if (N > 0) {
    q.memcpy(d_grads, grads, sizeof(float) * N).wait();
  }

  const int variants[] = {1, 2};
  for (int vi = 0; vi < 2; ++vi) {
    const int k = variants[vi];
    if (N > 0) {
      q.memcpy(d_params_out, params, sizeof(float) * N).wait();
      q.memcpy(d_m_out, m, sizeof(float) * N).wait();
      q.memcpy(d_v_out, v, sizeof(float) * N).wait();
    }
    adamw(q, k, d_params_out, d_grads, d_m_out, d_v_out, 10, N);
    q.wait();
    if (N > 0) {
      q.memcpy(params_out, d_params_out, sizeof(float) * N).wait();
    }
    std::fwrite(params_out, sizeof(float), N, fout);
    if (N > 0) {
      q.memcpy(m_out, d_m_out, sizeof(float) * N).wait();
    }
    std::fwrite(m_out, sizeof(float), N, fout);
    if (N > 0) {
      q.memcpy(v_out, d_v_out, sizeof(float) * N).wait();
    }
    std::fwrite(v_out, sizeof(float), N, fout);
  }
  std::fclose(fout);

  if (d_grads) sycl::free(d_grads, q);
  if (d_params_out) sycl::free(d_params_out, q);
  if (d_m_out) sycl::free(d_m_out, q);
  if (d_v_out) sycl::free(d_v_out, q);
  std::free(params);
  std::free(grads);
  std::free(m);
  std::free(v);
  std::free(params_out);
  std::free(m_out);
  std::free(v_out);
  return 0;
}
