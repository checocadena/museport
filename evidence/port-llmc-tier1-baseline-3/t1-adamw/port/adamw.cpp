// AdamW optimizer kernels ported to idiomatic SYCL 2020 (fp32).
#include <sycl/sycl.hpp>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>

namespace {

inline long ceil_div(long a, long b) { return (a + b - 1) / b; }

inline float lerp_f(float start, float end, float weight) {
  return sycl::fma(weight, end, sycl::fma(-weight, start, start));
}

class AdamwKernel1;
class AdamwKernel2;

void adamw_dispatch1(sycl::queue &q, float *params, const float *grads,
                     float *m, float *v, long num_parameters,
                     float learning_rate, float beta1, float beta2,
                     float beta1_correction, float beta2_correction, float eps,
                     float weight_decay) {
  if (num_parameters <= 0) return;
  constexpr long block_size = 256;
  const long num_blocks = ceil_div(num_parameters, block_size);
  const sycl::range<1> local(static_cast<std::size_t>(block_size));
  const sycl::range<1> global(
      static_cast<std::size_t>(num_blocks * block_size));
  q.submit([&](sycl::handler &h) {
     h.parallel_for<AdamwKernel1>(
         sycl::nd_range<1>(global, local),
         [=](sycl::nd_item<1> item) {
           const long i = static_cast<long>(item.get_global_id(0));
           if (i >= num_parameters) return;
           // update the first moment (momentum)
           m[i] = beta1 * m[i] + (1.0f - beta1) * grads[i];
           // update the second moment (RMSprop)
           v[i] = beta2 * v[i] + (1.0f - beta2) * grads[i] * grads[i];
           const float m_hat = m[i] / beta1_correction;
           const float v_hat = v[i] / beta2_correction;
           params[i] -= learning_rate *
                        (m_hat / (sycl::sqrt(v_hat) + eps) +
                         weight_decay * params[i]);
         });
   });
}

void adamw_dispatch2(sycl::queue &q, float *params, const float *grads,
                     float *m, float *v, long num_parameters,
                     float learning_rate, float beta1, float beta2,
                     float beta1_correction, float beta2_correction, float eps,
                     float weight_decay) {
  if (num_parameters <= 0) return;
  constexpr long block_size = 256;
  const long num_blocks = ceil_div(num_parameters, block_size);
  const sycl::range<1> local(static_cast<std::size_t>(block_size));
  const sycl::range<1> global(
      static_cast<std::size_t>(num_blocks * block_size));
  q.submit([&](sycl::handler &h) {
     h.parallel_for<AdamwKernel2>(
         sycl::nd_range<1>(global, local),
         [=](sycl::nd_item<1> item) {
           const long i = static_cast<long>(item.get_global_id(0));
           if (i >= num_parameters) return;
           const float grad = grads[i];
           float mm = m[i];
           float vv = v[i];
           // update the first moment (momentum)
           mm = lerp_f(grad, mm, beta1);
           m[i] = mm;
           // update the second moment (RMSprop)
           vv = lerp_f(grad * grad, vv, beta2);
           v[i] = vv;
           mm /= beta1_correction; // m_hat
           vv /= beta2_correction; // v_hat
           params[i] -= learning_rate * (mm / (sycl::sqrt(vv) + eps) +
                                         weight_decay * params[i]);
         });
   });
}

void adamw(sycl::queue &q, int kernel_num, float *params, const float *grads,
           float *m, float *v, int t, long num_parameters,
           float learning_rate = 1e-3f, float beta1 = 0.9f,
           float beta2 = 0.999f, float eps = 1e-8f,
           float weight_decay = 0.0f) {
  const float beta1_correction = 1.0f - powf(beta1, static_cast<float>(t));
  const float beta2_correction = 1.0f - powf(beta2, static_cast<float>(t));
  switch (kernel_num) {
    case 1:
      adamw_dispatch1(q, params, grads, m, v, num_parameters, learning_rate,
                      beta1, beta2, beta1_correction, beta2_correction, eps,
                      weight_decay);
      break;
    case 2:
      adamw_dispatch2(q, params, grads, m, v, num_parameters, learning_rate,
                      beta1, beta2, beta1_correction, beta2_correction, eps,
                      weight_decay);
      break;
    default:
      std::printf("Invalid kernel number\n");
      std::exit(1);
  }
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 4) {
    std::fprintf(stderr, "usage: %s <input> <output> N\n", argv[0]);
    return 2;
  }
  const long N = std::atol(argv[3]);
  if (N < 0) {
    std::fprintf(stderr, "N must be non-negative\n");
    return 2;
  }
  const std::size_t n = static_cast<std::size_t>(N);

  FILE *fin = std::fopen(argv[1], "rb");
  if (!fin) {
    std::perror(argv[1]);
    return 2;
  }
  float *params = static_cast<float *>(std::malloc(sizeof(float) * n));
  float *grads = static_cast<float *>(std::malloc(sizeof(float) * n));
  float *m = static_cast<float *>(std::malloc(sizeof(float) * n));
  float *v = static_cast<float *>(std::malloc(sizeof(float) * n));
  if (!params || !grads || !m || !v) {
    std::fprintf(stderr, "host malloc failed\n");
    return 2;
  }
  auto read_array = [&](float *dst, const char *what) {
    if (n > 0 &&
        std::fread(dst, sizeof(float), n, fin) != n) {
      std::fprintf(stderr, "input too short at %s\n", what);
      std::exit(2);
    }
  };
  read_array(params, "params");
  read_array(grads, "grads");
  read_array(m, "m");
  read_array(v, "v");
  std::fclose(fin);

  FILE *fout = std::fopen(argv[2], "wb");
  if (!fout) {
    std::perror(argv[2]);
    return 2;
  }

  sycl::queue q;
  float *d_params =
      n == 0 ? nullptr : sycl::malloc_device<float>(n, q);
  float *d_grads = n == 0 ? nullptr : sycl::malloc_device<float>(n, q);
  float *d_m = n == 0 ? nullptr : sycl::malloc_device<float>(n, q);
  float *d_v = n == 0 ? nullptr : sycl::malloc_device<float>(n, q);
  float *d_params_out =
      n == 0 ? nullptr : sycl::malloc_device<float>(n, q);
  float *d_m_out = n == 0 ? nullptr : sycl::malloc_device<float>(n, q);
  float *d_v_out = n == 0 ? nullptr : sycl::malloc_device<float>(n, q);
  float *params_out = static_cast<float *>(std::malloc(sizeof(float) * (n == 0 ? 1 : n)));
  float *m_out = static_cast<float *>(std::malloc(sizeof(float) * (n == 0 ? 1 : n)));
  float *v_out = static_cast<float *>(std::malloc(sizeof(float) * (n == 0 ? 1 : n)));
  if (n > 0 && (!d_params || !d_grads || !d_m || !d_v || !d_params_out ||
                !d_m_out || !d_v_out || !params_out || !m_out || !v_out)) {
    std::fprintf(stderr, "device/host malloc failed\n");
    return 2;
  }

  if (n > 0) {
    q.memcpy(d_params, params, sizeof(float) * n).wait();
    q.memcpy(d_grads, grads, sizeof(float) * n).wait();
    q.memcpy(d_m, m, sizeof(float) * n).wait();
    q.memcpy(d_v, v, sizeof(float) * n).wait();
  }

  const int variants[] = {1, 2};
  for (int vi = 0; vi < 2; ++vi) {
    const int k = variants[vi];
    if (n > 0) {
      q.memcpy(d_params_out, params, sizeof(float) * n).wait();
      q.memcpy(d_m_out, m, sizeof(float) * n).wait();
      q.memcpy(d_v_out, v, sizeof(float) * n).wait();
    }
    adamw(q, k, d_params_out, d_grads, d_m_out, d_v_out, 10, N);
    q.wait();
    if (n > 0) {
      q.memcpy(params_out, d_params_out, sizeof(float) * n).wait();
      std::fwrite(params_out, sizeof(float), n, fout);
      q.memcpy(m_out, d_m_out, sizeof(float) * n).wait();
      std::fwrite(m_out, sizeof(float), n, fout);
      q.memcpy(v_out, d_v_out, sizeof(float) * n).wait();
      std::fwrite(v_out, sizeof(float), n, fout);
    }
  }
  std::fclose(fout);

  sycl::free(d_params, q);
  sycl::free(d_grads, q);
  sycl::free(d_m, q);
  sycl::free(d_v, q);
  sycl::free(d_params_out, q);
  sycl::free(d_m_out, q);
  sycl::free(d_v_out, q);
  std::free(params);
  std::free(grads);
  std::free(m);
  std::free(v);
  std::free(params_out);
  std::free(m_out);
  std::free(v_out);
  return 0;
}
