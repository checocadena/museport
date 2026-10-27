#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

template <typename T>
inline T ceil_div(T a, T b) { return (a + b - 1) / b; }

struct SoftmaxParams {
  float scale;
  float offset;
};

// ---- kernel 1: scalar online softmax, one work-group per row ----
class FusedK1;
void launch_k1(sycl::queue &q, float *dlogits, float *losses, const float *logits,
               const float *dlosses, const int *targets, int B, int T, int V, int P,
               int block_size) {
  const int64_t N = (int64_t)B * T;
  const std::size_t local = (std::size_t)block_size;
  const std::size_t global = (std::size_t)N * local;
  q.submit([&](sycl::handler &h) {
    h.parallel_for<FusedK1>(sycl::nd_range<1>(global, local),
                            [=](sycl::nd_item<1> it) {
      auto g = it.get_group();
      int64_t idx = (int64_t)it.get_group(0);
      if (idx >= N) return;
      std::size_t lid = it.get_local_id(0);
      std::size_t lsz = it.get_local_range(0);
      const float *x = logits + idx * (int64_t)P;
      float thread_max = -INFINITY;
      float thread_sum = 0.0f;
      for (int i = (int)lid; i < V; i += (int)lsz) {
        float v = x[i];
        float old = thread_max;
        thread_max = sycl::fmax(thread_max, v);
        thread_sum *= sycl::exp(old - thread_max);
        thread_sum += sycl::exp(v - thread_max);
      }
      float gmax = sycl::reduce_over_group(g, thread_max, sycl::maximum<float>{});
      thread_sum *= sycl::exp(thread_max - gmax);
      float gsum = sycl::reduce_over_group(g, thread_sum, sycl::plus<float>{});
      float scale = 1.0f / gsum;
      float offset = gmax;
      int ix = targets[idx];
      if (lid == 0) {
        float prob = sycl::exp(x[ix] - offset) * scale;
        losses[idx] = -sycl::log(prob);
      }
      float dloss = dlosses[idx];
      float *dl = dlogits + idx * (int64_t)P;
      for (int i = (int)lid; i < V; i += (int)lsz) {
        float prob = sycl::exp(x[i] - offset) * scale;
        float ind = (i == ix) ? 1.0f : 0.0f;
        dl[i] = (prob - ind) * dloss;
      }
    });
  });
}

// ---- kernel 2: vectorized (float4) forward + gradient ----
class FusedK2;
void launch_k2(sycl::queue &q, float *dlogits, float *losses, const float *logits,
               const float *dlosses, const int *targets, int B, int T, int V, int P,
               int block_size) {
  const int64_t N = (int64_t)B * T;
  const std::size_t local = (std::size_t)block_size;
  const std::size_t global = (std::size_t)N * local;
  const int V4 = (V + 3) / 4;
  q.submit([&](sycl::handler &h) {
    h.parallel_for<FusedK2>(sycl::nd_range<1>(global, local),
                            [=](sycl::nd_item<1> it) {
      auto g = it.get_group();
      int64_t idx = (int64_t)it.get_group(0);
      if (idx >= N) return;
      std::size_t lid = it.get_local_id(0);
      std::size_t lsz = it.get_local_range(0);
      const float *x = logits + idx * (int64_t)P;
      const sycl::vec<float, 4> *xv =
          reinterpret_cast<const sycl::vec<float, 4> *>(x);
      float thread_max = -INFINITY;
      float thread_sum = 0.0f;
      // reverse order to mirror CUDA kernel 2 cache behaviour
      for (int i = V4 - 1 - (int)lid; i >= 0; i -= (int)lsz) {
        sycl::vec<float, 4> v4 = xv[i];
        for (int k = 0; k < 4; ++k) {
          int e = i * 4 + k;
          if (e >= V) continue;
          float v = v4[k];
          float old = thread_max;
          thread_max = sycl::fmax(thread_max, v);
          thread_sum *= sycl::exp(old - thread_max);
          thread_sum += sycl::exp(v - thread_max);
        }
      }
      float gmax = sycl::reduce_over_group(g, thread_max, sycl::maximum<float>{});
      thread_sum *= sycl::exp(thread_max - gmax);
      float gsum = sycl::reduce_over_group(g, thread_sum, sycl::plus<float>{});
      float scale = 1.0f / gsum;
      float offset = gmax;
      int ix = targets[idx];
      if (lid == 0) {
        float prob = sycl::exp(x[ix] - offset) * scale;
        losses[idx] = -sycl::log(prob);
      }
      float dloss = dlosses[idx];
      float *dl = dlogits + idx * (int64_t)P;
      for (int i = (int)lid; i < V4; i += (int)lsz) {
        sycl::vec<float, 4> v4 = xv[i];
        for (int k = 0; k < 4; ++k) {
          int e = i * 4 + k;
          if (e >= V) continue;
          float prob = sycl::exp(v4[k] - offset) * scale;
          float ind = (e == ix) ? 1.0f : 0.0f;
          dl[e] = (prob - ind) * dloss;
        }
      }
    });
  });
}

// ---- kernel 3: scalar, reverse loop, no vectorization ----
class FusedK3;
void launch_k3(sycl::queue &q, float *dlogits, float *losses, const float *logits,
               const float *dlosses, const int *targets, int B, int T, int V, int P,
               int block_size) {
  const int64_t N = (int64_t)B * T;
  const std::size_t local = (std::size_t)block_size;
  const std::size_t global = (std::size_t)N * local;
  q.submit([&](sycl::handler &h) {
    h.parallel_for<FusedK3>(sycl::nd_range<1>(global, local),
                            [=](sycl::nd_item<1> it) {
      auto g = it.get_group();
      int64_t idx = (int64_t)it.get_group(0);
      if (idx >= N) return;
      std::size_t lid = it.get_local_id(0);
      std::size_t lsz = it.get_local_range(0);
      const float *x = logits + idx * (int64_t)P;
      float thread_max = -INFINITY;
      float thread_sum = 0.0f;
      for (int i = V - 1 - (int)lid; i >= 0; i -= (int)lsz) {
        float v = x[i];
        float old = thread_max;
        thread_max = sycl::fmax(thread_max, v);
        thread_sum *= sycl::exp(old - thread_max);
        thread_sum += sycl::exp(v - thread_max);
      }
      float gmax = sycl::reduce_over_group(g, thread_max, sycl::maximum<float>{});
      thread_sum *= sycl::exp(thread_max - gmax);
      float gsum = sycl::reduce_over_group(g, thread_sum, sycl::plus<float>{});
      float scale = 1.0f / gsum;
      float offset = gmax;
      int ix = targets[idx];
      if (lid == 0) {
        float prob = sycl::exp(x[ix] - offset) * scale;
        losses[idx] = -sycl::log(prob);
      }
      float dloss = dlosses[idx];
      float *dl = dlogits + idx * (int64_t)P;
      for (int i = (int)lid; i < V; i += (int)lsz) {
        float prob = sycl::exp(x[i] - offset) * scale;
        float ind = (i == ix) ? 1.0f : 0.0f;
        dl[i] = (prob - ind) * dloss;
      }
    });
  });
}

// ---- kernel 4: packed-128 (4x float) style with reverse blocked loop ----
class FusedK4;
void launch_k4(sycl::queue &q, float *dlogits, float *losses, const float *logits,
               const float *dlosses, const int *targets, int B, int T, int V, int P,
               int block_size) {
  const int64_t N = (int64_t)B * T;
  const std::size_t local = (std::size_t)block_size;
  const std::size_t global = (std::size_t)N * local;
  constexpr int W = 4;
  const int blocks = ceil_div(V, W);
  q.submit([&](sycl::handler &h) {
    h.parallel_for<FusedK4>(sycl::nd_range<1>(global, local),
                            [=](sycl::nd_item<1> it) {
      auto g = it.get_group();
      int64_t idx = (int64_t)it.get_group(0);
      if (idx >= N) return;
      std::size_t lid = it.get_local_id(0);
      std::size_t lsz = it.get_local_range(0);
      const float *x = logits + idx * (int64_t)P;
      float thread_max = -INFINITY;
      float thread_sum = 0.0f;
      for (int i = blocks - 1 - (int)lid; i >= 0; i -= (int)lsz) {
        for (int k = 0; k < W; ++k) {
          int e = i * W + k;
          if (e >= V) continue;
          float v = x[e];
          float old = thread_max;
          thread_max = sycl::fmax(thread_max, v);
          thread_sum *= sycl::exp(old - thread_max);
          thread_sum += sycl::exp(v - thread_max);
        }
      }
      float gmax = sycl::reduce_over_group(g, thread_max, sycl::maximum<float>{});
      thread_sum *= sycl::exp(thread_max - gmax);
      float gsum = sycl::reduce_over_group(g, thread_sum, sycl::plus<float>{});
      float scale = 1.0f / gsum;
      float offset = gmax;
      int ix = targets[idx];
      if (lid == 0) {
        float prob = sycl::exp(x[ix] - offset) * scale;
        losses[idx] = -sycl::log(prob);
      }
      float dloss = dlosses[idx];
      float *dl = dlogits + idx * (int64_t)P;
      for (int i = (int)lid; i < blocks; i += (int)lsz) {
        float buf[W];
        for (int k = 0; k < W; ++k) {
          int e = i * W + k;
          float v = (e < V) ? x[e] : 0.0f;
          buf[k] = v;
        }
        for (int k = 0; k < W; ++k) {
          int e = i * W + k;
          if (e >= V) continue;
          float prob = sycl::exp(buf[k] - offset) * scale;
          float ind = (e == ix) ? 1.0f : 0.0f;
          dl[e] = (prob - ind) * dloss;
        }
      }
    });
  });
}

// ---- kernel 5: split bulk (multiple of 4) + scalar tail ----
class FusedK5;
void launch_k5(sycl::queue &q, float *dlogits, float *losses, const float *logits,
               const float *dlosses, const int *targets, int B, int T, int V, int P,
               int block_size) {
  const int64_t N = (int64_t)B * T;
  const std::size_t local = (std::size_t)block_size;
  const std::size_t global = (std::size_t)N * local;
  constexpr int W = 4;
  const int bulk = V / W * W; // elements covered without bounds checks
  const int nvec = bulk / W;
  q.submit([&](sycl::handler &h) {
    h.parallel_for<FusedK5>(sycl::nd_range<1>(global, local),
                            [=](sycl::nd_item<1> it) {
      auto g = it.get_group();
      int64_t idx = (int64_t)it.get_group(0);
      if (idx >= N) return;
      std::size_t lid = it.get_local_id(0);
      std::size_t lsz = it.get_local_range(0);
      const float *x = logits + idx * (int64_t)P;
      float thread_max = -INFINITY;
      float thread_sum = 0.0f;
      // tail (unaligned) elements first, one per work-item strided
      for (int i = bulk + (int)lid; i < V; i += (int)lsz) {
        float v = x[i];
        float old = thread_max;
        thread_max = sycl::fmax(thread_max, v);
        thread_sum *= sycl::exp(old - thread_max);
        thread_sum += sycl::exp(v - thread_max);
      }
      // bulk without bounds checks
      for (int i = nvec - 1 - (int)lid; i >= 0; i -= (int)lsz) {
        for (int k = 0; k < W; ++k) {
          float v = x[i * W + k];
          float old = thread_max;
          thread_max = sycl::fmax(thread_max, v);
          thread_sum *= sycl::exp(old - thread_max);
          thread_sum += sycl::exp(v - thread_max);
        }
      }
      float gmax = sycl::reduce_over_group(g, thread_max, sycl::maximum<float>{});
      thread_sum *= sycl::exp(thread_max - gmax);
      float gsum = sycl::reduce_over_group(g, thread_sum, sycl::plus<float>{});
      float scale = 1.0f / gsum;
      float offset = gmax;
      int ix = targets[idx];
      if (lid == 0) {
        float prob = sycl::exp(x[ix] - offset) * scale;
        losses[idx] = -sycl::log(prob);
      }
      float dloss = dlosses[idx];
      float *dl = dlogits + idx * (int64_t)P;
      for (int i = (int)lid; i < nvec; i += (int)lsz) {
        for (int k = 0; k < W; ++k) {
          int e = i * W + k;
          float prob = sycl::exp(x[e] - offset) * scale;
          float ind = (e == ix) ? 1.0f : 0.0f;
          dl[e] = (prob - ind) * dloss;
        }
      }
      for (int i = bulk + (int)lid; i < V; i += (int)lsz) {
        float prob = sycl::exp(x[i] - offset) * scale;
        float ind = (i == ix) ? 1.0f : 0.0f;
        dl[i] = (prob - ind) * dloss;
      }
    });
  });
}

void fused_classifier(sycl::queue &q, int kernel_num, float *dlogits, float *losses,
                      const float *logits, const float *dlosses, const int *targets,
                      int B, int T, int V, int P, int block_size) {
  switch (kernel_num) {
  case 1: launch_k1(q, dlogits, losses, logits, dlosses, targets, B, T, V, P, block_size); break;
  case 2: launch_k2(q, dlogits, losses, logits, dlosses, targets, B, T, V, P, block_size); break;
  case 3: launch_k3(q, dlogits, losses, logits, dlosses, targets, B, T, V, P, block_size); break;
  case 4: launch_k4(q, dlogits, losses, logits, dlosses, targets, B, T, V, P, block_size); break;
  case 5: launch_k5(q, dlogits, losses, logits, dlosses, targets, B, T, V, P, block_size); break;
  default: std::fprintf(stderr, "Invalid kernel number\n"); std::exit(1);
  }
}

int main(int argc, char **argv) {
  if (argc != 6) {
    std::fprintf(stderr, "usage: %s <input> <output> B T V\n", argv[0]);
    return 2;
  }
  const int B = std::atoi(argv[3]);
  const int T = std::atoi(argv[4]);
  const int V = std::atoi(argv[5]);
  const int64_t N = (int64_t)B * T;
  const int64_t NV = N * V;
  FILE *fin = std::fopen(argv[1], "rb");
  if (!fin) { std::perror(argv[1]); return 2; }
  std::vector<float> logits((std::size_t)NV);
  std::vector<float> dlosses((std::size_t)N);
  std::vector<int> targets((std::size_t)N);
  std::vector<int> outliers((std::size_t)N * 3);
  if (std::fread(logits.data(), sizeof(float), (std::size_t)NV, fin) != (std::size_t)NV) {
    std::fprintf(stderr, "input too short at logits\n"); return 2;
  }
  if (std::fread(dlosses.data(), sizeof(float), (std::size_t)N, fin) != (std::size_t)N) {
    std::fprintf(stderr, "input too short at dlosses\n"); return 2;
  }
  if (std::fread(targets.data(), sizeof(int), (std::size_t)N, fin) != (std::size_t)N) {
    std::fprintf(stderr, "input too short at targets\n"); return 2;
  }
  if (std::fread(outliers.data(), sizeof(int), (std::size_t)N * 3, fin) != (std::size_t)N * 3) {
    std::fprintf(stderr, "input too short at outliers\n"); return 2;
  }
  std::fclose(fin);
  FILE *fout = std::fopen(argv[2], "wb");
  if (!fout) { std::perror(argv[2]); return 2; }

  for (int k = 0; k < 3; ++k) {
    for (int64_t j = 0; j < N; ++j) {
      logits[(std::size_t)j * V + (std::size_t)outliers[(std::size_t)j * 3 + k]] *= 20.0f;
    }
  }

  const int P = (V + 63) & ~63;
  const int64_t NP = N * (int64_t)P;

  sycl::queue q;
  float *d_logits_p = sycl::malloc_device<float>((std::size_t)NP, q);
  float *d_dlogits_p = sycl::malloc_device<float>((std::size_t)NP, q);
  float *d_losses = sycl::malloc_device<float>((std::size_t)N, q);
  float *d_dlosses = sycl::malloc_device<float>((std::size_t)N, q);
  int *d_targets = sycl::malloc_device<int>((std::size_t)N, q);
  if (!d_logits_p || !d_dlogits_p || !d_losses || !d_dlosses || !d_targets) {
    std::fprintf(stderr, "device alloc failed\n"); return 1;
  }

  // host padded staging (zero padding; kernels only touch first V per row)
  std::vector<float> h_logits_p((std::size_t)NP, 0.0f);
  for (int64_t r = 0; r < N; ++r) {
    std::memcpy(h_logits_p.data() + r * P, logits.data() + r * V, (std::size_t)V * sizeof(float));
  }
  q.memcpy(d_logits_p, h_logits_p.data(), (std::size_t)NP * sizeof(float)).wait();
  q.memcpy(d_dlosses, dlosses.data(), (std::size_t)N * sizeof(float)).wait();
  q.memcpy(d_targets, targets.data(), (std::size_t)N * sizeof(int)).wait();

  std::vector<float> losses((std::size_t)N);
  std::vector<float> dlogits((std::size_t)NV);
  std::vector<float> h_dlogits_p((std::size_t)NP);
  const int variants[] = {1, 2, 3, 4, 5};
  for (int vi = 0; vi < 5; ++vi) {
    const int k = variants[vi];
    q.fill(d_losses, 0.0f, (std::size_t)N).wait();
    q.fill(d_dlogits_p, 0.0f, (std::size_t)NP).wait();
    fused_classifier(q, k, d_dlogits_p, d_losses, d_logits_p, d_dlosses, d_targets,
                     B, T, V, P, 256);
    q.wait();
    q.memcpy(losses.data(), d_losses, (std::size_t)N * sizeof(float)).wait();
    std::fwrite(losses.data(), sizeof(float), (std::size_t)N, fout);
    q.memcpy(h_dlogits_p.data(), d_dlogits_p, (std::size_t)NP * sizeof(float)).wait();
    for (int64_t r = 0; r < N; ++r) {
      std::memcpy(dlogits.data() + r * V, h_dlogits_p.data() + r * P,
                  (std::size_t)V * sizeof(float));
    }
    std::fwrite(dlogits.data(), sizeof(float), (std::size_t)NV, fout);
  }
  std::fclose(fout);
  sycl::free(d_logits_p, q);
  sycl::free(d_dlogits_p, q);
  sycl::free(d_losses, q);
  sycl::free(d_dlosses, q);
  sycl::free(d_targets, q);
  return 0;
}
