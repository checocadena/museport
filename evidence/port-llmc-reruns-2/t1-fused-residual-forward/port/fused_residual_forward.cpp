#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <algorithm>
#include <vector>

using floatX = float;
using floatN = float;

template <class T>
inline T ceil_div(T a, T b) {
  return (a + b - 1) / b;
}

template <class T>
struct alignas(16) Packed128 {
  static constexpr int size = 16 / sizeof(T);
  T payload[size];
  T &operator[](int i) { return payload[i]; }
  const T &operator[](int i) const { return payload[i]; }
};

template <class T>
inline Packed128<T> load128(const T *p) {
  Packed128<T> v;
  std::memcpy(v.payload, p, sizeof(v.payload));
  return v;
}
template <class T>
inline Packed128<T> load128cs(const T *p) {
  Packed128<T> v;
  std::memcpy(v.payload, p, sizeof(v.payload));
  return v;
}
template <class T>
inline void store128(T *p, Packed128<T> v) {
  std::memcpy(p, v.payload, sizeof(v.payload));
}
template <class T>
inline void store128cs(T *p, Packed128<T> v) {
  std::memcpy(p, v.payload, sizeof(v.payload));
}

inline float sg_sum(sycl::sub_group sg, float v) {
  return sycl::reduce_over_group(sg, v, sycl::plus<float>());
}

class Residual1Kernel;
class Layernorm1Kernel;
class Fused2Kernel;
class Fused3Kernel;
class Fused4Kernel;
class Fused5Kernel;
class Fused6Kernel;

static std::size_t sub_group_width(sycl::queue &q) {
  auto sizes =
      q.get_device().get_info<sycl::info::device::sub_group_sizes>();
  for (auto s : sizes) {
    if (s == 32) {
      return 32;
    }
  }
  if (!sizes.empty()) {
    return sizes.back();
  }
  return 32;
}

void fused_residual_forward1(sycl::queue &q, floatX *residual, floatX *normed,
                             floatX *mean, floatX *rstd, const floatX *inp1,
                             const floatX *inp2, const floatX *weight,
                             const floatX *bias, int N, int C,
                             int block_size) {
  long total = (long)N * (long)C;
  std::size_t local = (std::size_t)block_size;
  std::size_t global =
      ((std::size_t)total + local - 1) / local * local;
  if (global == 0) {
    global = local;
  }
  q.submit([&](sycl::handler &h) {
    h.parallel_for<class Residual1Kernel>(
        sycl::nd_range<1>({global}, {local}), [=](sycl::nd_item<1> it) {
          std::size_t idx = it.get_global_id(0);
          if (idx < (std::size_t)total) {
            residual[idx] = inp1[idx] + inp2[idx];
          }
        });
  }).wait();
  std::size_t global2 = ((std::size_t)N + local - 1) / local * local;
  if (global2 == 0) {
    global2 = local;
  }
  q.submit([&](sycl::handler &h) {
    h.parallel_for<class Layernorm1Kernel>(
        sycl::nd_range<1>({global2}, {local}), [=](sycl::nd_item<1> it) {
          int idx = (int)it.get_global_id(0);
          if (idx < N) {
            const floatX *row = residual + (long)idx * (long)C;
            float m = 0.0f;
            for (int i = 0; i < C; ++i) {
              m += (float)row[i];
            }
            m /= (float)C;
            float v = 0.0f;
            for (int i = 0; i < C; ++i) {
              float d = (float)row[i] - m;
              v += d * d;
            }
            v /= (float)C;
            float s = 1.0f / sycl::sqrt(v + 1e-5f);
            floatX *out = normed + (long)idx * (long)C;
            for (int i = 0; i < C; ++i) {
              float n = s * ((float)row[i] - m);
              out[i] = (floatX)(n * (float)weight[i] + (float)bias[i]);
            }
            mean[idx] = (floatX)m;
            rstd[idx] = (floatX)s;
          }
        });
  }).wait();
}

void fused_residual_forward2(sycl::queue &q, floatX *residual, floatX *normed,
                             floatX *mean, floatX *rstd, const floatX *inp1,
                             const floatX *inp2, const floatX *weight,
                             const floatX *bias, int N, int C,
                             int block_size) {
  std::size_t local = (std::size_t)block_size;
  std::size_t global = ((std::size_t)N + local - 1) / local * local;
  if (global == 0) {
    global = local;
  }
  q.submit([&](sycl::handler &h) {
    h.parallel_for<class Fused2Kernel>(
        sycl::nd_range<1>({global}, {local}), [=](sycl::nd_item<1> it) {
          int idx = (int)it.get_global_id(0);
          if (idx >= N) {
            return;
          }
          const floatX *a = inp1 + (long)idx * (long)C;
          const floatX *b = inp2 + (long)idx * (long)C;
          floatX *res = residual + (long)idx * (long)C;
          floatX *out = normed + (long)idx * (long)C;
          float m = 0.0f;
          for (int c = 0; c < C; ++c) {
            float o = (float)a[c] + (float)b[c];
            m += o;
            res[c] = (floatX)o;
          }
          m /= (float)C;
          float v = 0.0f;
          for (int c = 0; c < C; ++c) {
            float d = (float)res[c] - m;
            v += d * d;
          }
          v /= (float)C;
          float s = 1.0f / sycl::sqrt(v + 1e-5f);
          for (int c = 0; c < C; ++c) {
            float n = s * ((float)res[c] - m);
            out[c] = (floatX)(n * (float)weight[c] + (float)bias[c]);
          }
          mean[idx] = (floatX)m;
          rstd[idx] = (floatX)s;
        });
  }).wait();
}

void fused_residual_forward3(sycl::queue &q, floatX *residual, floatX *normed,
                             floatX *mean, floatX *rstd, const floatX *inp1,
                             const floatX *inp2, const floatX *weight,
                             const floatX *bias, int N, int C,
                             int block_size) {
  std::size_t lanes = sub_group_width(q);
  int block_y = std::max(1, block_size / (int)lanes);
  std::size_t local_y = (std::size_t)block_y;
  std::size_t local_x = lanes;
  int grid = (N + block_y - 1) / block_y;
  if (grid < 1) {
    grid = 1;
  }
  sycl::range<2> local_range(local_y, local_x);
  sycl::range<2> global_range((std::size_t)grid * local_y, local_x);
  q.submit([&](sycl::handler &h) {
    h.parallel_for<class Fused3Kernel>(
        sycl::nd_range<2>(global_range, local_range),
        [=](sycl::nd_item<2> it) {
          auto sg = it.get_sub_group();
          std::size_t sg_n = sg.get_local_range()[0];
          int ly = (int)it.get_local_id(0);
          int lane = (int)sg.get_local_id()[0];
          int idx =
              (int)(it.get_group(0) * it.get_local_range(0) + (std::size_t)ly);
          if (idx >= N) {
            return;
          }
          const floatX *a = inp1 + (long)idx * (long)C;
          const floatX *b = inp2 + (long)idx * (long)C;
          floatX *res = residual + (long)idx * (long)C;
          floatX *out = normed + (long)idx * (long)C;
          float m = 0.0f;
          for (int c = lane; c < C; c += (int)sg_n) {
            float o = (float)a[c] + (float)b[c];
            m += o;
            res[c] = (floatX)o;
          }
          m = sg_sum(sg, m) / (float)C;
          float v = 0.0f;
          for (int c = lane; c < C; c += (int)sg_n) {
            float d = (float)res[c] - m;
            v += d * d;
          }
          v = sg_sum(sg, v) / (float)C;
          float s = 1.0f / sycl::sqrt(v + 1e-5f);
          for (int c = lane; c < C; c += (int)sg_n) {
            float n = s * ((float)res[c] - m);
            out[c] = (floatX)(n * (float)weight[c] + (float)bias[c]);
          }
          if (lane == 0) {
            mean[idx] = (floatX)m;
            rstd[idx] = (floatX)s;
          }
        });
  }).wait();
}

void fused_residual_forward4(sycl::queue &q, floatX *residual, floatX *normed,
                             floatX *mean, floatX *rstd, const floatX *inp1,
                             const floatX *inp2, const floatX *weight,
                             const floatX *bias, int N, int C,
                             int block_size) {
  std::size_t lanes = sub_group_width(q);
  int block_y = std::max(1, block_size / (int)lanes);
  std::size_t local_y = (std::size_t)block_y;
  std::size_t local_x = lanes;
  int grid = (N + block_y - 1) / block_y;
  if (grid < 1) {
    grid = 1;
  }
  sycl::range<2> local_range(local_y, local_x);
  sycl::range<2> global_range((std::size_t)grid * local_y, local_x);
  using Vec = Packed128<floatX>;
  q.submit([&](sycl::handler &h) {
    h.parallel_for<class Fused4Kernel>(
        sycl::nd_range<2>(global_range, local_range),
        [=](sycl::nd_item<2> it) {
          auto sg = it.get_sub_group();
          std::size_t sg_n = sg.get_local_range()[0];
          int lane = (int)sg.get_local_id()[0];
          int idx =
              (int)(it.get_group(0) * it.get_local_range(0) +
                    (std::size_t)it.get_local_id(0));
          if (idx >= N) {
            return;
          }
          const floatX *a = inp1 + (long)idx * (long)C;
          const floatX *b = inp2 + (long)idx * (long)C;
          floatX *res = residual + (long)idx * (long)C;
          floatX *out = normed + (long)idx * (long)C;
          constexpr int V = Packed128<floatX>::size;
          float sum = 0.0f;
          float sum_sq = 0.0f;
          int c = lane * V;
          for (; c < C; c += (int)sg_n * V) {
            Vec in1 = load128cs(a + c);
            Vec in2 = load128cs(b + c);
            Vec o;
            for (int k = 0; k < V; ++k) {
              float f = (float)in1[k] + (float)in2[k];
              o[k] = (floatX)f;
              sum += f;
              sum_sq += f * f;
            }
            store128(res + c, o);
          }
          sum = sg_sum(sg, sum);
          sum_sq = sg_sum(sg, sum_sq);
          float m = sum / (float)C;
          float vv = sum_sq / (float)C - m * m;
          float s = sycl::rsqrt(vv + 1e-5f);
          c -= (int)sg_n * V;
          for (; c >= 0; c -= (int)sg_n * V) {
            Vec r = load128cs(res + c);
            Vec w = load128(weight + c);
            Vec bi = load128(bias + c);
            Vec o;
            for (int k = 0; k < V; ++k) {
              float n = s * ((float)r[k] - m);
              o[k] = (floatX)(n * (float)w[k] + (float)bi[k]);
            }
            store128cs(out + c, o);
          }
          if (lane == 0) {
            mean[idx] = (floatX)m;
            rstd[idx] = (floatX)s;
          }
        });
  }).wait();
}

void fused_residual_forward5(sycl::queue &q, floatX *residual, floatX *normed,
                             floatX *mean, floatX *rstd, const floatX *inp1,
                             const floatX *inp2, const floatX *weight,
                             const floatX *bias, int N, int C,
                             int block_size) {
  std::size_t lanes = sub_group_width(q);
  int block_y = std::max(1, block_size / (int)lanes);
  std::size_t local_y = (std::size_t)block_y;
  std::size_t local_x = lanes;
  int grid = (N + block_y - 1) / block_y;
  if (grid < 1) {
    grid = 1;
  }
  std::size_t smem_floats = (std::size_t)(2 + block_y) * (std::size_t)C;
  std::size_t local_bytes =
      q.get_device().get_info<sycl::info::device::local_mem_size>();
  if (smem_floats * sizeof(float) > local_bytes) {
    fused_residual_forward4(q, residual, normed, mean, rstd, inp1, inp2,
                            weight, bias, N, C, block_size);
    return;
  }
  sycl::range<2> local_range(local_y, local_x);
  sycl::range<2> global_range((std::size_t)grid * local_y, local_x);
  using Vec = Packed128<floatX>;
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> scratch(sycl::range<1>(smem_floats), h);
    h.parallel_for<class Fused5Kernel>(
        sycl::nd_range<2>(global_range, local_range),
        [=](sycl::nd_item<2> it) {
          auto sg = it.get_sub_group();
          std::size_t sg_n = sg.get_local_range()[0];
          int ly = (int)it.get_local_id(0);
          int lane = (int)sg.get_local_id()[0];
          std::size_t group_y = it.get_local_range(0);
          constexpr int V = Packed128<floatX>::size;
          int packs = C / V;
          auto mp = scratch.get_multi_ptr<sycl::access::decorated::no>();
          float *base = mp.get();
          float *w_base = base;
          float *b_base = base + (std::size_t)C;
          float *r_base =
              base + (std::size_t)(2 + ly) * (std::size_t)C;
          int flat = lane + (int)sg_n * ly;
          int step_flat = (int)(group_y * sg_n);
          for (int i = flat; i < packs; i += step_flat) {
            Vec w = load128(weight + i * V);
            Vec b = load128(bias + i * V);
            store128(w_base + i * V, w);
            store128(b_base + i * V, b);
          }
          sycl::group_barrier(it.get_group());
          int idx =
              (int)(it.get_group(0) * it.get_local_range(0) + (std::size_t)ly);
          if (idx >= N) {
            return;
          }
          const floatX *a = inp1 + (long)idx * (long)C;
          const floatX *bp = inp2 + (long)idx * (long)C;
          floatX *res = residual + (long)idx * (long)C;
          floatX *out = normed + (long)idx * (long)C;
          float sum = 0.0f;
          for (int c = lane * V; c < C; c += (int)sg_n * V) {
            Vec in1 = load128cs(a + c);
            Vec in2 = load128cs(bp + c);
            Vec o;
            for (int k = 0; k < V; ++k) {
              float f = (float)in1[k] + (float)in2[k];
              o[k] = (floatX)f;
              sum += f;
            }
            store128cs(res + c, o);
            store128(r_base + c, o);
          }
          sum = sg_sum(sg, sum);
          float m = sum / (float)C;
          float v = 0.0f;
          for (int c = lane * V; c < C; c += (int)sg_n * V) {
            Vec r = load128(r_base + c);
            for (int k = 0; k < V; ++k) {
              float d = (float)r[k] - m;
              v += d * d;
            }
          }
          v = sg_sum(sg, v) / (float)C;
          float s = sycl::rsqrt(v + 1e-5f);
          for (int c = lane * V; c < C; c += (int)sg_n * V) {
            Vec r = load128(r_base + c);
            Vec w = load128(w_base + c);
            Vec b = load128(b_base + c);
            Vec o;
            for (int k = 0; k < V; ++k) {
              float n = s * ((float)r[k] - m);
              o[k] = (floatX)(n * (float)w[k] + (float)b[k]);
            }
            store128cs(out + c, o);
          }
          if (lane == 0) {
            mean[idx] = (floatX)m;
            rstd[idx] = (floatX)s;
          }
        });
  }).wait();
}

void fused_residual_forward6(sycl::queue &q, floatX *residual, floatX *normed,
                             floatX *mean, floatX *rstd, const floatX *inp1,
                             const floatX *inp2, const floatX *weight,
                             const floatX *bias, int N, int C,
                             int block_size) {
  std::size_t lanes = sub_group_width(q);
  constexpr int V = Packed128<floatX>::size;
  int warps_per_token =
      std::max(1, (int)((std::size_t)C / (std::size_t)V / lanes));
  int total_warps = std::max(1, block_size / (int)lanes);
  int block_z = std::max(1, total_warps / warps_per_token);
  int block_y = std::max(1, total_warps / block_z);
  std::size_t cus =
      q.get_device().get_info<sycl::info::device::max_compute_units>();
  int num_blocks = std::max(1, (int)(cus * 2));
  if (num_blocks > 64) {
    num_blocks = 64;
  }
  std::size_t smem_floats = (std::size_t)(2 + block_z) * (std::size_t)C +
                            (std::size_t)64 * (std::size_t)block_z;
  std::size_t local_bytes =
      q.get_device().get_info<sycl::info::device::local_mem_size>();
  if (smem_floats * sizeof(float) > local_bytes) {
    fused_residual_forward4(q, residual, normed, mean, rstd, inp1, inp2,
                            weight, bias, N, C, block_size);
    return;
  }
  sycl::range<3> local_range((std::size_t)block_z, (std::size_t)block_y,
                             lanes);
  sycl::range<3> global_range((std::size_t)num_blocks * (std::size_t)block_z,
                              (std::size_t)block_y, lanes);
  using Vec = Packed128<floatX>;
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> scratch(sycl::range<1>(smem_floats), h);
    h.parallel_for<class Fused6Kernel>(
        sycl::nd_range<3>(global_range, local_range),
        [=](sycl::nd_item<3> it) {
          auto sg = it.get_sub_group();
          std::size_t sg_n = sg.get_local_range()[0];
          int lz = (int)it.get_local_id(0);
          int ly = (int)it.get_local_id(1);
          int lane = (int)sg.get_local_id()[0];
          int group_y = (int)it.get_local_range(1);
          int group_z = (int)it.get_local_range(0);
          int groups = (int)it.get_group_range(0);
          int g0 = (int)it.get_group(0);
          auto mp = scratch.get_multi_ptr<sycl::access::decorated::no>();
          float *base = mp.get();
          float *w_base = base;
          float *b_base = base + (std::size_t)C;
          float *mean_var_base =
              base + (std::size_t)(2 + group_z) * (std::size_t)C;
          auto res_for = [&](int z) -> float * {
            return base + (std::size_t)(2 + z) * (std::size_t)C;
          };
          int packs = C / V;
          int flat = lane + (int)sg_n * ly;
          int flat_step = group_y * (int)sg_n;
          for (int i = flat; i < packs; i += flat_step) {
            Vec w = load128(weight + i * V);
            Vec b = load128(bias + i * V);
            store128(w_base + i * V, w);
            store128(b_base + i * V, b);
          }
          int stride = groups * group_z;
          int iters = (N + stride - 1) / stride;
          if (iters < 1) {
            iters = 1;
          }
          for (int iter = 0; iter < iters; ++iter) {
            int tidx = g0 * group_z + lz + iter * stride;
            bool active = tidx < N;
            float *r_base = res_for(lz);
            float *s_mean =
                mean_var_base + (std::size_t)lz * 32;
            float *s_var = mean_var_base +
                           (std::size_t)32 * (std::size_t)group_z +
                           (std::size_t)lz * 32;
            float sum = 0.0f;
            if (active) {
              const floatX *a = inp1 + (long)tidx * (long)C;
              const floatX *bp = inp2 + (long)tidx * (long)C;
              floatX *res = residual + (long)tidx * (long)C;
              for (int c = flat * V; c < C; c += flat_step * V) {
                Vec in1 = load128cs(a + c);
                Vec in2 = load128cs(bp + c);
                Vec o;
                for (int k = 0; k < V; ++k) {
                  float f = (float)in1[k] + (float)in2[k];
                  o[k] = (floatX)f;
                  sum += f;
                }
                store128cs(res + c, o);
                store128(r_base + c, o);
              }
            }
            sum = sg_sum(sg, active ? sum : 0.0f);
            if (lane == 0) {
              s_mean[ly] = active ? sum : 0.0f;
            }
            sycl::group_barrier(it.get_group());
            float part = 0.0f;
            if (lane < group_y) {
              part = s_mean[lane];
            }
            float m = sg_sum(sg, part) / (float)C;
            float v = 0.0f;
            if (active) {
              for (int c = flat * V; c < C; c += flat_step * V) {
                Vec r = load128(r_base + c);
                for (int k = 0; k < V; ++k) {
                  float d = (float)r[k] - m;
                  v += d * d;
                }
              }
            }
            v = sg_sum(sg, active ? v : 0.0f);
            if (lane == 0) {
              s_var[ly] = active ? v : 0.0f;
            }
            sycl::group_barrier(it.get_group());
            float vpart = 0.0f;
            if (lane < group_y) {
              vpart = s_var[lane];
            }
            float vv = sg_sum(sg, vpart) / (float)C;
            float s = sycl::rsqrt(vv + 1e-5f);
            if (active) {
              floatX *out = normed + (long)tidx * (long)C;
              for (int c = flat * V; c < C; c += flat_step * V) {
                Vec r = load128(r_base + c);
                Vec w = load128(w_base + c);
                Vec b = load128(b_base + c);
                Vec o;
                for (int k = 0; k < V; ++k) {
                  float n = s * ((float)r[k] - m);
                  o[k] = (floatX)(n * (float)w[k] + (float)b[k]);
                }
                store128(out + c, o);
              }
              if (lane == 0 && ly == 0) {
                mean[tidx] = (floatX)m;
                rstd[tidx] = (floatX)s;
              }
            }
            sycl::group_barrier(it.get_group());
          }
        });
  }).wait();
}

void fused_residual_forward(sycl::queue &q, int kernel_num, floatX *residual,
                            floatX *normed, floatX *mean, floatX *rstd,
                            const floatX *inp1, const floatX *inp2,
                            const floatX *weight, const floatX *bias, int N,
                            int C, int block_size) {
  switch (kernel_num) {
    case 1:
      fused_residual_forward1(q, residual, normed, mean, rstd, inp1, inp2,
                              weight, bias, N, C, block_size);
      break;
    case 2:
      fused_residual_forward2(q, residual, normed, mean, rstd, inp1, inp2,
                              weight, bias, N, C, block_size);
      break;
    case 3:
      fused_residual_forward3(q, residual, normed, mean, rstd, inp1, inp2,
                              weight, bias, N, C, block_size);
      break;
    case 4:
      fused_residual_forward4(q, residual, normed, mean, rstd, inp1, inp2,
                              weight, bias, N, C, block_size);
      break;
    case 5:
      fused_residual_forward5(q, residual, normed, mean, rstd, inp1, inp2,
                              weight, bias, N, C, block_size);
      break;
    case 6:
      fused_residual_forward6(q, residual, normed, mean, rstd, inp1, inp2,
                              weight, bias, N, C, block_size);
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
  int B = std::atoi(argv[3]);
  int T = std::atoi(argv[4]);
  int C = std::atoi(argv[5]);
  std::size_t n_tc = (std::size_t)B * (std::size_t)T * (std::size_t)C;
  std::size_t n_t = (std::size_t)B * (std::size_t)T;
  FILE *fin = std::fopen(argv[1], "rb");
  if (!fin) {
    std::perror(argv[1]);
    return 2;
  }
  std::vector<float> inp1(n_tc), inp2(n_tc);
  std::vector<float> weight((std::size_t)C), bias((std::size_t)C);
  if (std::fread(inp1.data(), sizeof(float), n_tc, fin) != n_tc) {
    std::fprintf(stderr, "input too short at inp1\n");
    return 2;
  }
  if (std::fread(inp2.data(), sizeof(float), n_tc, fin) != n_tc) {
    std::fprintf(stderr, "input too short at inp2\n");
    return 2;
  }
  if (std::fread(weight.data(), sizeof(float), (std::size_t)C, fin) !=
      (std::size_t)C) {
    std::fprintf(stderr, "input too short at weight\n");
    return 2;
  }
  if (std::fread(bias.data(), sizeof(float), (std::size_t)C, fin) !=
      (std::size_t)C) {
    std::fprintf(stderr, "input too short at bias\n");
    return 2;
  }
  std::fclose(fin);
  FILE *fout = std::fopen(argv[2], "wb");
  if (!fout) {
    std::perror(argv[2]);
    return 2;
  }
  sycl::queue q;
  floatX *d_inp1 = sycl::malloc_device<floatX>(n_tc, q);
  floatX *d_inp2 = sycl::malloc_device<floatX>(n_tc, q);
  floatX *d_weight = sycl::malloc_device<floatX>((std::size_t)C, q);
  floatX *d_bias = sycl::malloc_device<floatX>((std::size_t)C, q);
  floatX *d_residual = sycl::malloc_device<floatX>(n_tc, q);
  floatX *d_normed = sycl::malloc_device<floatX>(n_tc, q);
  floatX *d_mean = sycl::malloc_device<floatX>(n_t, q);
  floatX *d_rstd = sycl::malloc_device<floatX>(n_t, q);
  q.memcpy(d_inp1, inp1.data(), sizeof(float) * n_tc).wait();
  q.memcpy(d_inp2, inp2.data(), sizeof(float) * n_tc).wait();
  q.memcpy(d_weight, weight.data(), sizeof(float) * (std::size_t)C).wait();
  q.memcpy(d_bias, bias.data(), sizeof(float) * (std::size_t)C).wait();
  std::vector<float> residual(n_tc), normed(n_tc);
  std::vector<float> mean(n_t), rstd(n_t);
  const int variants[6] = {1, 2, 3, 4, 5, 6};
  for (int vi = 0; vi < 6; ++vi) {
    int k = variants[vi];
    q.memset(d_residual, 0, sizeof(float) * n_tc).wait();
    q.memset(d_normed, 0, sizeof(float) * n_tc).wait();
    q.memset(d_mean, 0, sizeof(float) * n_t).wait();
    q.memset(d_rstd, 0, sizeof(float) * n_t).wait();
    fused_residual_forward(q, k, d_residual, d_normed, d_mean, d_rstd, d_inp1,
                           d_inp2, d_weight, d_bias, B * T, C, 256);
    q.memcpy(residual.data(), d_residual, sizeof(float) * n_tc).wait();
    std::fwrite(residual.data(), sizeof(float), n_tc, fout);
    q.memcpy(normed.data(), d_normed, sizeof(float) * n_tc).wait();
    std::fwrite(normed.data(), sizeof(float), n_tc, fout);
    q.memcpy(mean.data(), d_mean, sizeof(float) * n_t).wait();
    std::fwrite(mean.data(), sizeof(float), n_t, fout);
    q.memcpy(rstd.data(), d_rstd, sizeof(float) * n_t).wait();
    std::fwrite(rstd.data(), sizeof(float), n_t, fout);
  }
  std::fclose(fout);
  sycl::free(d_inp1, q);
  sycl::free(d_inp2, q);
  sycl::free(d_weight, q);
  sycl::free(d_bias, q);
  sycl::free(d_residual, q);
  sycl::free(d_normed, q);
  sycl::free(d_mean, q);
  sycl::free(d_rstd, q);
  return 0;
}
