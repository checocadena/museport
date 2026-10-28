#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <vector>

template <typename T>
static T ceil_div(T a, T b) { return (a + b - 1) / b; }

static constexpr float kNegInf() {
  return -std::numeric_limits<float>::infinity();
}

struct SumMax {
  float maxval;
  float sum;
};

static inline SumMax sumax_combine(SumMax a, SumMax b) {
  bool a_bigger = (a.maxval > b.maxval);
  SumMax bigger = a_bigger ? a : b;
  SumMax smaller = a_bigger ? b : a;
  SumMax r;
  r.maxval = bigger.maxval;
  r.sum = bigger.sum + smaller.sum * sycl::exp(smaller.maxval - bigger.maxval);
  return r;
}

// ---------------- kernel 1 : one row per work-item, 3 passes ----------------
class SoftmaxK1;
static void run_k1(sycl::queue &q, float *d_out, const float *d_inp, int N,
                   int C) {
  constexpr int B = 256;
  int g = ceil_div(N, B);
  q.submit([&](sycl::handler &h) {
    h.parallel_for<SoftmaxK1>(
        sycl::nd_range<1>(sycl::range<1>((size_t)g * B), sycl::range<1>(B)),
        [=](sycl::nd_item<1> it) {
          int i = (int)it.get_global_id(0);
          if (i < N) {
            const float *ir = d_inp + (size_t)i * (size_t)C;
            float *orr = d_out + (size_t)i * (size_t)C;
            float mx = kNegInf();
            for (int j = 0; j < C; ++j)
              if (ir[j] > mx) mx = ir[j];
            double s = 0.0;
            for (int j = 0; j < C; ++j) {
              orr[j] = sycl::exp(ir[j] - mx);
              s += (double)orr[j];
            }
            float inv = (float)(1.0 / s);
            for (int j = 0; j < C; ++j) orr[j] *= inv;
          }
        });
  });
  q.wait();
}

// ---------------- kernel 2 : one row per work-group, local-memory tree ------
// Variant-2 algorithm: coarsened max, whole-block tree reduce in local
// memory, exp, coarsened sum, whole-block tree reduce, normalize.
class SoftmaxK2;
static void run_k2(sycl::queue &q, float *d_out, const float *d_inp, int N,
                   int C) {
  constexpr int B = 256;
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> shm(sycl::range<1>(B), h);
    h.parallel_for<SoftmaxK2>(
        sycl::nd_range<1>(sycl::range<1>((size_t)N * B), sycl::range<1>(B)),
        [=](sycl::nd_item<1> it) {
          int idx = (int)it.get_group(0);
          int tid = (int)it.get_local_id(0);
          int bs = (int)it.get_local_range(0);
          if (idx >= N) return;
          sycl::group<1> grp = it.get_group();
          const float *x = d_inp + (size_t)idx * (size_t)C;
          float mx = kNegInf();
          for (int i = tid; i < C; i += bs) mx = sycl::fmax(mx, x[i]);
          shm[tid] = mx;
          for (int s = bs / 2; s >= 1; s /= 2) {
            sycl::group_barrier(grp);
            if (tid < s) shm[tid] = sycl::fmax(shm[tid], shm[tid + s]);
          }
          sycl::group_barrier(grp);
          float off = shm[0];
          for (int i = tid; i < C; i += bs)
            d_out[(size_t)idx * (size_t)C + i] = sycl::exp(x[i] - off);
          sycl::group_barrier(grp);
          const float *xo = d_out + (size_t)idx * (size_t)C;
          float sv = 0.0f;
          for (int i = tid; i < C; i += bs) sv += xo[i];
          shm[tid] = sv;
          for (int s = bs / 2; s >= 1; s /= 2) {
            sycl::group_barrier(grp);
            if (tid < s) shm[tid] += shm[tid + s];
          }
          sycl::group_barrier(grp);
          float sum = shm[0];
          for (int i = tid; i < C; i += bs)
            d_out[(size_t)idx * (size_t)C + i] = xo[i] / sum;
        });
  });
  q.wait();
}

// ---------------- kernel 3 : single warp (32) per row -----------------------
// Variant-3 algorithm: must use 32 threads per row; intra-warp reductions
// for max and sum (SYCL: local-memory tree over the 32-thread work-group)
// plus warp-style broadcast of the reduced value.
class SoftmaxK3;
static void run_k3(sycl::queue &q, float *d_out, const float *d_inp, int N,
                   int C) {
  constexpr int B = 32;
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> tmp(sycl::range<1>(B), h);
    h.parallel_for<SoftmaxK3>(
        sycl::nd_range<1>(sycl::range<1>((size_t)N * B), sycl::range<1>(B)),
        [=](sycl::nd_item<1> it) {
          int idx = (int)it.get_group(0);
          int tid = (int)it.get_local_id(0);
          int bs = (int)it.get_local_range(0);
          if (idx >= N) return;
          sycl::group<1> grp = it.get_group();
          const float *x = d_inp + (size_t)idx * (size_t)C;
          // thread coarsening + intra-warp max (warp = whole group here)
          float mx = kNegInf();
          for (int i = tid; i < C; i += bs) mx = sycl::fmax(mx, x[i]);
          tmp[tid] = mx;
          for (int s = bs / 2; s >= 1; s /= 2) {
            sycl::group_barrier(grp);
            if (tid < s) tmp[tid] = sycl::fmax(tmp[tid], tmp[tid + s]);
          }
          sycl::group_barrier(grp);
          float off = tmp[0]; // broadcast warp max to all lanes
          for (int i = tid; i < C; i += bs)
            d_out[(size_t)idx * (size_t)C + i] = sycl::exp(x[i] - off);
          sycl::group_barrier(grp);
          const float *xo = d_out + (size_t)idx * (size_t)C;
          float sv = 0.0f;
          for (int i = tid; i < C; i += bs) sv += xo[i];
          tmp[tid] = sv;
          for (int s = bs / 2; s >= 1; s /= 2) {
            sycl::group_barrier(grp);
            if (tid < s) tmp[tid] += tmp[tid + s];
          }
          sycl::group_barrier(grp);
          float tot = tmp[0]; // broadcast warp sum
          for (int i = tid; i < C; i += bs)
            d_out[(size_t)idx * (size_t)C + i] = xo[i] / tot;
        });
  });
  q.wait();
}

// ---------------- kernel 4 : warps of 32 + inter-warp shared ----------------
// Variant-4 algorithm: any block size multiple of 32; per-thread coarsening,
// intra-warp reductions, shared-memory inter-warp fan-in by thread 0,
// broadcast. SYCL: logical warps of 32 emulated with local memory so the
// result is independent of the hardware sub-group size.
class SoftmaxK4;
static void run_k4(sycl::queue &q, float *d_out, const float *d_inp, int N,
                   int C) {
  constexpr int B = 256;
  constexpr int kWarp = 32;
  constexpr int kWarps = B / kWarp;
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> tmp(sycl::range<1>(B), h);
    sycl::local_accessor<float, 1> inter(sycl::range<1>(kWarps), h);
    h.parallel_for<SoftmaxK4>(
        sycl::nd_range<1>(sycl::range<1>((size_t)N * B), sycl::range<1>(B)),
        [=](sycl::nd_item<1> it) {
          int idx = (int)it.get_group(0);
          int tid = (int)it.get_local_id(0);
          int bs = (int)it.get_local_range(0);
          if (idx >= N) return;
          sycl::group<1> grp = it.get_group();
          int warp = tid / kWarp;
          int lane = tid % kWarp;
          const float *x = d_inp + (size_t)idx * (size_t)C;
          float mx = kNegInf();
          for (int i = tid; i < C; i += bs) mx = sycl::fmax(mx, x[i]);
          // intra-warp max via local tree over the 32 logical lanes
          tmp[tid] = mx;
          for (int s = kWarp / 2; s >= 1; s /= 2) {
            sycl::group_barrier(grp);
            if (lane < s)
              tmp[(size_t)warp * kWarp + lane] = sycl::fmax(
                  tmp[(size_t)warp * kWarp + lane],
                  tmp[(size_t)warp * kWarp + lane + s]);
          }
          sycl::group_barrier(grp);
          float wmax = tmp[(size_t)warp * kWarp];
          if (lane == 0) inter[warp] = wmax;
          sycl::group_barrier(grp);
          if (tid == 0) {
            float v = inter[0];
            for (int i = 1; i < kWarps; ++i) v = sycl::fmax(v, inter[i]);
            inter[0] = v;
          }
          sycl::group_barrier(grp);
          float off = inter[0];
          for (int i = tid; i < C; i += bs)
            d_out[(size_t)idx * (size_t)C + i] = sycl::exp(x[i] - off);
          sycl::group_barrier(grp);
          const float *xo = d_out + (size_t)idx * (size_t)C;
          float sv = 0.0f;
          for (int i = tid; i < C; i += bs) sv += xo[i];
          tmp[tid] = sv;
          for (int s = kWarp / 2; s >= 1; s /= 2) {
            sycl::group_barrier(grp);
            if (lane < s)
              tmp[(size_t)warp * kWarp + lane] +=
                  tmp[(size_t)warp * kWarp + lane + s];
          }
          sycl::group_barrier(grp);
          float wsum = tmp[(size_t)warp * kWarp];
          if (lane == 0) inter[warp] = wsum;
          sycl::group_barrier(grp);
          if (tid == 0) {
            float v = inter[0];
            for (int i = 1; i < kWarps; ++i) v += inter[i];
            inter[0] = v;
          }
          sycl::group_barrier(grp);
          float sum = inter[0];
          for (int i = tid; i < C; i += bs)
            d_out[(size_t)idx * (size_t)C + i] = xo[i] / sum;
        });
  });
  q.wait();
}

// ---------------- kernel 5 : online, one row per work-item ------------------
class SoftmaxK5;
static void run_k5(sycl::queue &q, float *d_out, const float *d_inp, int N,
                   int C) {
  constexpr int B = 256;
  int g = ceil_div(N, B);
  q.submit([&](sycl::handler &h) {
    h.parallel_for<SoftmaxK5>(
        sycl::nd_range<1>(sycl::range<1>((size_t)g * B), sycl::range<1>(B)),
        [=](sycl::nd_item<1> it) {
          int i = (int)it.get_global_id(0);
          if (i < N) {
            const float *ir = d_inp + (size_t)i * (size_t)C;
            float *orr = d_out + (size_t)i * (size_t)C;
            float mx = kNegInf();
            double s = 0.0;
            for (int j = 0; j < C; ++j) {
              float prev = mx;
              float cur = ir[j];
              if (cur > mx) {
                mx = cur;
                s = s * (double)sycl::exp(prev - mx) +
                    (double)sycl::exp(cur - mx);
              } else {
                s += (double)sycl::exp(cur - mx);
              }
            }
            for (int j = 0; j < C; ++j)
              orr[j] = sycl::exp(ir[j] - mx) / (float)s;
          }
        });
  });
  q.wait();
}

// ---------------- kernel 6 : online, one row per warp -----------------------
// Variant-6 algorithm: per-lane SumMax coarsening + cooperative warp
// reduction with the online combine operator. SYCL: logical warps of 32,
// one row per warp, reduction via local-memory tree with sumax_combine.
class SoftmaxK6;
static void run_k6(sycl::queue &q, float *d_out, const float *d_inp, int N,
                   int C) {
  constexpr int B = 256;
  constexpr int kWarp = 32;
  constexpr int kWarps = B / kWarp;
  int g = ceil_div(N, kWarps);
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<SumMax, 1> tmp(sycl::range<1>(B), h);
    h.parallel_for<SoftmaxK6>(
        sycl::nd_range<1>(sycl::range<1>((size_t)g * B), sycl::range<1>(B)),
        [=](sycl::nd_item<1> it) {
          int warp = (int)it.get_local_id(0) / kWarp;
          int lane = (int)it.get_local_id(0) % kWarp;
          int tid = (int)it.get_local_id(0);
          size_t row = it.get_group(0) * (size_t)kWarps + (size_t)warp;
          sycl::group<1> grp = it.get_group();
          if (row >= (size_t)N) {
            tmp[tid] = SumMax{kNegInf(), 0.0f};
            sycl::group_barrier(grp);
            return;
          }
          const float *x = d_inp + row * (size_t)C;
          SumMax part{kNegInf(), 0.0f};
          for (int i = lane; i < C; i += kWarp) {
            SumMax e{x[i], 1.0f};
            part = sumax_combine(part, e);
          }
          tmp[tid] = part;
          for (int s = kWarp / 2; s >= 1; s /= 2) {
            sycl::group_barrier(grp);
            if (lane < s)
              tmp[(size_t)(warp * kWarp + lane)] = sumax_combine(
                  tmp[(size_t)(warp * kWarp + lane)],
                  tmp[(size_t)(warp * kWarp + lane + s)]);
          }
          sycl::group_barrier(grp);
          SumMax tot = tmp[(size_t)warp * kWarp];
          for (int i = lane; i < C; i += kWarp)
            d_out[row * (size_t)C + i] =
                sycl::exp(x[i] - tot.maxval) / tot.sum;
        });
  });
  q.wait();
}

// ---------------- kernel 7 : large-C unrolled + warps + shared --------------
// Variant-7 algorithm: UNROLL_FACTOR register blocking with clamped
// redundant reads, intra-warp max/sum, split shared storage for max and
// sum. SYCL: logical warps of 32 with local-memory trees.
class SoftmaxK7;
static void run_k7(sycl::queue &q, float *d_out, const float *d_inp, int N,
                   int C) {
  constexpr int B = 256;
  constexpr int kWarp = 32;
  constexpr int kWarps = B / kWarp;
  constexpr int UF = 8;
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> tmp(sycl::range<1>(B), h);
    sycl::local_accessor<float, 1> maxs(sycl::range<1>(kWarps), h);
    sycl::local_accessor<float, 1> sums(sycl::range<1>(kWarps), h);
    h.parallel_for<SoftmaxK7>(
        sycl::nd_range<1>(sycl::range<1>((size_t)N * B), sycl::range<1>(B)),
        [=](sycl::nd_item<1> it) {
          int idx = (int)it.get_group(0);
          int tid = (int)it.get_local_id(0);
          int bs = (int)it.get_local_range(0);
          if (idx >= N) return;
          sycl::group<1> grp = it.get_group();
          int warp = tid / kWarp;
          int lane = tid % kWarp;
          const float *x = d_inp + (size_t)idx * (size_t)C;
          float *y = d_out + (size_t)idx * (size_t)C;
          auto clamp_at = [&](int v) { return v < C - 1 ? v : C - 1; };
          float mx = kNegInf();
          if (tid < C) {
            for (int i = tid; i < C; i += bs * UF) {
#pragma unroll
              for (int u = 0; u < UF; ++u)
                mx = sycl::fmax(mx, x[clamp_at(i + u * bs)]);
            }
          }
          tmp[tid] = mx;
          for (int s = kWarp / 2; s >= 1; s /= 2) {
            sycl::group_barrier(grp);
            if (lane < s)
              tmp[(size_t)warp * kWarp + lane] = sycl::fmax(
                  tmp[(size_t)warp * kWarp + lane],
                  tmp[(size_t)warp * kWarp + lane + s]);
          }
          sycl::group_barrier(grp);
          if (lane == 0) maxs[warp] = tmp[(size_t)warp * kWarp];
          sycl::group_barrier(grp);
          if (tid == 0) {
            float v = maxs[0];
#pragma unroll
            for (int i = 1; i < kWarps; ++i) v = sycl::fmax(v, maxs[i]);
            maxs[0] = v;
          }
          sycl::group_barrier(grp);
          float off = maxs[0];
          float sv = 0.0f;
          if (tid < C) {
            for (int i = tid; i < C; i += bs * UF) {
              float regs[UF];
#pragma unroll
              for (int u = 0; u < UF; ++u) regs[u] = x[clamp_at(i + u * bs)];
#pragma unroll
              for (int u = 0; u < UF; ++u) {
                if (i + u * bs < C) {
                  float o = sycl::exp(regs[u] - off);
                  y[clamp_at(i + u * bs)] = o;
                  sv += o;
                }
              }
            }
          }
          tmp[tid] = sv;
          for (int s = kWarp / 2; s >= 1; s /= 2) {
            sycl::group_barrier(grp);
            if (lane < s)
              tmp[(size_t)warp * kWarp + lane] +=
                  tmp[(size_t)warp * kWarp + lane + s];
          }
          sycl::group_barrier(grp);
          if (lane == 0) sums[warp] = tmp[(size_t)warp * kWarp];
          sycl::group_barrier(grp);
          if (tid == 0) {
            float v = sums[0];
#pragma unroll
            for (int i = 1; i < kWarps; ++i) v += sums[i];
            sums[0] = v;
          }
          sycl::group_barrier(grp);
          float sum = sums[0];
          if (tid < C) {
            for (int i = tid; i < C; i += bs * UF) {
              float regs[UF];
#pragma unroll
              for (int u = 0; u < UF; ++u) regs[u] = y[clamp_at(i + u * bs)];
#pragma unroll
              for (int u = 0; u < UF; ++u) {
                if (i + u * bs < C) y[i + u * bs] = regs[u] / sum;
              }
            }
          }
        });
  });
  q.wait();
}

// ---------------- kernel 8 : online, one row per warp, warp merge -----------
// Variant-8 algorithm: per-lane online max/sum coarsening then the online
// warp-merge reduction. SYCL: logical warps of 32, one row per warp,
// merge emulated with local memory and explicit out-of-range handling
// matching shfl_down semantics (OOB lane keeps its own value; only lane 0
// is broadcast at the end).
class SoftmaxK8;
static void run_k8(sycl::queue &q, float *d_out, const float *d_inp, int N,
                   int C) {
  constexpr int B = 256;
  constexpr int kWarp = 32;
  constexpr int kWarps = B / kWarp;
  int g = ceil_div(N, kWarps);
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> lmax(sycl::range<1>(B), h);
    sycl::local_accessor<float, 1> lsum(sycl::range<1>(B), h);
    h.parallel_for<SoftmaxK8>(
        sycl::nd_range<1>(sycl::range<1>((size_t)g * B), sycl::range<1>(B)),
        [=](sycl::nd_item<1> it) {
          int warp = (int)it.get_local_id(0) / kWarp;
          int lane = (int)it.get_local_id(0) % kWarp;
          int tid = (int)it.get_local_id(0);
          size_t row = it.get_group(0) * (size_t)kWarps + (size_t)warp;
          sycl::group<1> grp = it.get_group();
          if (row >= (size_t)N) {
            lmax[tid] = kNegInf();
            lsum[tid] = 0.0f;
            sycl::group_barrier(grp);
            return;
          }
          const float *x = d_inp + row * (size_t)C;
          float *y = d_out + row * (size_t)C;
          float mx = kNegInf(), sv = 0.0f;
          for (int i = lane; i < C; i += kWarp) {
            float bigger = sycl::fmax(mx, x[i]);
            sv = sv * sycl::exp(mx - bigger) + sycl::exp(x[i] - bigger);
            mx = bigger;
          }
          size_t base = (size_t)warp * kWarp;
          lmax[tid] = mx;
          lsum[tid] = sv;
          sycl::group_barrier(grp);
          for (int off = kWarp / 2; off > 0; off >>= 1) {
            float cur_mx = lmax[base + lane];
            float cur_sv = lsum[base + lane];
            float omx, osv;
            if (lane + off < kWarp) {
              omx = lmax[base + lane + off];
              osv = lsum[base + lane + off];
            } else {
              omx = cur_mx;
              osv = cur_sv;
            }
            sycl::group_barrier(grp);
            float nmx, nsv;
            if (omx > cur_mx) {
              nsv = cur_sv * sycl::exp(cur_mx - omx) + osv;
              nmx = omx;
            } else {
              osv *= sycl::exp(omx - cur_mx);
              nsv = cur_sv + osv;
              nmx = cur_mx;
            }
            lmax[base + lane] = nmx;
            lsum[base + lane] = nsv;
            sycl::group_barrier(grp);
          }
          mx = lmax[base + 0];
          sv = lsum[base + 0];
          for (int i = lane; i < C; i += kWarp)
            y[i] = sycl::exp(x[i] - mx) / sv;
        });
  });
  q.wait();
}

int main(int argc, char **argv) {
  if (argc != 6) {
    std::fprintf(stderr, "usage: %s <input> <output> B T V\n", argv[0]);
    return 2;
  }
  const int B = std::atoi(argv[3]);
  const int T = std::atoi(argv[4]);
  const int V = std::atoi(argv[5]);
  size_t N = (size_t)B * (size_t)T;
  size_t C = (size_t)V;
  std::FILE *fin = std::fopen(argv[1], "rb");
  if (!fin) {
    std::perror(argv[1]);
    return 2;
  }
  std::vector<float> inp(N * C);
  if (std::fread(inp.data(), sizeof(float), N * C, fin) != N * C) {
    std::fprintf(stderr, "input too short at inp\n");
    return 2;
  }
  std::vector<int> outliers(N * 3);
  if (std::fread(outliers.data(), sizeof(int), N * 3, fin) != N * 3) {
    std::fprintf(stderr, "input too short at outliers\n");
    return 2;
  }
  std::fclose(fin);
  for (int k = 0; k < 3; ++k)
    for (size_t j = 0; j < N; ++j) inp[j * C + outliers[j * 3 + k]] *= 20.0f;

  sycl::queue q{sycl::default_selector_v};
  float *d_inp = sycl::malloc_device<float>(N * C, q);
  float *d_out = sycl::malloc_device<float>(N * C, q);
  if (!d_inp || !d_out) {
    std::fprintf(stderr, "device alloc failed\n");
    return 1;
  }
  q.memcpy(d_inp, inp.data(), sizeof(float) * N * C).wait();
  std::vector<float> out(N * C);
  std::FILE *fout = std::fopen(argv[2], "wb");
  if (!fout) {
    std::perror(argv[2]);
    return 2;
  }
  const int variants[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  for (int vi = 0; vi < 8; ++vi) {
    q.memset(d_out, 0, sizeof(float) * N * C).wait();
    switch (variants[vi]) {
      case 1: run_k1(q, d_out, d_inp, (int)N, (int)C); break;
      case 2: run_k2(q, d_out, d_inp, (int)N, (int)C); break;
      case 3: run_k3(q, d_out, d_inp, (int)N, (int)C); break;
      case 4: run_k4(q, d_out, d_inp, (int)N, (int)C); break;
      case 5: run_k5(q, d_out, d_inp, (int)N, (int)C); break;
      case 6: run_k6(q, d_out, d_inp, (int)N, (int)C); break;
      case 7: run_k7(q, d_out, d_inp, (int)N, (int)C); break;
      case 8: run_k8(q, d_out, d_inp, (int)N, (int)C); break;
    }
    q.memcpy(out.data(), d_out, sizeof(float) * N * C).wait();
    std::fwrite(out.data(), sizeof(float), N * C, fout);
  }
  std::fclose(fout);
  sycl::free(d_inp, q);
  sycl::free(d_out, q);
  return 0;
}
