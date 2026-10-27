#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <algorithm>

template <class T> T ceil_div(T a, T b) { return (a + b - 1) / b; }

struct SumMax {
  float maxval;
  float sum;
};
struct SumMaxOp {
  SumMax operator()(const SumMax &a, const SumMax &b) const {
    bool a_bigger = (a.maxval > b.maxval);
    const SumMax &big = a_bigger ? a : b;
    const SumMax &small = a_bigger ? b : a;
    SumMax r;
    r.maxval = big.maxval;
    r.sum = big.sum + small.sum * sycl::exp(small.maxval - big.maxval);
    return r;
  }
};

// ---- variant 1: one work-item per row, sequential loops ----
class SoftmaxK1;
void run_k1(sycl::queue &q, float *out, const float *inp, int N, int C,
            int block) {
  int grid = ceil_div(N, block);
  q.parallel_for<SoftmaxK1>(
      sycl::nd_range<1>(sycl::range<1>(grid * block), sycl::range<1>(block)),
      [=](sycl::nd_item<1> it) {
        int i = (int)it.get_global_id(0);
        if (i < N) {
          const float *row_in = inp + (size_t)i * C;
          float *row_out = out + (size_t)i * C;
          float maxval = -INFINITY;
          for (int j = 0; j < C; ++j)
            if (row_in[j] > maxval)
              maxval = row_in[j];
          double sum = 0.0;
          for (int j = 0; j < C; ++j) {
            row_out[j] = sycl::exp(row_in[j] - maxval);
            sum += row_out[j];
          }
          float inv = (float)(1.0 / sum);
          for (int j = 0; j < C; ++j)
            row_out[j] *= inv;
        }
      });
}

// ---- variant 2: one work-group per row, local-memory tree reductions ----
class SoftmaxK2;
void run_k2(sycl::queue &q, float *out, const float *inp, int N, int C,
            int block) {
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> sh(sycl::range<1>(block), h);
    h.parallel_for<SoftmaxK2>(
        sycl::nd_range<1>(sycl::range<1>((size_t)N * block),
                          sycl::range<1>(block)),
        [=](sycl::nd_item<1> it) {
          int idx = (int)it.get_group(0);
          int tid = (int)it.get_local_id(0);
          int bs = (int)it.get_local_range(0);
          const float *x = inp + (size_t)idx * C;
          float maxval = -INFINITY;
          for (int i = tid; i < C; i += bs)
            maxval = sycl::fmax(maxval, x[i]);
          sh[tid] = maxval;
          for (int s = bs / 2; s >= 1; s /= 2) {
            sycl::group_barrier(it.get_group());
            if (tid < s)
              sh[tid] = sycl::fmax(sh[tid], sh[tid + s]);
          }
          sycl::group_barrier(it.get_group());
          float offset = sh[0];
          for (int i = tid; i < C; i += bs)
            out[(size_t)idx * C + i] = sycl::exp(x[i] - offset);
          sycl::group_barrier(it.get_group());
          const float *xo = out + (size_t)idx * C;
          float sumval = 0.0f;
          for (int i = tid; i < C; i += bs)
            sumval += xo[i];
          sh[tid] = sumval;
          for (int s = bs / 2; s >= 1; s /= 2) {
            sycl::group_barrier(it.get_group());
            if (tid < s)
              sh[tid] += sh[tid + s];
          }
          sycl::group_barrier(it.get_group());
          float sum = sh[0];
          for (int i = tid; i < C; i += bs)
            out[(size_t)idx * C + i] = xo[i] / sum;
        });
  });
}

// ---- variant 3: one work-group (32) per row, work-group reductions ----
class SoftmaxK3;
void run_k3(sycl::queue &q, float *out, const float *inp, int N, int C,
            int /*block*/) {
  const int bs = 32;
  q.parallel_for<SoftmaxK3>(
      sycl::nd_range<1>(sycl::range<1>((size_t)N * bs), sycl::range<1>(bs)),
      [=](sycl::nd_item<1> it) {
        sycl::group<1> grp = it.get_group();
        int idx = (int)it.get_group(0);
        int tid = (int)it.get_local_id(0);
        int bl = (int)it.get_local_range(0);
        const float *x = inp + (size_t)idx * C;
        float part = -INFINITY;
        for (int i = tid; i < C; i += bl)
          part = sycl::fmax(part, x[i]);
        float offset =
            sycl::reduce_over_group(grp, part, sycl::maximum<float>());
        for (int i = tid; i < C; i += bl)
          out[(size_t)idx * C + i] = sycl::exp(x[i] - offset);
        const float *xo = out + (size_t)idx * C;
        float s = 0.0f;
        for (int i = tid; i < C; i += bl)
          s += xo[i];
        float sum = sycl::reduce_over_group(grp, s, sycl::plus<float>());
        for (int i = tid; i < C; i += bl)
          out[(size_t)idx * C + i] = xo[i] / sum;
      });
}

// ---- variant 4: sub-group + local inter-warp reduction, any block ----
class SoftmaxK4;
void run_k4(sycl::queue &q, float *out, const float *inp, int N, int C,
            int block) {
  q.submit([&](sycl::handler &h) {
    // enough for up to 64 sub-groups per work-group
    sycl::local_accessor<float, 1> sh(sycl::range<1>(64), h);
    h.parallel_for<SoftmaxK4>(
        sycl::nd_range<1>(sycl::range<1>((size_t)N * block),
                          sycl::range<1>(block)),
        [=](sycl::nd_item<1> it) {
          sycl::sub_group sg = it.get_sub_group();
          int idx = (int)it.get_group(0);
          int tid = (int)it.get_local_id(0);
          int bl = (int)it.get_local_range(0);
          int nsg = (int)sg.get_group_range()[0];
          int sgid = (int)sg.get_group_id()[0];
          int lane = (int)sg.get_local_id()[0];
          const float *x = inp + (size_t)idx * C;
          float maxval = -INFINITY;
          for (int i = tid; i < C; i += bl)
            maxval = sycl::fmax(maxval, x[i]);
          float sgmax =
              sycl::reduce_over_group(sg, maxval, sycl::maximum<float>());
          if (lane == 0)
            sh[sgid] = sgmax;
          sycl::group_barrier(it.get_group());
          if (tid == 0) {
            float v = sh[0];
            for (int i = 1; i < nsg; ++i)
              v = sycl::fmax(v, sh[i]);
            sh[0] = v;
          }
          sycl::group_barrier(it.get_group());
          float offset = sh[0];
          for (int i = tid; i < C; i += bl)
            out[(size_t)idx * C + i] = sycl::exp(x[i] - offset);
          const float *xo = out + (size_t)idx * C;
          float sumval = 0.0f;
          for (int i = tid; i < C; i += bl)
            sumval += xo[i];
          float sgsum = sycl::reduce_over_group(sg, sumval, sycl::plus<float>());
          if (lane == 0)
            sh[sgid] = sgsum;
          sycl::group_barrier(it.get_group());
          if (tid == 0) {
            float v = sh[0];
            for (int i = 1; i < nsg; ++i)
              v += sh[i];
            sh[0] = v;
          }
          sycl::group_barrier(it.get_group());
          float sum = sh[0];
          for (int i = tid; i < C; i += bl)
            out[(size_t)idx * C + i] = xo[i] / sum;
        });
  });
}

// ---- variant 5: online, one work-item per row ----
class SoftmaxK5;
void run_k5(sycl::queue &q, float *out, const float *inp, int N, int C,
            int block) {
  int grid = ceil_div(N, block);
  q.parallel_for<SoftmaxK5>(
      sycl::nd_range<1>(sycl::range<1>(grid * block), sycl::range<1>(block)),
      [=](sycl::nd_item<1> it) {
        int i = (int)it.get_global_id(0);
        if (i < N) {
          const float *row_in = inp + (size_t)i * C;
          float *row_out = out + (size_t)i * C;
          float maxval = -INFINITY;
          double sum = 0.0;
          for (int j = 0; j < C; ++j) {
            float prev = maxval;
            float cur = row_in[j];
            if (cur > maxval) {
              maxval = cur;
              sum = sum * sycl::exp(prev - maxval) + sycl::exp(cur - maxval);
            } else {
              sum += sycl::exp(cur - maxval);
            }
          }
          float inv = (float)(1.0 / sum);
          for (int j = 0; j < C; ++j)
            row_out[j] = sycl::exp(row_in[j] - maxval) * inv;
        }
      });
}

// ---- variant 6: online, one sub-group per row with SumMax reduction ----
class SoftmaxK6;
void run_k6(sycl::queue &q, float *out, const float *inp, int N, int C,
            int block) {
  const int warp = 32;
  int grid = ceil_div(N * warp, block);
  q.parallel_for<SoftmaxK6>(
      sycl::nd_range<1>(sycl::range<1>((size_t)grid * block),
                        sycl::range<1>(block)),
      [=](sycl::nd_item<1> it) {
        sycl::sub_group sg = it.get_sub_group();
        int nsg = (int)sg.get_group_range()[0];
        int sgid = (int)sg.get_group_id()[0];
        int idx = (int)it.get_group(0) * nsg + sgid;
        if (idx >= N)
          return;
        const float *x = inp + (size_t)idx * C;
        SumMax part{-INFINITY, 0.0f};
        int lane = (int)sg.get_local_id()[0];
        int ssize = (int)sg.get_local_range()[0];
        for (int i = lane; i < C; i += ssize) {
          SumMax cur{x[i], 1.0f};
          part = SumMaxOp{}(part, cur);
        }
        // sub-group reduction with the online combine op (mirrors cg::reduce)
        SumMax total = part;
        for (int off = ssize / 2; off > 0; off >>= 1) {
          SumMax other{sycl::shift_group_left(sg, total.maxval, (size_t)off),
                       sycl::shift_group_left(sg, total.sum, (size_t)off)};
          total = SumMaxOp{}(total, other);
        }
        total.maxval = sycl::select_from_group(sg, total.maxval, 0);
        total.sum = sycl::select_from_group(sg, total.sum, 0);
        for (int i = lane; i < C; i += ssize)
          out[(size_t)idx * C + i] =
              sycl::exp(x[i] - total.maxval) / total.sum;
      });
}

// ---- variant 7: large-C unrolled, sub-group + split local storage ----
class SoftmaxK7;
void run_k7(sycl::queue &q, float *out, const float *inp, int N, int C,
            int block) {
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> sh(sycl::range<1>(128), h);
    h.parallel_for<SoftmaxK7>(
        sycl::nd_range<1>(sycl::range<1>((size_t)N * block),
                          sycl::range<1>(block)),
        [=](sycl::nd_item<1> it) {
          constexpr int UNROLL = 8;
          sycl::sub_group sg = it.get_sub_group();
          int idx = (int)it.get_group(0);
          int tid = (int)it.get_local_id(0);
          int bl = (int)it.get_local_range(0);
          int nsg = (int)sg.get_group_range()[0];
          int sgid = (int)sg.get_group_id()[0];
          int lane = (int)sg.get_local_id()[0];
          // sh[0..63] = maxvals, sh[64..127] = sumvals
          if (tid >= C) {
            sh[sgid] = -INFINITY;
            sh[(size_t)64 + sgid] = 0.0f;
            return;
          }
          const float *x = inp + (size_t)idx * C;
          float *y = out + (size_t)idx * C;
          float maxval = -INFINITY;
          for (int i = tid; i < C; i += bl * UNROLL) {
#pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
              int j = i + u * bl;
              if (j >= C)
                j = C - 1;
              maxval = sycl::fmax(maxval, x[j]);
            }
          }
          float sgmax =
              sycl::reduce_over_group(sg, maxval, sycl::maximum<float>());
          if (lane == 0)
            sh[sgid] = sgmax;
          sycl::group_barrier(it.get_group());
          if (tid == 0) {
            float v = sh[0];
            for (int i = 1; i < nsg; ++i)
              v = sycl::fmax(v, sh[i]);
            sh[0] = v;
          }
          sycl::group_barrier(it.get_group());
          float offset = sh[0];
          float sumval = 0.0f;
          for (int i = tid; i < C; i += bl * UNROLL) {
            float regs[UNROLL];
#pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
              int j = i + u * bl;
              if (j >= C)
                j = C - 1;
              regs[u] = x[j];
            }
#pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
              int j = i + u * bl;
              if (j < C) {
                float o = sycl::exp(regs[u] - offset);
                int wj = j;
                if (wj >= C)
                  wj = C - 1;
                y[wj] = o;
                sumval += o;
              }
            }
          }
          float sgsum =
              sycl::reduce_over_group(sg, sumval, sycl::plus<float>());
          if (lane == 0)
            sh[(size_t)64 + sgid] = sgsum;
          sycl::group_barrier(it.get_group());
          if (tid == 0) {
            float v = sh[64];
            for (int i = 1; i < nsg; ++i)
              v += sh[(size_t)64 + i];
            sh[64] = v;
          }
          sycl::group_barrier(it.get_group());
          float sum = sh[64];
          for (int i = tid; i < C; i += bl * UNROLL) {
            float regs[UNROLL];
#pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
              int j = i + u * bl;
              if (j >= C)
                j = C - 1;
              regs[u] = y[j];
            }
#pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
              int j = i + u * bl;
              if (j < C)
                y[j] = regs[u] / sum;
            }
          }
        });
  });
}

// ---- variant 8: online fused, one sub-group per row, manual rescaling tree
class SoftmaxK8;
void run_k8(sycl::queue &q, float *out, const float *inp, int N, int C,
            int block) {
  const int warp = 32;
  int grid = ceil_div(N * warp, block);
  q.parallel_for<SoftmaxK8>(
      sycl::nd_range<1>(sycl::range<1>((size_t)grid * block),
                        sycl::range<1>(block)),
      [=](sycl::nd_item<1> it) {
        sycl::sub_group sg = it.get_sub_group();
        int tid = (int)it.get_local_id(0);
        if (tid >= C)
          return;
        int nsg = (int)sg.get_group_range()[0];
        int sgid = (int)sg.get_group_id()[0];
        int row = (int)it.get_group(0) * nsg + sgid;
        if (row >= N)
          return;
        int lane = (int)sg.get_local_id()[0];
        int ssize = (int)sg.get_local_range()[0];
        const float *x = inp + (size_t)row * C;
        float *y = out + (size_t)row * C;
        float maxval = -INFINITY, sumval = 0.0f;
        for (int i = lane; i < C; i += ssize) {
          float bigger = sycl::fmax(maxval, x[i]);
          sumval = sumval * sycl::exp(maxval - bigger) +
                   sycl::exp(x[i] - bigger);
          maxval = bigger;
        }
        for (int off = ssize / 2; off > 0; off >>= 1) {
          float oMax = sycl::shift_group_left(sg, maxval, (size_t)off);
          float oSum = sycl::shift_group_left(sg, sumval, (size_t)off);
          if (oMax > maxval) {
            sumval *= sycl::exp(maxval - oMax);
            maxval = oMax;
          } else {
            oSum *= sycl::exp(oMax - maxval);
          }
          sumval += oSum;
        }
        maxval = sycl::select_from_group(sg, maxval, 0);
        sumval = sycl::select_from_group(sg, sumval, 0);
        for (int i = lane; i < C; i += ssize)
          y[i] = sycl::exp(x[i] - maxval) / sumval;
      });
}

int main(int argc, char **argv) {
  if (argc != 6) {
    fprintf(stderr, "usage: %s <input> <output> B T V\n", argv[0]);
    return 2;
  }
  int B = atoi(argv[3]);
  int T = atoi(argv[4]);
  int V = atoi(argv[5]);
  size_t N = (size_t)B * T;
  size_t C = (size_t)V;
  size_t NV = N * C;
  FILE *fin = fopen(argv[1], "rb");
  if (!fin) {
    perror(argv[1]);
    return 2;
  }
  float *inp = (float *)malloc(sizeof(float) * NV);
  if (fread(inp, sizeof(float), NV, fin) != NV) {
    fprintf(stderr, "input too short at inp\n");
    return 2;
  }
  int *outliers = (int *)malloc(sizeof(int) * (size_t)(B * T * 3));
  if (fread(outliers, sizeof(int), (size_t)(B * T * 3), fin) !=
      (size_t)(B * T * 3)) {
    fprintf(stderr, "input too short at outliers\n");
    return 2;
  }
  fclose(fin);
  for (int k = 0; k < 3; ++k)
    for (int j = 0; j < B * T; ++j)
      inp[(size_t)j * V + outliers[(size_t)j * 3 + k]] *= 20;
  free(outliers);

  sycl::queue q;
  float *d_inp = sycl::malloc_device<float>(NV, q);
  float *d_out = sycl::malloc_device<float>(NV, q);
  float *out = (float *)malloc(sizeof(float) * NV);
  q.memcpy(d_inp, inp, sizeof(float) * NV).wait();

  const int variants[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  FILE *fout = fopen(argv[2], "wb");
  if (!fout) {
    perror(argv[2]);
    return 2;
  }
  for (int vi = 0; vi < 8; ++vi) {
    int k = variants[vi];
    q.memset(d_out, 0, sizeof(float) * NV).wait();
    switch (k) {
    case 1:
      run_k1(q, d_out, d_inp, (int)N, (int)C, 256);
      break;
    case 2:
      run_k2(q, d_out, d_inp, (int)N, (int)C, 256);
      break;
    case 3:
      run_k3(q, d_out, d_inp, (int)N, (int)C, 256);
      break;
    case 4:
      run_k4(q, d_out, d_inp, (int)N, (int)C, 256);
      break;
    case 5:
      run_k5(q, d_out, d_inp, (int)N, (int)C, 256);
      break;
    case 6:
      run_k6(q, d_out, d_inp, (int)N, (int)C, 256);
      break;
    case 7:
      run_k7(q, d_out, d_inp, (int)N, (int)C, 256);
      break;
    case 8:
      run_k8(q, d_out, d_inp, (int)N, (int)C, 256);
      break;
    }
    q.wait();
    q.memcpy(out, d_out, sizeof(float) * NV).wait();
    fwrite(out, sizeof(float), NV, fout);
  }
  fclose(fout);
  sycl::free(d_inp, q);
  sycl::free(d_out, q);
  free(inp);
  free(out);
  return 0;
}
