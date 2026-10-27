#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <vector>

template <typename T>
T ceil_div(T a, T b) { return (a + b - 1) / b; }

constexpr float kNegInf = std::numeric_limits<float>::infinity() * -1.0f;

// combine for online softmax partials (variant 6)
struct SumMax {
  float maxval;
  float sum;
};
inline SumMax combine_sum_max(SumMax a, SumMax b) {
  bool a_bigger = (a.maxval > b.maxval);
  SumMax bigger = a_bigger ? a : b;
  SumMax smaller = a_bigger ? b : a;
  SumMax r;
  r.maxval = bigger.maxval;
  r.sum = bigger.sum + smaller.sum * sycl::exp(smaller.maxval - bigger.maxval);
  return r;
}

// ---- variant 1: naive per-row per-thread ----
void softmax_forward1(sycl::queue &q, float *out, const float *inp, int N, int C,
                      int block_size) {
  int grid = (int)ceil_div(N, block_size);
  sycl::nd_range<1> rng(grid * block_size, block_size);
  q.submit([&](sycl::handler &h) {
    h.parallel_for(rng, [=](sycl::nd_item<1> it) {
      int i = (int)it.get_global_id(0);
      if (i < N) {
        const float *inp_row = inp + (size_t)i * (size_t)C;
        float *out_row = out + (size_t)i * (size_t)C;
        float maxval = kNegInf;
        for (int j = 0; j < C; ++j) {
          if (inp_row[j] > maxval) maxval = inp_row[j];
        }
        double sum = 0.0;
        for (int j = 0; j < C; ++j) {
          float e = sycl::exp(inp_row[j] - maxval);
          out_row[j] = e;
          sum += (double)e;
        }
        float inv = (float)(1.0 / sum);
        for (int j = 0; j < C; ++j) out_row[j] *= inv;
      }
    });
  });
  q.wait();
}

// ---- variant 2: fused per-row per-block, local-memory tree reduction ----
void softmax_forward2(sycl::queue &q, float *out, const float *inp, int N, int C,
                      int block_size) {
  sycl::nd_range<1> rng((size_t)N * (size_t)block_size, (size_t)block_size);
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> shared((size_t)block_size, h);
    h.parallel_for(rng, [=](sycl::nd_item<1> it) {
      int idx = (int)it.get_group(0);
      int tid = (int)it.get_local_id(0);
      int bs = (int)it.get_local_range(0);
      const float *x = inp + (size_t)idx * (size_t)C;
      float maxval = kNegInf;
      for (int i = tid; i < C; i += bs) maxval = sycl::fmax(maxval, x[i]);
      shared[tid] = maxval;
      for (int stride = bs / 2; stride >= 1; stride /= 2) {
        sycl::group_barrier(it.get_group());
        if (tid < stride) shared[tid] = sycl::fmax(shared[tid], shared[tid + stride]);
      }
      sycl::group_barrier(it.get_group());
      float offset = shared[0];
      for (int i = tid; i < C; i += bs) out[(size_t)idx * (size_t)C + i] = sycl::exp(x[i] - offset);
      sycl::group_barrier(it.get_group());
      const float *y = out + (size_t)idx * (size_t)C;
      float sumval = 0.0f;
      for (int i = tid; i < C; i += bs) sumval += y[i];
      shared[tid] = sumval;
      for (int stride = bs / 2; stride >= 1; stride /= 2) {
        sycl::group_barrier(it.get_group());
        if (tid < stride) shared[tid] += shared[tid + stride];
      }
      sycl::group_barrier(it.get_group());
      float sum = shared[0];
      for (int i = tid; i < C; i += bs) out[(size_t)idx * (size_t)C + i] = y[i] / sum;
    });
  });
  q.wait();
}

// ---- variant 3: intra-sub-group reductions, one group (32) per row ----
void softmax_forward3(sycl::queue &q, float *out, const float *inp, int N, int C,
                      int /*block_size*/) {
  const int bs = 32;
  sycl::nd_range<1> rng((size_t)N * (size_t)bs, (size_t)bs);
  q.submit([&](sycl::handler &h) {
    h.parallel_for(rng, [=](sycl::nd_item<1> it) {
      auto sg = it.get_sub_group();
      auto grp = it.get_group();
      int idx = (int)it.get_group(0);
      int tid = (int)it.get_local_id(0);
      int bl = (int)it.get_local_range(0);
      size_t sg_size = sg.get_local_range()[0];
      size_t num_sgs = (size_t)bl / sg_size;
      const float *x = inp + (size_t)idx * (size_t)C;
      float maxval = kNegInf;
      for (int i = tid; i < C; i += bl) maxval = sycl::fmax(maxval, x[i]);
      // intra-sub-group max, then fold across sub-groups via broadcast (no local)
      float sg_max = sycl::reduce_over_group(sg, maxval, sycl::maximum<float>{});
      float offset = sg_max;
      for (size_t s = 0; s < num_sgs; ++s) {
        float v = sycl::group_broadcast(grp, sg_max, s * sg_size);
        offset = sycl::fmax(offset, v);
      }
      for (int i = tid; i < C; i += bl) out[(size_t)idx * (size_t)C + i] = sycl::exp(x[i] - offset);
      const float *y = out + (size_t)idx * (size_t)C;
      float sumval = 0.0f;
      for (int i = tid; i < C; i += bl) sumval += y[i];
      float sg_sum = sycl::reduce_over_group(sg, sumval, sycl::plus<float>{});
      float sum = 0.0f;
      for (size_t s = 0; s < num_sgs; ++s) sum += sycl::group_broadcast(grp, sg_sum, s * sg_size);
      for (int i = tid; i < C; i += bl) out[(size_t)idx * (size_t)C + i] = y[i] / sum;
    });
  });
  q.wait();
}

// ---- variant 4: sub-group + local-memory inter-sub-group ----
void softmax_forward4(sycl::queue &q, float *out, const float *inp, int N, int C,
                      int block_size) {
  sycl::nd_range<1> rng((size_t)N * (size_t)block_size, (size_t)block_size);
  q.submit([&](sycl::handler &h) {
    // enough for any sub-group size (worst case local/1)
    sycl::local_accessor<float, 1> scratch((size_t)block_size, h);
    h.parallel_for(rng, [=](sycl::nd_item<1> it) {
      auto sg = it.get_sub_group();
      auto grp = it.get_group();
      int idx = (int)it.get_group(0);
      int tid = (int)it.get_local_id(0);
      int bl = (int)it.get_local_range(0);
      size_t sg_size = sg.get_local_range()[0];
      size_t num_sgs = (size_t)bl / sg_size;
      size_t sg_id = sg.get_group_id()[0];
      const float *x = inp + (size_t)idx * (size_t)C;
      float maxval = kNegInf;
      for (int i = tid; i < C; i += bl) maxval = sycl::fmax(maxval, x[i]);
      float sg_max = sycl::reduce_over_group(sg, maxval, sycl::maximum<float>{});
      if (sg.leader()) scratch[sg_id] = sg_max;
      sycl::group_barrier(grp);
      if (tid == 0) {
        float v = scratch[0];
        for (size_t i = 1; i < num_sgs; ++i) v = sycl::fmax(v, scratch[i]);
        scratch[0] = v;
      }
      sycl::group_barrier(grp);
      float offset = scratch[0];
      for (int i = tid; i < C; i += bl) out[(size_t)idx * (size_t)C + i] = sycl::exp(x[i] - offset);
      const float *y = out + (size_t)idx * (size_t)C;
      float sumval = 0.0f;
      for (int i = tid; i < C; i += bl) sumval += y[i];
      float sg_sum = sycl::reduce_over_group(sg, sumval, sycl::plus<float>{});
      if (sg.leader()) scratch[sg_id] = sg_sum;
      sycl::group_barrier(grp);
      if (tid == 0) {
        float v = scratch[0];
        for (size_t i = 1; i < num_sgs; ++i) v += scratch[i];
        scratch[0] = v;
      }
      sycl::group_barrier(grp);
      float sum = scratch[0];
      for (int i = tid; i < C; i += bl) out[(size_t)idx * (size_t)C + i] = y[i] / sum;
    });
  });
  q.wait();
}

// ---- variant 5: online naive per-row per-thread ----
void softmax_forward_online1(sycl::queue &q, float *out, const float *inp, int N, int C,
                             int block_size) {
  int grid = (int)ceil_div(N, block_size);
  sycl::nd_range<1> rng(grid * block_size, block_size);
  q.submit([&](sycl::handler &h) {
    h.parallel_for(rng, [=](sycl::nd_item<1> it) {
      int i = (int)it.get_global_id(0);
      if (i < N) {
        const float *inp_row = inp + (size_t)i * (size_t)C;
        float *out_row = out + (size_t)i * (size_t)C;
        float maxval = kNegInf;
        double sum = 0.0;
        for (int j = 0; j < C; ++j) {
          float prev = maxval;
          float cur = inp_row[j];
          if (cur > maxval) {
            maxval = cur;
            sum = sum * (double)sycl::exp(prev - maxval) + (double)sycl::exp(cur - maxval);
          } else {
            sum += (double)sycl::exp(cur - maxval);
          }
        }
        float inv = (float)(1.0 / sum);
        for (int j = 0; j < C; ++j) out_row[j] = sycl::exp(inp_row[j] - maxval) * inv;
      }
    });
  });
  q.wait();
}

// ---- variant 6: online, one row per sub-group with SumMax sub-group reduce ----
void softmax_forward_online2(sycl::queue &q, float *out, const float *inp, int N, int C,
                             int block_size) {
  int grid = (int)ceil_div(N * 32, block_size);
  sycl::nd_range<1> rng((size_t)grid * (size_t)block_size, (size_t)block_size);
  q.submit([&](sycl::handler &h) {
    h.parallel_for(rng, [=](sycl::nd_item<1> it) {
      auto sg = it.get_sub_group();
      size_t sg_size = sg.get_local_range()[0];
      size_t sg_lid = sg.get_local_id()[0];
      size_t sg_gid = sg.get_group_id()[0];
      size_t gid = (size_t)it.get_group(0);
      size_t sgs_per_group = (size_t)it.get_local_range(0) / sg_size;
      size_t idx = gid * sgs_per_group + sg_gid;
      if (idx >= (size_t)N) return;
      const float *x = inp + idx * (size_t)C;
      SumMax part{kNegInf, 0.0f};
      for (int i = (int)sg_lid; i < C; i += (int)sg_size) {
        SumMax cur{x[i], 1.0f};
        part = combine_sum_max(part, cur);
      }
      // sub-group reduction with xor permutes (commutative op)
      for (size_t off = sg_size / 2; off > 0; off >>= 1) {
        unsigned m = (unsigned)off;
        float o_max = sycl::permute_group_by_xor(sg, part.maxval, m);
        float o_sum = sycl::permute_group_by_xor(sg, part.sum, m);
        part = combine_sum_max(part, SumMax{o_max, o_sum});
      }
      float tot_max = sycl::group_broadcast(sg, part.maxval, 0);
      float tot_sum = sycl::group_broadcast(sg, part.sum, 0);
      for (int i = (int)sg_lid; i < C; i += (int)sg_size)
        out[idx * (size_t)C + i] = sycl::exp(x[i] - tot_max) / tot_sum;
    });
  });
  q.wait();
}

// ---- variant 7: optimized for very large C, unrolled, sub-group + local ----
void softmax_forward7(sycl::queue &q, float *out, const float *inp, int N, int C,
                      int block_size) {
  sycl::nd_range<1> rng((size_t)N * (size_t)block_size, (size_t)block_size);
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> shared((size_t)2 * (size_t)block_size, h);
    h.parallel_for(rng, [=](sycl::nd_item<1> it) {
      constexpr int UNROLL = 8;
      auto sg = it.get_sub_group();
      auto grp = it.get_group();
      int idx = (int)it.get_group(0);
      int tid = (int)it.get_local_id(0);
      int bl = (int)it.get_local_range(0);
      size_t sg_size = sg.get_local_range()[0];
      size_t num_sgs = (size_t)bl / sg_size;
      size_t sg_id = sg.get_group_id()[0];
      size_t lane = sg.get_local_id()[0];
      auto mp = shared.get_multi_ptr<sycl::access::decorated::no>();
      float *maxvals = mp.get();
      float *sumvals = mp.get() + block_size;
      if (tid >= C) {
        return;
      }
      const float *x = inp + (size_t)idx * (size_t)C;
      float *y = out + (size_t)idx * (size_t)C;
      auto clamp_min = [=](int v) { return v < (C - 1) ? v : (C - 1); };
      float maxval = kNegInf;
      for (int i = tid; i < C; i += bl * UNROLL) {
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) maxval = sycl::fmax(maxval, x[clamp_min(i + u * bl)]);
      }
      maxval = sycl::reduce_over_group(sg, maxval, sycl::maximum<float>{});
      if (lane == 0) maxvals[sg_id] = maxval;
      sycl::group_barrier(grp);
      if (tid == 0) {
        float v = maxvals[0];
#pragma unroll
        for (size_t i = 1; i < num_sgs; ++i) v = sycl::fmax(v, maxvals[i]);
        maxvals[0] = v;
      }
      sycl::group_barrier(grp);
      float offset = maxvals[0];
      float sumval = 0.0f;
      for (int i = tid; i < C; i += bl * UNROLL) {
        float regs[UNROLL];
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) regs[u] = x[clamp_min(i + u * bl)];
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
          if (i + u * bl < C) {
            float o = sycl::exp(regs[u] - offset);
            y[clamp_min(i + u * bl)] = o;
            sumval += o;
          }
        }
      }
      sumval = sycl::reduce_over_group(sg, sumval, sycl::plus<float>{});
      if (lane == 0) sumvals[sg_id] = sumval;
      sycl::group_barrier(grp);
      if (tid == 0) {
        float v = sumvals[0];
#pragma unroll
        for (size_t i = 1; i < num_sgs; ++i) v += sumvals[i];
        sumvals[0] = v;
      }
      sycl::group_barrier(grp);
      float sum = sumvals[0];
      for (int i = tid; i < C; i += bl * UNROLL) {
        float regs[UNROLL];
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) regs[u] = y[clamp_min(i + u * bl)];
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
          if (i + u * bl < C) y[i + u * bl] = regs[u] / sum;
        }
      }
    });
  });
  q.wait();
}

// ---- variant 8: online, one row per sub-group, shift-based merge ----
void softmax_forward_online8(sycl::queue &q, float *out, const float *inp, int N, int C,
                             int block_size) {
  int grid = (int)ceil_div(N * 32, block_size);
  sycl::nd_range<1> rng((size_t)grid * (size_t)block_size, (size_t)block_size);
  q.submit([&](sycl::handler &h) {
    h.parallel_for(rng, [=](sycl::nd_item<1> it) {
      auto sg = it.get_sub_group();
      size_t sg_size = sg.get_local_range()[0];
      size_t sg_lid = sg.get_local_id()[0];
      size_t sg_gid = sg.get_group_id()[0];
      size_t gid = (size_t)it.get_group(0);
      size_t sgs_per_group = (size_t)it.get_local_range(0) / sg_size;
      size_t row = gid * sgs_per_group + sg_gid;
      int tid = (int)it.get_local_id(0);
      if (tid >= C) return;
      if (row >= (size_t)N) return;
      const float *x = inp + row * (size_t)C;
      float *y = out + row * (size_t)C;
      float maxval = kNegInf, sumval = 0.0f;
      for (int i = (int)sg_lid; i < C; i += (int)sg_size) {
        float bigger = sycl::fmax(maxval, x[i]);
        sumval = sumval * sycl::exp(maxval - bigger) + sycl::exp(x[i] - bigger);
        maxval = bigger;
      }
      for (size_t off = sg_size / 2; off > 0; off >>= 1) {
        size_t src = sg_lid + off < sg_size ? sg_lid + off : sg_lid;
        float o_max = sycl::select_from_group(sg, maxval, src);
        float o_sum = sycl::select_from_group(sg, sumval, src);
        if (sg_lid + off < sg_size) {
          if (o_max > maxval) {
            sumval *= sycl::exp(maxval - o_max);
            maxval = o_max;
          } else {
            o_sum *= sycl::exp(o_max - maxval);
          }
          sumval += o_sum;
        }
      }
      maxval = sycl::group_broadcast(sg, maxval, 0);
      sumval = sycl::group_broadcast(sg, sumval, 0);
      for (int i = (int)sg_lid; i < C; i += (int)sg_size) y[i] = sycl::exp(x[i] - maxval) / sumval;
    });
  });
  q.wait();
}

void softmax_forward(sycl::queue &q, int k, float *out, const float *inp, int N, int C,
                     int block_size) {
  switch (k) {
    case 1: softmax_forward1(q, out, inp, N, C, block_size); break;
    case 2: softmax_forward2(q, out, inp, N, C, block_size); break;
    case 3: softmax_forward3(q, out, inp, N, C, block_size); break;
    case 4: softmax_forward4(q, out, inp, N, C, block_size); break;
    case 5: softmax_forward_online1(q, out, inp, N, C, block_size); break;
    case 6: softmax_forward_online2(q, out, inp, N, C, block_size); break;
    case 7: softmax_forward7(q, out, inp, N, C, block_size); break;
    case 8: softmax_forward_online8(q, out, inp, N, C, block_size); break;
    default: fprintf(stderr, "Invalid kernel number\n"); exit(1);
  }
}

int main(int argc, char **argv) {
  if (argc != 6) {
    fprintf(stderr, "usage: %s <input> <output> B T V\n", argv[0]);
    return 2;
  }
  const int B = atoi(argv[3]);
  const int T = atoi(argv[4]);
  const int V = atoi(argv[5]);
  size_t n_elem = (size_t)B * (size_t)T * (size_t)V;
  size_t n_out = (size_t)B * (size_t)T * 3;
  FILE *fin = fopen(argv[1], "rb");
  if (!fin) { perror(argv[1]); return 2; }
  std::vector<float> inp(n_elem);
  if (fread(inp.data(), sizeof(float), n_elem, fin) != n_elem) {
    fprintf(stderr, "input too short at inp\n");
    return 2;
  }
  std::vector<int> outliers(n_out);
  if (fread(outliers.data(), sizeof(int), n_out, fin) != n_out) {
    fprintf(stderr, "input too short at outliers\n");
    return 2;
  }
  fclose(fin);
  for (int k = 0; k < 3; ++k)
    for (int j = 0; j < B * T; ++j) inp[(size_t)j * (size_t)V + outliers[(size_t)j * 3 + k]] *= 20.0f;

  sycl::queue q;
  float *d_inp = sycl::malloc_device<float>(n_elem, q);
  float *d_out = sycl::malloc_device<float>(n_elem, q);
  std::vector<float> out(n_elem);
  q.memcpy(d_inp, inp.data(), n_elem * sizeof(float)).wait();

  FILE *fout = fopen(argv[2], "wb");
  if (!fout) { perror(argv[2]); return 2; }
  const int variants[] = {1, 2, 3, 4, 5, 6, 7, 8};
  for (int vi = 0; vi < 8; ++vi) {
    int k = variants[vi];
    q.memset(d_out, 0, n_elem * sizeof(float)).wait();
    softmax_forward(q, k, d_out, d_inp, B * T, V, 256);
    q.memcpy(out.data(), d_out, n_elem * sizeof(float)).wait();
    fwrite(out.data(), sizeof(float), n_elem, fout);
  }
  fclose(fout);
  sycl::free(d_inp, q);
  sycl::free(d_out, q);
  return 0;
}
