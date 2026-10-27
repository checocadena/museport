#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>

template <typename T> T ceil_div(T a, T b) { return (a + b - 1) / b; }

void layernorm_forward_cpu(float *out, float *mean, float *rstd,
                           const float *inp, const float *weight,
                           const float *bias, int B, int T, int C) {
  float eps = 1e-5f;
  for (int b = 0; b < B; b++) {
    for (int t = 0; t < T; t++) {
      const float *x = inp + b * T * C + t * C;
      float m = 0.0f;
      for (int i = 0; i < C; i++) m += x[i];
      m = m / C;
      float v = 0.0f;
      for (int i = 0; i < C; i++) {
        float xs = x[i] - m;
        v += xs * xs;
      }
      v = v / C;
      float s = 1.0f / sqrtf(v + eps);
      float *ob = out + b * T * C + t * C;
      for (int i = 0; i < C; i++) {
        float n = s * (x[i] - m);
        ob[i] = n * weight[i] + bias[i];
      }
      mean[b * T + t] = m;
      rstd[b * T + t] = s;
    }
  }
}

inline void atomic_add_global_f(float *addr, float v) {
  sycl::atomic_ref<float, sycl::memory_order::relaxed,
                   sycl::memory_scope::device,
                   sycl::access::address_space::global_space>
      ar(*addr);
  ar.fetch_add(v);
}
inline unsigned atomic_add_global_u(unsigned *addr, unsigned v) {
  sycl::atomic_ref<unsigned, sycl::memory_order::acq_rel,
                   sycl::memory_scope::device,
                   sycl::access::address_space::global_space>
      ar(*addr);
  return ar.fetch_add(v);
}

// sub-group reduction using built-in group algorithm (used by k2/k3/k5/k6)
inline float sg_reduce_plus(sycl::sub_group sg, float v) {
  return sycl::reduce_over_group(sg, v, sycl::plus<float>{});
}
// explicit xor-shuffle reduction (used by k7/k8/k9/k10, no group algorithm)
inline float sg_reduce_xor(sycl::sub_group sg, float v) {
  unsigned sg_n = sg.get_local_range()[0];
  for (unsigned mask = sg_n >> 1; mask > 0; mask >>= 1) {
    v += sycl::permute_group_by_xor(sg, v, (uint32_t)mask);
  }
  return v;
}

// ---------------- K1 : naive, one work-item per BT, global atomics -------------
class LnBwdK1;
void launch_k1(sycl::queue &q, float *dinp, float *dweight, float *dbias,
               const float *dout, const float *inp, const float *weight,
               const float *mean, const float *rstd, int B, int T, int C,
               int block) {
  int N = B * T;
  int grid = ceil_div(N, block);
  sycl::nd_range<1> rng(grid * block, block);
  q.submit([&](sycl::handler &h) {
    h.parallel_for<LnBwdK1>(rng, [=](sycl::nd_item<1> it) {
      int idx = it.get_global_id(0);
      if (idx >= N) return;
      int b = idx / T;
      int t = idx % T;
      const float *dout_bt = dout + (size_t)b * T * C + (size_t)t * C;
      const float *inp_bt = inp + (size_t)b * T * C + (size_t)t * C;
      float *dinp_bt = dinp + (size_t)b * T * C + (size_t)t * C;
      float mean_bt = mean[(size_t)b * T + t];
      float rstd_bt = rstd[(size_t)b * T + t];
      float dm = 0.0f, dnm = 0.0f;
      for (int i = 0; i < C; i++) {
        float n = (inp_bt[i] - mean_bt) * rstd_bt;
        float d = weight[i] * dout_bt[i];
        dm += d;
        dnm += d * n;
      }
      dm /= (float)C;
      dnm /= (float)C;
      for (int i = 0; i < C; i++) {
        float n = (inp_bt[i] - mean_bt) * rstd_bt;
        float d = weight[i] * dout_bt[i];
        atomic_add_global_f(&dbias[i], dout_bt[i]);
        atomic_add_global_f(&dweight[i], n * dout_bt[i]);
        float dv = (d - dm - n * dnm) * rstd_bt;
        dinp_bt[i] += dv;
      }
    });
  });
}

// ---------------- K2 : warp per BT, local staging, tmp buffers + copy ---------
class LnBwdK2Main;
class LnBwdK2Copy;
void launch_k2(sycl::queue &q, float *dinp, float *dweight, float *dbias,
               const float *dout, const float *inp, const float *weight,
               const float *mean, const float *rstd, int B, int T, int C,
               int block) {
  int N = B * T;
  auto dev = q.get_device();
  size_t sg_hint = 16;
  try {
    auto sgs = dev.get_info<sycl::info::device::sub_group_sizes>();
    if (!sgs.empty()) sg_hint = sgs.back();
  } catch (...) {}
  // ensure enough sub-groups to cover N (over-provision, portable)
  int grid = ceil_div((long long)N * (long long)sg_hint, (long long)block);
  if (grid < 1) grid = 1;
  float *dtmp_w = sycl::malloc_device<float>(C, q);
  float *dtmp_b = sycl::malloc_device<float>(C, q);
  q.memset(dtmp_w, 0, sizeof(float) * C).wait();
  q.memset(dtmp_b, 0, sizeof(float) * C).wait();
  size_t lmem = (size_t)2 * C;
  sycl::nd_range<1> rng(grid * block, block);
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> loc(sycl::range<1>(lmem), h);
    h.parallel_for<LnBwdK2Main>(rng, [=](sycl::nd_item<1> it) {
      auto sg = it.get_sub_group();
      unsigned sg_lid = sg.get_local_linear_id();
      unsigned sg_n = sg.get_local_linear_range();
      unsigned nsg = sg.get_group_linear_range();
      unsigned sg_id = sg.get_group_linear_id();
      unsigned lid = it.get_local_id(0);
      unsigned lrange = it.get_local_range(0);
      unsigned gid = it.get_group(0);
      unsigned long warp_idx = (unsigned long)gid * nsg + sg_id;
      float *db_s = (float *)loc.get_multi_ptr<sycl::access::decorated::no>().get();
      float *dw_s = db_s + C;
      for (unsigned i = lid; i < (unsigned)C; i += lrange) {
        db_s[i] = 0.0f;
        dw_s[i] = 0.0f;
      }
      it.barrier(sycl::access::fence_space::local_space);
      bool active = ((long long)warp_idx < N);
      float dm = 0.0f, dnm = 0.0f;
      int b = -1, t = -1;
      const float *dout_bt = nullptr;
      const float *inp_bt = nullptr;
      float *dinp_bt = nullptr;
      float mean_bt = 0, rstd_bt = 0;
      if (active) {
        b = (int)warp_idx / T;
        t = (int)warp_idx % T;
        dout_bt = dout + (size_t)b * T * C + (size_t)t * C;
        inp_bt = inp + (size_t)b * T * C + (size_t)t * C;
        dinp_bt = dinp + (size_t)b * T * C + (size_t)t * C;
        mean_bt = mean[(size_t)b * T + t];
        rstd_bt = rstd[(size_t)b * T + t];
        for (unsigned i = sg_lid; i < (unsigned)C; i += sg_n) {
          float n = (inp_bt[i] - mean_bt) * rstd_bt;
          float d = weight[i] * dout_bt[i];
          dm += d;
          dnm += d * n;
        }
        dm = sg_reduce_plus(sg, dm);
        dnm = sg_reduce_plus(sg, dnm);
        dm /= (float)C;
        dnm /= (float)C;
      }
      // inactive sub-groups still must not call reduce divergently; they skip.
      if (active) {
        for (unsigned i = sg_lid; i < (unsigned)C; i += sg_n) {
          float n = (inp_bt[i] - mean_bt) * rstd_bt;
          float d = weight[i] * dout_bt[i];
          {
            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                             sycl::memory_scope::work_group,
                             sycl::access::address_space::local_space>
                ar(db_s[i]);
            ar.fetch_add(dout_bt[i]);
          }
          {
            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                             sycl::memory_scope::work_group,
                             sycl::access::address_space::local_space>
                ar(dw_s[i]);
            ar.fetch_add(n * dout_bt[i]);
          }
          float dv = (d - dm - n * dnm) * rstd_bt;
          dinp_bt[i] = dinp_bt[i] + dv;
        }
      }
      it.barrier(sycl::access::fence_space::local_space);
      for (unsigned i = lid; i < (unsigned)C; i += lrange) {
        atomic_add_global_f(&dtmp_b[i], db_s[i]);
        atomic_add_global_f(&dtmp_w[i], dw_s[i]);
      }
    });
  }).wait();
  // copy tmp -> final
  {
    sycl::nd_range<1> rng2(512, 512);
    // 1 group of 512, grid covers C
    int cgrid = ceil_div(C, 512);
    q.submit([&](sycl::handler &h) {
      h.parallel_for<LnBwdK2Copy>(sycl::nd_range<1>(cgrid * 512, 512),
                                  [=](sycl::nd_item<1> it) {
        int i = it.get_global_id(0);
        if (i < C) {
          dbias[i] = dtmp_b[i];
          dweight[i] = dtmp_w[i];
        }
      });
    }).wait();
  }
  sycl::free(dtmp_w, q);
  sycl::free(dtmp_b, q);
}

// ---------------- K3 : persistent warps, shared staging, global atomics -------
class LnBwdK3;
void launch_k3(sycl::queue &q, float *dinp, float *dweight, float *dbias,
               const float *dout, const float *inp, const float *weight,
               const float *mean, const float *rstd, int B, int T, int C,
               int block, int ncu) {
  int grid = (1024 / block) * ncu;
  if (grid < 1) grid = 1;
  size_t lmem = (size_t)2 * C;
  sycl::nd_range<1> rng(grid * block, block);
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> loc(sycl::range<1>(lmem), h);
    h.parallel_for<LnBwdK3>(rng, [=](sycl::nd_item<1> it) {
      auto sg = it.get_sub_group();
      unsigned sg_lid = sg.get_local_linear_id();
      unsigned sg_n = sg.get_local_linear_range();
      unsigned nsg = sg.get_group_linear_range();
      unsigned sg_id = sg.get_group_linear_id();
      unsigned lid = it.get_local_id(0);
      unsigned lrange = it.get_local_range(0);
      unsigned long base = (unsigned long)it.get_group(0) * nsg + sg_id;
      unsigned long stride = (unsigned long)it.get_group_range(0) * nsg;
      float *db_s = (float *)loc.get_multi_ptr<sycl::access::decorated::no>().get();
      float *dw_s = db_s + C;
      for (unsigned i = lid; i < (unsigned)C; i += lrange) {
        db_s[i] = 0.0f;
        dw_s[i] = 0.0f;
      }
      it.barrier(sycl::access::fence_space::local_space);
      int N = B * T;
      for (unsigned long idx = base; idx < (unsigned long)N; idx += stride) {
        int b = (int)idx / T;
        int t = (int)idx % T;
        const float *dout_bt = dout + (size_t)b * T * C + (size_t)t * C;
        const float *inp_bt = inp + (size_t)b * T * C + (size_t)t * C;
        float *dinp_bt = dinp + (size_t)b * T * C + (size_t)t * C;
        float mean_bt = mean[(size_t)b * T + t];
        float rstd_bt = rstd[(size_t)b * T + t];
        float dm = 0.0f, dnm = 0.0f;
        for (unsigned i = sg_lid; i < (unsigned)C; i += sg_n) {
          float n = (inp_bt[i] - mean_bt) * rstd_bt;
          float d = weight[i] * dout_bt[i];
          dm += d;
          dnm += d * n;
        }
        dm = sg_reduce_plus(sg, dm) / (float)C;
        dnm = sg_reduce_plus(sg, dnm) / (float)C;
        for (unsigned i = sg_lid; i < (unsigned)C; i += sg_n) {
          float dout_i = dout_bt[i];
          float n = (inp_bt[i] - mean_bt) * rstd_bt;
          float d = weight[i] * dout_i;
          {
            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                             sycl::memory_scope::work_group,
                             sycl::access::address_space::local_space>
                ar(db_s[i]);
            ar.fetch_add(dout_i);
          }
          {
            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                             sycl::memory_scope::work_group,
                             sycl::access::address_space::local_space>
                ar(dw_s[i]);
            ar.fetch_add(n * dout_i);
          }
          float dv = (d - dm - n * dnm) * rstd_bt;
          dinp_bt[i] = dinp_bt[i] + dv;
        }
      }
      it.barrier(sycl::access::fence_space::local_space);
      for (unsigned i = lid; i < (unsigned)C; i += lrange) {
        atomic_add_global_f(&dbias[i], db_s[i]);
        atomic_add_global_f(&dweight[i], dw_s[i]);
      }
    });
  });
}

// ---------------- K5 : per-block scratch + flag, last block reduces -----------
class LnBwdK5;
void launch_k5(sycl::queue &q, float *dinp, float *dweight, float *dbias,
               float *scratch, const float *dout, const float *inp,
               const float *weight, const float *mean, const float *rstd,
               int B, int T, int C, int block, int ncu) {
  int grid = ncu;
  if (grid < 1) grid = 1;
  q.memset(scratch, 0, sizeof(float) * ((size_t)grid * 2 * C + 1)).wait();
  size_t lmem = (size_t)2 * C + 1;
  sycl::nd_range<1> rng(grid * block, block);
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> loc(sycl::range<1>(lmem), h);
    h.parallel_for<LnBwdK5>(rng, [=](sycl::nd_item<1> it) {
      auto sg = it.get_sub_group();
      unsigned sg_lid = sg.get_local_linear_id();
      unsigned sg_n = sg.get_local_linear_range();
      unsigned nsg = sg.get_group_linear_range();
      unsigned sg_id = sg.get_group_linear_id();
      unsigned lid = it.get_local_id(0);
      unsigned lrange = it.get_local_range(0);
      unsigned long base = (unsigned long)it.get_group(0) * nsg + sg_id;
      unsigned long stride = (unsigned long)it.get_group_range(0) * nsg;
      float *raw = (float *)loc.get_multi_ptr<sycl::access::decorated::no>().get();
      float *db_s = raw;
      float *dw_s = raw + C;
      unsigned *flag_loc =
          reinterpret_cast<unsigned *>(raw + 2 * C);
      for (unsigned i = lid; i < (unsigned)C; i += lrange) {
        db_s[i] = 0.0f;
        dw_s[i] = 0.0f;
      }
      it.barrier(sycl::access::fence_space::local_space);
      int N = B * T;
      for (unsigned long idx = base; idx < (unsigned long)N; idx += stride) {
        int b = (int)idx / T;
        int t = (int)idx % T;
        const float *dout_bt = dout + (size_t)b * T * C + (size_t)t * C;
        const float *inp_bt = inp + (size_t)b * T * C + (size_t)t * C;
        float *dinp_bt = dinp + (size_t)b * T * C + (size_t)t * C;
        float mean_bt = mean[(size_t)b * T + t];
        float rstd_bt = rstd[(size_t)b * T + t];
        float dm = 0.0f, dnm = 0.0f;
        for (unsigned i = sg_lid; i < (unsigned)C; i += sg_n) {
          float n = (inp_bt[i] - mean_bt) * rstd_bt;
          float d = weight[i] * dout_bt[i];
          dm += d;
          dnm += d * n;
        }
        dm = sg_reduce_plus(sg, dm) / (float)C;
        dnm = sg_reduce_plus(sg, dnm) / (float)C;
        for (unsigned i = sg_lid; i < (unsigned)C; i += sg_n) {
          float dout_i = dout_bt[i];
          float n = (inp_bt[i] - mean_bt) * rstd_bt;
          float d = weight[i] * dout_i;
          {
            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                             sycl::memory_scope::work_group,
                             sycl::access::address_space::local_space>
                ar(db_s[i]);
            ar.fetch_add(dout_i);
          }
          {
            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                             sycl::memory_scope::work_group,
                             sycl::access::address_space::local_space>
                ar(dw_s[i]);
            ar.fetch_add(n * dout_i);
          }
          float dv = (d - dm - n * dnm) * rstd_bt;
          dinp_bt[i] = dinp_bt[i] + dv;
        }
      }
      it.barrier(sycl::access::fence_space::local_space);
      float *s_db = scratch;
      float *s_dw = scratch + (size_t)C * (size_t)it.get_group_range(0);
      unsigned *gflag = reinterpret_cast<unsigned *>(
          scratch + (size_t)2 * C * (size_t)it.get_group_range(0));
      unsigned gdim = it.get_group_range(0);
      unsigned gidx = it.get_group(0);
      for (unsigned i = lid; i < (unsigned)C; i += lrange) {
        s_db[i + (size_t)C * gidx] = db_s[i];
        s_dw[i + (size_t)C * gidx] = dw_s[i];
      }
      sycl::atomic_fence(sycl::memory_order::acq_rel,
                         sycl::memory_scope::device);
      it.barrier(sycl::access::fence_space::global_space);
      if (lid == 0) {
        unsigned ticket = atomic_add_global_u(gflag, 1);
        *flag_loc = ticket;
      }
      it.barrier(sycl::access::fence_space::local_space);
      unsigned ticket = *flag_loc;
      if (ticket == gdim - 1) {
        for (unsigned i = lid; i < (unsigned)C; i += lrange) {
          float sb = 0.0f, sw = 0.0f;
          for (unsigned j = 0; j < gdim; j++) {
            sb += s_db[i + (size_t)j * C];
            sw += s_dw[i + (size_t)j * C];
          }
          dbias[i] = dbias[i] + sb;
          dweight[i] = dweight[i] + sw;
        }
      }
    });
  });
}

// ---------------- K6 : single shared scratch, global atomics + flag ----------
class LnBwdK6;
void launch_k6(sycl::queue &q, float *dinp, float *dweight, float *dbias,
               float *scratch, const float *dout, const float *inp,
               const float *weight, const float *mean, const float *rstd,
               int B, int T, int C, int block, int ncu) {
  int grid = (1024 / block) * ncu;
  if (grid < 1) grid = 1;
  q.memset(scratch, 0, sizeof(float) * ((size_t)1 + 2 * C)).wait();
  size_t lmem = (size_t)2 * C + 1;
  sycl::nd_range<1> rng(grid * block, block);
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> loc(sycl::range<1>(lmem), h);
    h.parallel_for<LnBwdK6>(rng, [=](sycl::nd_item<1> it) {
      auto sg = it.get_sub_group();
      unsigned sg_lid = sg.get_local_linear_id();
      unsigned sg_n = sg.get_local_linear_range();
      unsigned nsg = sg.get_group_linear_range();
      unsigned sg_id = sg.get_group_linear_id();
      unsigned lid = it.get_local_id(0);
      unsigned lrange = it.get_local_range(0);
      unsigned long base = (unsigned long)it.get_group(0) * nsg + sg_id;
      unsigned long stride = (unsigned long)it.get_group_range(0) * nsg;
      float *raw = (float *)loc.get_multi_ptr<sycl::access::decorated::no>().get();
      float *db_s = raw;
      float *dw_s = raw + C;
      unsigned *flag_loc = reinterpret_cast<unsigned *>(raw + 2 * C);
      for (unsigned i = lid; i < (unsigned)C; i += lrange) {
        db_s[i] = 0.0f;
        dw_s[i] = 0.0f;
      }
      it.barrier(sycl::access::fence_space::local_space);
      int N = B * T;
      for (unsigned long idx = base; idx < (unsigned long)N; idx += stride) {
        int b = (int)idx / T;
        int t = (int)idx % T;
        const float *dout_bt = dout + (size_t)b * T * C + (size_t)t * C;
        const float *inp_bt = inp + (size_t)b * T * C + (size_t)t * C;
        float *dinp_bt = dinp + (size_t)b * T * C + (size_t)t * C;
        float mean_bt = mean[(size_t)b * T + t];
        float rstd_bt = rstd[(size_t)b * T + t];
        float dm = 0.0f, dnm = 0.0f;
        for (unsigned i = sg_lid; i < (unsigned)C; i += sg_n) {
          float n = (inp_bt[i] - mean_bt) * rstd_bt;
          float d = weight[i] * dout_bt[i];
          dm += d;
          dnm += d * n;
        }
        dm = sg_reduce_plus(sg, dm) / (float)C;
        dnm = sg_reduce_plus(sg, dnm) / (float)C;
        for (unsigned i = sg_lid; i < (unsigned)C; i += sg_n) {
          float dout_i = dout_bt[i];
          float n = (inp_bt[i] - mean_bt) * rstd_bt;
          float d = weight[i] * dout_i;
          {
            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                             sycl::memory_scope::work_group,
                             sycl::access::address_space::local_space>
                ar(db_s[i]);
            ar.fetch_add(dout_i);
          }
          {
            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                             sycl::memory_scope::work_group,
                             sycl::access::address_space::local_space>
                ar(dw_s[i]);
            ar.fetch_add(n * dout_i);
          }
          float dv = (d - dm - n * dnm) * rstd_bt;
          dinp_bt[i] = dinp_bt[i] + dv;
        }
      }
      it.barrier(sycl::access::fence_space::local_space);
      float *s_db = scratch;
      float *s_dw = scratch + C;
      unsigned *gflag = reinterpret_cast<unsigned *>(scratch + 2 * C);
      unsigned gdim = it.get_group_range(0);
      for (unsigned i = lid; i < (unsigned)C; i += lrange) {
        atomic_add_global_f(&s_db[i], db_s[i]);
        atomic_add_global_f(&s_dw[i], dw_s[i]);
      }
      it.barrier(sycl::access::fence_space::global_space);
      if (lid == 0) {
        *flag_loc = atomic_add_global_u(gflag, 1);
      }
      it.barrier(sycl::access::fence_space::local_space);
      if (*flag_loc == gdim - 1) {
        for (unsigned i = lid; i < (unsigned)C; i += lrange) {
          dbias[i] = s_db[i];
          dweight[i] = s_dw[i];
        }
      }
    });
  });
}

// ---------------- K7 : like K6 but xor-shuffle reduction, no group algo ------
class LnBwdK7;
void launch_k7(sycl::queue &q, float *dinp, float *dweight, float *dbias,
               float *scratch, const float *dout, const float *inp,
               const float *weight, const float *mean, const float *rstd,
               int B, int T, int C, int block, int ncu) {
  int grid = (1024 / block) * ncu;
  if (grid < 1) grid = 1;
  q.memset(scratch, 0, sizeof(float) * ((size_t)1 + 2 * C)).wait();
  size_t lmem = (size_t)2 * C + 1;
  sycl::nd_range<1> rng(grid * block, block);
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> loc(sycl::range<1>(lmem), h);
    h.parallel_for<LnBwdK7>(rng, [=](sycl::nd_item<1> it) {
      auto sg = it.get_sub_group();
      unsigned sg_lid = sg.get_local_linear_id();
      unsigned sg_n = sg.get_local_linear_range();
      unsigned nsg = sg.get_group_linear_range();
      unsigned sg_id = sg.get_group_linear_id();
      unsigned lid = it.get_local_id(0);
      unsigned lrange = it.get_local_range(0);
      // derive warp indices arithmetically (no cooperative groups)
      unsigned warps_in_block = lrange / sg_n;
      unsigned warp_id = lid / sg_n;
      unsigned lane = lid % sg_n;
      unsigned long base =
          (unsigned long)it.get_group(0) * warps_in_block + warp_id;
      unsigned long stride =
          (unsigned long)it.get_group_range(0) * warps_in_block;
      (void)sg_lid;
      float *raw = (float *)loc.get_multi_ptr<sycl::access::decorated::no>().get();
      float *db_s = raw;
      float *dw_s = raw + C;
      unsigned *flag_loc = reinterpret_cast<unsigned *>(raw + 2 * C);
      for (unsigned i = lid; i < (unsigned)C; i += lrange) {
        db_s[i] = 0.0f;
        dw_s[i] = 0.0f;
      }
      it.barrier(sycl::access::fence_space::local_space);
      int N = B * T;
      for (unsigned long idx = base; idx < (unsigned long)N; idx += stride) {
        int b = (int)idx / T;
        int t = (int)idx % T;
        const float *dout_bt = dout + (size_t)b * T * C + (size_t)t * C;
        const float *inp_bt = inp + (size_t)b * T * C + (size_t)t * C;
        float *dinp_bt = dinp + (size_t)b * T * C + (size_t)t * C;
        float mean_bt = mean[(size_t)b * T + t];
        float rstd_bt = rstd[(size_t)b * T + t];
        float dm = 0.0f, dnm = 0.0f;
        for (unsigned i = lane; i < (unsigned)C; i += sg_n) {
          float n = (inp_bt[i] - mean_bt) * rstd_bt;
          float d = weight[i] * dout_bt[i];
          dm += d;
          dnm += d * n;
        }
        dm = sg_reduce_xor(sg, dm) / (float)C;
        dnm = sg_reduce_xor(sg, dnm) / (float)C;
        for (unsigned i = lane; i < (unsigned)C; i += sg_n) {
          float dout_i = dout_bt[i];
          float n = (inp_bt[i] - mean_bt) * rstd_bt;
          float d = weight[i] * dout_i;
          {
            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                             sycl::memory_scope::work_group,
                             sycl::access::address_space::local_space>
                ar(db_s[i]);
            ar.fetch_add(dout_i);
          }
          {
            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                             sycl::memory_scope::work_group,
                             sycl::access::address_space::local_space>
                ar(dw_s[i]);
            ar.fetch_add(n * dout_i);
          }
          float dv = (d - dm - n * dnm) * rstd_bt;
          dinp_bt[i] = dinp_bt[i] + dv;
        }
      }
      it.barrier(sycl::access::fence_space::local_space);
      float *s_db = scratch;
      float *s_dw = scratch + C;
      unsigned *gflag = reinterpret_cast<unsigned *>(scratch + 2 * C);
      unsigned gdim = it.get_group_range(0);
      for (unsigned i = lid; i < (unsigned)C; i += lrange) {
        atomic_add_global_f(&s_db[i], db_s[i]);
        atomic_add_global_f(&s_dw[i], dw_s[i]);
      }
      it.barrier(sycl::access::fence_space::global_space);
      if (lid == 0) *flag_loc = atomic_add_global_u(gflag, 1);
      it.barrier(sycl::access::fence_space::local_space);
      if (*flag_loc == gdim - 1) {
        for (unsigned i = lid; i < (unsigned)C; i += lrange) {
          dbias[i] = s_db[i];
          dweight[i] = s_dw[i];
        }
      }
    });
  });
}

// ---------------- K8 : vectorized 128-bit, bank-friendly shared order --------
class LnBwdK8;
void launch_k8(sycl::queue &q, float *dinp, float *dweight, float *dbias,
               float *scratch, const float *dout, const float *inp,
               const float *weight, const float *mean, const float *rstd,
               int B, int T, int C, int block, int ncu) {
  int grid = (1024 / block) * ncu;
  if (grid < 1) grid = 1;
  q.memset(scratch, 0, sizeof(float) * ((size_t)1 + 2 * C)).wait();
  size_t lmem = (size_t)2 * C + 1;
  sycl::nd_range<1> rng(grid * block, block);
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> loc(sycl::range<1>(lmem), h);
    h.parallel_for<LnBwdK8>(rng, [=](sycl::nd_item<1> it) {
      auto sg = it.get_sub_group();
      unsigned sg_n = sg.get_local_linear_range();
      unsigned nsg = sg.get_group_linear_range();
      unsigned sg_id = sg.get_group_linear_id();
      unsigned lid = it.get_local_id(0);
      unsigned lrange = it.get_local_range(0);
      unsigned warps_in_block = lrange / sg_n;
      unsigned warp_id = lid / sg_n;
      unsigned lane = lid % sg_n;
      unsigned long base =
          (unsigned long)it.get_group(0) * warps_in_block + warp_id;
      // use sub-group size actually present
      unsigned long stride =
          (unsigned long)it.get_group_range(0) * warps_in_block;
      (void)nsg;
      (void)sg_id;
      constexpr unsigned VEC = 4;
      unsigned c_per_iter = sg_n * VEC;
      int iters = C / (int)c_per_iter;
      float *raw = (float *)loc.get_multi_ptr<sycl::access::decorated::no>().get();
      float *db_s = raw;
      float *dw_s = raw + C;
      unsigned *flag_loc = reinterpret_cast<unsigned *>(raw + 2 * C);
      for (unsigned i = lid; i < (unsigned)C; i += lrange) {
        db_s[i] = 0.0f;
        dw_s[i] = 0.0f;
      }
      it.barrier(sycl::access::fence_space::local_space);
      int N = B * T;
      using vec4 = sycl::vec<float, 4>;
      for (unsigned long idx = base; idx < (unsigned long)N; idx += stride) {
        int b = (int)idx / T;
        int t = (int)idx % T;
        const float *dout_bt = dout + (size_t)b * T * C + (size_t)t * C;
        const float *inp_bt = inp + (size_t)b * T * C + (size_t)t * C;
        float *dinp_bt = dinp + (size_t)b * T * C + (size_t)t * C;
        float mean_bt = mean[(size_t)b * T + t];
        float rstd_bt = rstd[(size_t)b * T + t];
        float dm = 0.0f, dnm = 0.0f;
        for (unsigned i = lane * VEC; i < (unsigned)C; i += sg_n * VEC) {
          vec4 dvo = reinterpret_cast<const vec4 *>(dout_bt + i)[0];
          vec4 ivo = reinterpret_cast<const vec4 *>(inp_bt + i)[0];
          vec4 wvo = reinterpret_cast<const vec4 *>(weight + i)[0];
          for (unsigned k = 0; k < VEC; k++) {
            float n = (ivo[k] - mean_bt) * rstd_bt;
            float d = wvo[k] * dvo[k];
            dm += d;
            dnm += d * n;
          }
        }
        dm = sg_reduce_xor(sg, dm) / (float)C;
        dnm = sg_reduce_xor(sg, dnm) / (float)C;
        for (int ic = 0; ic < iters; ic++) {
          unsigned gidx = lane * VEC + (unsigned)ic * c_per_iter;
          unsigned sidx = lane + (unsigned)ic * c_per_iter;
          vec4 dvo = reinterpret_cast<const vec4 *>(dout_bt + gidx)[0];
          vec4 ivo = reinterpret_cast<const vec4 *>(inp_bt + gidx)[0];
          vec4 wvo = reinterpret_cast<const vec4 *>(weight + gidx)[0];
          vec4 din = reinterpret_cast<const vec4 *>(dinp_bt + gidx)[0];
          for (unsigned x = 0; x < VEC; x++) {
            float dout_i = dvo[x];
            float n = (ivo[x] - mean_bt) * rstd_bt;
            float d = wvo[x] * dout_i;
            {
              sycl::atomic_ref<float, sycl::memory_order::relaxed,
                               sycl::memory_scope::work_group,
                               sycl::access::address_space::local_space>
                  ar(db_s[sidx + x * sg_n]);
              ar.fetch_add(dout_i);
            }
            {
              sycl::atomic_ref<float, sycl::memory_order::relaxed,
                               sycl::memory_scope::work_group,
                               sycl::access::address_space::local_space>
                  ar(dw_s[sidx + x * sg_n]);
              ar.fetch_add(n * dout_i);
            }
            float dv = (d - dm - n * dnm) * rstd_bt;
            din[x] = din[x] + dv;
          }
          reinterpret_cast<vec4 *>(dinp_bt + gidx)[0] = din;
        }
      }
      it.barrier(sycl::access::fence_space::local_space);
      float *s_db = scratch;
      float *s_dw = scratch + C;
      unsigned *gflag = reinterpret_cast<unsigned *>(scratch + 2 * C);
      unsigned gdim = it.get_group_range(0);
      for (unsigned i = lid; i < (unsigned)C; i += lrange) {
        atomic_add_global_f(&s_db[i], db_s[i]);
        atomic_add_global_f(&s_dw[i], dw_s[i]);
      }
      it.barrier(sycl::access::fence_space::global_space);
      if (lid == 0) *flag_loc = atomic_add_global_u(gflag, 1);
      it.barrier(sycl::access::fence_space::local_space);
      if (*flag_loc == gdim - 1) {
        for (int ic = (int)warp_id; ic < iters; ic += (int)warps_in_block) {
          unsigned gidx = lane * VEC + (unsigned)ic * c_per_iter;
          unsigned sidx = lane + (unsigned)ic * c_per_iter;
          vec4 dbo = reinterpret_cast<const vec4 *>(dbias + gidx)[0];
          vec4 dwo = reinterpret_cast<const vec4 *>(dweight + gidx)[0];
          for (unsigned x = 0; x < VEC; x++) {
            dbo[x] = s_db[sidx + x * sg_n] + dbo[x];
            dwo[x] = s_dw[sidx + x * sg_n] + dwo[x];
          }
          reinterpret_cast<vec4 *>(dbias + gidx)[0] = dbo;
          reinterpret_cast<vec4 *>(dweight + gidx)[0] = dwo;
        }
      }
    });
  });
}

// ---------------- K9 : vectorized + inter-warp reduction, per-block scratch --
class LnBwdK9;
void launch_k9(sycl::queue &q, float *dinp, float *dweight, float *dbias,
               float *scratch, const float *dout, const float *inp,
               const float *weight, const float *mean, const float *rstd,
               int B, int T, int C, int block, int ncu) {
  int grid = (1024 / block) * ncu;
  if (grid < 1) grid = 1;
  q.memset(scratch, 0, sizeof(float)).wait();
  size_t lmem = (size_t)2 * C + (size_t)2 * block + 1;
  sycl::nd_range<1> rng(grid * block, block);
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> loc(sycl::range<1>(lmem), h);
    h.parallel_for<LnBwdK9>(rng, [=](sycl::nd_item<1> it) {
      auto sg = it.get_sub_group();
      unsigned sg_n = sg.get_local_linear_range();
      unsigned lid = it.get_local_id(0);
      unsigned lrange = it.get_local_range(0);
      unsigned warps_in_block = lrange / sg_n;
      unsigned warp_id = lid / sg_n;
      unsigned lane = lid % sg_n;
      unsigned long base =
          (unsigned long)it.get_group(0) * warps_in_block + warp_id;
      unsigned long stride =
          (unsigned long)it.get_group_range(0) * warps_in_block;
      constexpr unsigned VEC = 4;
      unsigned c_per_iter = sg_n * VEC;
      int iters = ceil_div(C, (int)c_per_iter) + 2;
      float *raw = (float *)loc.get_multi_ptr<sycl::access::decorated::no>().get();
      float *db_s = raw;
      float *dw_s = raw + C;
      float *db_tmp = raw + 2 * C;
      float *dw_tmp = raw + 2 * C + lrange;
      unsigned *flag_loc =
          reinterpret_cast<unsigned *>(raw + 2 * C + 2 * lrange);
      for (unsigned i = lid; i < (unsigned)C; i += lrange) {
        db_s[i] = 0.0f;
        dw_s[i] = 0.0f;
      }
      it.barrier(sycl::access::fence_space::local_space);
      int N = B * T;
      using vec4 = sycl::vec<float, 4>;
      for (unsigned long idx = base; idx < (unsigned long)N; idx += stride) {
        int b = (int)idx / T;
        int t = (int)idx % T;
        const float *dout_bt = dout + (size_t)b * T * C + (size_t)t * C;
        const float *inp_bt = inp + (size_t)b * T * C + (size_t)t * C;
        float *dinp_bt = dinp + (size_t)b * T * C + (size_t)t * C;
        float mean_bt = mean[(size_t)b * T + t];
        float rstd_bt = rstd[(size_t)b * T + t];
        float dm = 0.0f, dnm = 0.0f;
        for (unsigned i = lane * VEC; i < (unsigned)C; i += sg_n * VEC) {
          vec4 dvo = reinterpret_cast<const vec4 *>(dout_bt + i)[0];
          vec4 ivo = reinterpret_cast<const vec4 *>(inp_bt + i)[0];
          vec4 wvo = reinterpret_cast<const vec4 *>(weight + i)[0];
          for (unsigned k = 0; k < VEC; k++) {
            float n = (ivo[k] - mean_bt) * rstd_bt;
            float d = wvo[k] * dvo[k];
            dm += d;
            dnm += d * n;
          }
        }
        dm = sg_reduce_xor(sg, dm) / (float)C;
        dnm = sg_reduce_xor(sg, dnm) / (float)C;
        for (int ic = 0; ic < iters; ic++) {
          unsigned gidx = lane * VEC + (unsigned)ic * c_per_iter;
          unsigned sidx = lane + (unsigned)ic * c_per_iter;
          if (gidx >= (unsigned)C) break;
          vec4 dvo = reinterpret_cast<const vec4 *>(dout_bt + gidx)[0];
          vec4 ivo = reinterpret_cast<const vec4 *>(inp_bt + gidx)[0];
          vec4 din = reinterpret_cast<const vec4 *>(dinp_bt + gidx)[0];
          vec4 wvo = reinterpret_cast<const vec4 *>(weight + gidx)[0];
          for (unsigned x = 0; x < VEC; x++) {
            float dout_i = dvo[x];
            float n = (ivo[x] - mean_bt) * rstd_bt;
            float d = wvo[x] * dout_i;
            if (warp_id != 0) {
              db_tmp[lid] = dout_i;
              dw_tmp[lid] = n * dout_i;
            }
            it.barrier(sycl::access::fence_space::local_space);
            if (warp_id == 0) {
              float ab = dout_i, aw = n * dout_i;
              for (unsigned j = 1; j < warps_in_block; j++) {
                ab += db_tmp[lid + j * sg_n];
                aw += dw_tmp[lid + j * sg_n];
              }
              db_s[sidx + x * sg_n] += ab;
              dw_s[sidx + x * sg_n] += aw;
            }
            it.barrier(sycl::access::fence_space::local_space);
            float dv = (d - dm - n * dnm) * rstd_bt;
            din[x] = din[x] + dv;
          }
          reinterpret_cast<vec4 *>(dinp_bt + gidx)[0] = din;
        }
      }
      it.barrier(sycl::access::fence_space::local_space);
      unsigned *gflag = reinterpret_cast<unsigned *>(scratch);
      float *sbase = scratch + 32;
      float *s_db = sbase;
      float *s_dw = sbase + C;
      unsigned gdim = it.get_group_range(0);
      unsigned gidxb = it.get_group(0);
      for (unsigned i = lid; i < (unsigned)C; i += lrange) {
        s_db[i + (size_t)2 * C * gidxb] = db_s[i];
        s_dw[i + (size_t)2 * C * gidxb] = dw_s[i];
      }
      it.barrier(sycl::access::fence_space::global_space);
      if (lid == 0) *flag_loc = atomic_add_global_u(gflag, 1);
      it.barrier(sycl::access::fence_space::local_space);
      if (*flag_loc == gdim - 1) {
        for (unsigned i = lid * VEC; i < (unsigned)C; i += lrange * VEC) {
          vec4 ab{0, 0, 0, 0}, aw{0, 0, 0, 0};
          for (unsigned rb = 0; rb < gdim; rb++) {
            unsigned off = i + (unsigned)2 * C * rb;
            vec4 b4 = reinterpret_cast<const vec4 *>(s_db + off)[0];
            vec4 w4 = reinterpret_cast<const vec4 *>(s_dw + off)[0];
            for (unsigned k = 0; k < VEC; k++) {
              ab[k] += b4[k];
              aw[k] += w4[k];
            }
          }
          reinterpret_cast<vec4 *>(db_s + i)[0] = ab;
          reinterpret_cast<vec4 *>(dw_s + i)[0] = aw;
        }
        it.barrier(sycl::access::fence_space::local_space);
        for (int ic = (int)warp_id; ic < iters; ic += (int)warps_in_block) {
          unsigned gidx = lane * VEC + (unsigned)ic * c_per_iter;
          unsigned sidx = lane + (unsigned)ic * c_per_iter;
          if (gidx >= (unsigned)C) break;
          vec4 dbo = reinterpret_cast<const vec4 *>(dbias + gidx)[0];
          vec4 dwo = reinterpret_cast<const vec4 *>(dweight + gidx)[0];
          for (unsigned x = 0; x < VEC; x++) {
            dbo[x] = db_s[sidx + x * sg_n] + dbo[x];
            dwo[x] = dw_s[sidx + x * sg_n] + dwo[x];
          }
          reinterpret_cast<vec4 *>(dbias + gidx)[0] = dbo;
          reinterpret_cast<vec4 *>(dweight + gidx)[0] = dwo;
        }
      }
    });
  });
}

// ---------------- K10 : vector shared accesses, rounded C --------------------
class LnBwdK10;
void launch_k10(sycl::queue &q, float *dinp, float *dweight, float *dbias,
                float *scratch, const float *dout, const float *inp,
                const float *weight, const float *mean, const float *rstd,
                int B, int T, int C, int block, int ncu) {
  int blk = (block == 1024) ? 512 : block;
  int grid = (1024 / blk) * ncu;
  if (grid < 1) grid = 1;
  q.memset(scratch, 0, sizeof(float)).wait();
  constexpr unsigned VEC = 4;
  // rounded_C to multiple of 128 (sg=32 assumed max, use 128 granularity)
  size_t rounded = ((C + 128 - 1) / 128) * 128;
  size_t lmem = (size_t)2 * rounded + (size_t)2 * (blk)*VEC;
  sycl::nd_range<1> rng(grid * blk, blk);
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> loc(sycl::range<1>(lmem), h);
    h.parallel_for<LnBwdK10>(
        rng, [=](sycl::nd_item<1> it) [[intel::reqd_sub_group_size(16)]] {
          auto sg = it.get_sub_group();
          unsigned sg_n = sg.get_local_linear_range();
          unsigned lid = it.get_local_id(0);
          unsigned lrange = it.get_local_range(0);
          unsigned warps_in_block = lrange / sg_n;
          unsigned warp_id = lid / sg_n;
          unsigned lane = lid % sg_n;
          unsigned long base =
              (unsigned long)it.get_group(0) * warps_in_block + warp_id;
          unsigned long stride =
              (unsigned long)it.get_group_range(0) * warps_in_block;
          unsigned c_per_iter = sg_n * VEC;
          int iters = ceil_div(C, (int)c_per_iter);
          float *raw =
              (float *)loc.get_multi_ptr<sycl::access::decorated::no>().get();
          float *db_s = raw;
          float *dw_s = raw + rounded;
          float *db_tmp = raw + 2 * rounded - sg_n * VEC;
          float *dw_tmp =
              raw + 2 * rounded + (size_t)VEC * lrange - (size_t)2 * sg_n * VEC;
          using vec4 = sycl::vec<float, 4>;
          for (unsigned i = lid * VEC; i < rounded; i += lrange * VEC) {
            reinterpret_cast<vec4 *>(db_s + i)[0] = vec4(0, 0, 0, 0);
            reinterpret_cast<vec4 *>(dw_s + i)[0] = vec4(0, 0, 0, 0);
          }
          it.barrier(sycl::access::fence_space::local_space);
          int N = B * T;
          for (unsigned long bt = base; bt < (unsigned long)N; bt += stride) {
            const float *dout_bt = dout + bt * (size_t)C;
            const float *inp_bt = inp + bt * (size_t)C;
            float *dinp_bt = dinp + bt * (size_t)C;
            float dm = 0.0f, dnm = 0.0f;
            for (unsigned i = lane * VEC; i < (unsigned)C; i += sg_n * VEC) {
              vec4 dvo = reinterpret_cast<const vec4 *>(dout_bt + i)[0];
              vec4 ivo = reinterpret_cast<const vec4 *>(inp_bt + i)[0];
              vec4 wvo = reinterpret_cast<const vec4 *>(weight + i)[0];
              for (unsigned k = 0; k < VEC; k++) {
                float d = wvo[k] * dvo[k];
                dm += d;
                dnm += d * ivo[k];
              }
            }
            float mean_bt = mean[bt];
            float rstd_bt = rstd[bt];
            dm = sg_reduce_xor(sg, dm) / (float)C;
            dnm = sg_reduce_xor(sg, dnm) / (float)C * rstd_bt -
                  dm * mean_bt * rstd_bt;
            for (int cc = 0; cc < iters; cc++) {
              unsigned gidx = lane * VEC + (unsigned)cc * c_per_iter;
              vec4 dvo{0, 0, 0, 0}, ivo{0, 0, 0, 0}, din{0, 0, 0, 0},
                  wvo{0, 0, 0, 0};
              bool in = (gidx < (unsigned)C);
              if (in) {
                dvo = reinterpret_cast<const vec4 *>(dout_bt + gidx)[0];
                ivo = reinterpret_cast<const vec4 *>(inp_bt + gidx)[0];
                din = reinterpret_cast<const vec4 *>(dinp_bt + gidx)[0];
                wvo = reinterpret_cast<const vec4 *>(weight + gidx)[0];
              }
              for (unsigned o = 0; o < VEC / VEC; ++o) {
                vec4 dbf, dwf;
                for (unsigned f = 0; f < VEC; ++f) {
                  unsigned x = o * VEC + f;
                  float dout_i = dvo[x];
                  float n = (ivo[x] - mean_bt) * rstd_bt;
                  if (!in) {
                    dout_i = 0;
                    n = 0;
                  }
                  dbf[f] = dout_i;
                  dwf[f] = n * dout_i;
                  float dv = (wvo[x] * dvo[x] - dm - n * dnm) * rstd_bt;
                  if (!in) dv = 0;
                  din[x] = din[x] + (in ? dv : 0);
                }
                if (warp_id != 0) {
                  reinterpret_cast<vec4 *>(db_tmp + (size_t)lid * VEC)[0] =
                      dbf;
                  reinterpret_cast<vec4 *>(dw_tmp + (size_t)lid * VEC)[0] =
                      dwf;
                }
                it.barrier(sycl::access::fence_space::local_space);
                if (warp_id == 0 && in) {
                  for (unsigned j = 1; j < warps_in_block; j++) {
                    vec4 a = reinterpret_cast<const vec4 *>(
                        db_tmp + (size_t)(lid + j * sg_n) * VEC)[0];
                    vec4 b2 = reinterpret_cast<const vec4 *>(
                        dw_tmp + (size_t)(lid + j * sg_n) * VEC)[0];
                    for (unsigned f = 0; f < VEC; f++) {
                      dbf[f] += a[f];
                      dwf[f] += b2[f];
                    }
                  }
                }
                it.barrier(sycl::access::fence_space::local_space);
                if (warp_id == 0 && in) {
                  vec4 ob =
                      reinterpret_cast<const vec4 *>(db_s + gidx + o * VEC)[0];
                  vec4 ow =
                      reinterpret_cast<const vec4 *>(dw_s + gidx + o * VEC)[0];
                  for (unsigned f = 0; f < VEC; f++) {
                    dbf[f] += ob[f];
                    dwf[f] += ow[f];
                  }
                  reinterpret_cast<vec4 *>(db_s + gidx + o * VEC)[0] = dbf;
                  reinterpret_cast<vec4 *>(dw_s + gidx + o * VEC)[0] = dwf;
                }
              }
              if (in) {
                reinterpret_cast<vec4 *>(dinp_bt + gidx)[0] = din;
              }
            }
          }
          it.barrier(sycl::access::fence_space::local_space);
          unsigned *gflag = reinterpret_cast<unsigned *>(scratch);
          float *sbase = scratch + 32;
          float *s_db = sbase;
          float *s_dw = sbase + C;
          unsigned gdim = it.get_group_range(0);
          unsigned gidxb = it.get_group(0);
          using vec4 = sycl::vec<float, 4>;
          for (unsigned i = lid * VEC; i < (unsigned)C; i += lrange * VEC) {
            reinterpret_cast<vec4 *>(s_db + i + (size_t)2 * C * gidxb)[0] =
                reinterpret_cast<const vec4 *>(db_s + i)[0];
            reinterpret_cast<vec4 *>(s_dw + i + (size_t)2 * C * gidxb)[0] =
                reinterpret_cast<const vec4 *>(dw_s + i)[0];
          }
          it.barrier(sycl::access::fence_space::global_space);
          float *raw2 =
              (float *)loc.get_multi_ptr<sycl::access::decorated::no>().get();
          unsigned *tmp_flag =
              reinterpret_cast<unsigned *>(raw2 + 2 * rounded);
          if (lid == 0) *tmp_flag = atomic_add_global_u(gflag, 1);
          it.barrier(sycl::access::fence_space::local_space);
          if (*tmp_flag == gdim - 1) {
            for (unsigned i = lid * VEC; i < (unsigned)C; i += lrange * VEC) {
              vec4 ab{0, 0, 0, 0}, aw{0, 0, 0, 0};
              for (unsigned rb = 0; rb < gdim; rb++) {
                unsigned off = i + (unsigned)2 * C * rb;
                vec4 b4 = reinterpret_cast<const vec4 *>(s_db + off)[0];
                vec4 w4 = reinterpret_cast<const vec4 *>(s_dw + off)[0];
                for (unsigned k = 0; k < VEC; k++) {
                  ab[k] += b4[k];
                  aw[k] += w4[k];
                }
              }
              reinterpret_cast<vec4 *>(db_s + i)[0] = ab;
              reinterpret_cast<vec4 *>(dw_s + i)[0] = aw;
            }
            it.barrier(sycl::access::fence_space::local_space);
            for (int cc = (int)warp_id; cc < iters; cc += (int)warps_in_block) {
              unsigned gidx = lane * VEC + (unsigned)cc * c_per_iter;
              if (gidx >= (unsigned)C) break;
              vec4 dbo = reinterpret_cast<const vec4 *>(dbias + gidx)[0];
              vec4 dwo = reinterpret_cast<const vec4 *>(dweight + gidx)[0];
              for (unsigned o = 0; o < VEC / VEC; ++o) {
                vec4 sb =
                    reinterpret_cast<const vec4 *>(db_s + gidx + o * VEC)[0];
                vec4 sw =
                    reinterpret_cast<const vec4 *>(dw_s + gidx + o * VEC)[0];
                for (unsigned f = 0; f < VEC; f++) {
                  unsigned x = o * VEC + f;
                  dbo[x] = sb[f] + dbo[x];
                  dwo[x] = sw[f] + dwo[x];
                }
              }
              reinterpret_cast<vec4 *>(dbias + gidx)[0] = dbo;
              reinterpret_cast<vec4 *>(dweight + gidx)[0] = dwo;
            }
          }
        });
  });
}

int main(int argc, char **argv) {
  if (argc != 6) {
    fprintf(stderr, "usage: %s <input> <output> B T C\n", argv[0]);
    return 2;
  }
  int B = atoi(argv[3]);
  int T = atoi(argv[4]);
  int C = atoi(argv[5]);
  FILE *fin = fopen(argv[1], "rb");
  if (!fin) {
    perror(argv[1]);
    return 2;
  }
  size_t nBTC = (size_t)B * T * C;
  size_t nBT = (size_t)B * T;
  std::vector<float> inp(nBTC), weight(C), bias(C), dout(nBTC);
  if (fread(inp.data(), 4, nBTC, fin) != nBTC) return 2;
  if (fread(weight.data(), 4, C, fin) != (size_t)C) return 2;
  if (fread(bias.data(), 4, C, fin) != (size_t)C) return 2;
  if (fread(dout.data(), 4, nBTC, fin) != nBTC) return 2;
  fclose(fin);
  std::vector<float> ln_out(nBTC), mean(nBT), rstd(nBT);
  layernorm_forward_cpu(ln_out.data(), mean.data(), rstd.data(), inp.data(),
                        weight.data(), bias.data(), B, T, C);
  sycl::queue q{sycl::default_selector_v};
  int ncu = (int)q.get_device().get_info<sycl::info::device::max_compute_units>();
  if (ncu < 1) ncu = 1;
  float *d_inp = sycl::malloc_device<float>(nBTC, q);
  float *d_weight = sycl::malloc_device<float>(C, q);
  float *d_bias = sycl::malloc_device<float>(C, q);
  float *d_dout = sycl::malloc_device<float>(nBTC, q);
  float *d_dinp = sycl::malloc_device<float>(nBTC, q);
  float *d_dweight = sycl::malloc_device<float>(C, q);
  float *d_dbias = sycl::malloc_device<float>(C, q);
  float *d_mean = sycl::malloc_device<float>(nBT, q);
  float *d_rstd = sycl::malloc_device<float>(nBT, q);
  size_t scratch_elems = (size_t)32 * (size_t)ncu * (size_t)(2 * C + 1) + 64;
  // k9 needs 32 + 2*C*grid ; grid up to (1024/256)*ncu=4*ncu -> 8*C*ncu ; our
  // alloc is 64*C*ncu so safe
  float *d_scratch = sycl::malloc_device<float>(scratch_elems, q);
  q.memcpy(d_inp, inp.data(), sizeof(float) * nBTC).wait();
  q.memcpy(d_weight, weight.data(), sizeof(float) * C).wait();
  q.memcpy(d_bias, bias.data(), sizeof(float) * C).wait();
  q.memcpy(d_dout, dout.data(), sizeof(float) * nBTC).wait();
  q.memcpy(d_mean, mean.data(), sizeof(float) * nBT).wait();
  q.memcpy(d_rstd, rstd.data(), sizeof(float) * nBT).wait();
  std::vector<float> dinp(nBTC), dweight(C), dbias(C);
  FILE *fout = fopen(argv[2], "wb");
  if (!fout) {
    perror(argv[2]);
    return 2;
  }
  const int variants[9] = {1, 2, 3, 5, 6, 7, 8, 9, 10};
  for (int vi = 0; vi < 9; vi++) {
    int k = variants[vi];
    q.memset(d_dinp, 0, sizeof(float) * nBTC).wait();
    q.memset(d_dweight, 0, sizeof(float) * C).wait();
    q.memset(d_dbias, 0, sizeof(float) * C).wait();
    q.memset(d_scratch, 0, sizeof(float) * scratch_elems).wait();
    switch (k) {
    case 1:
      launch_k1(q, d_dinp, d_dweight, d_dbias, d_dout, d_inp, d_weight, d_mean,
                d_rstd, B, T, C, 256);
      break;
    case 2:
      launch_k2(q, d_dinp, d_dweight, d_dbias, d_dout, d_inp, d_weight, d_mean,
                d_rstd, B, T, C, 256);
      break;
    case 3:
      launch_k3(q, d_dinp, d_dweight, d_dbias, d_dout, d_inp, d_weight, d_mean,
                d_rstd, B, T, C, 256, ncu);
      break;
    case 5:
      launch_k5(q, d_dinp, d_dweight, d_dbias, d_scratch, d_dout, d_inp,
                d_weight, d_mean, d_rstd, B, T, C, 256, ncu);
      break;
    case 6:
      launch_k6(q, d_dinp, d_dweight, d_dbias, d_scratch, d_dout, d_inp,
                d_weight, d_mean, d_rstd, B, T, C, 256, ncu);
      break;
    case 7:
      launch_k7(q, d_dinp, d_dweight, d_dbias, d_scratch, d_dout, d_inp,
                d_weight, d_mean, d_rstd, B, T, C, 256, ncu);
      break;
    case 8:
      launch_k8(q, d_dinp, d_dweight, d_dbias, d_scratch, d_dout, d_inp,
                d_weight, d_mean, d_rstd, B, T, C, 256, ncu);
      break;
    case 9:
      launch_k9(q, d_dinp, d_dweight, d_dbias, d_scratch, d_dout, d_inp,
                d_weight, d_mean, d_rstd, B, T, C, 256, ncu);
      break;
    case 10:
      launch_k10(q, d_dinp, d_dweight, d_dbias, d_scratch, d_dout, d_inp,
                 d_weight, d_mean, d_rstd, B, T, C, 256, ncu);
      break;
    }
    q.wait();
    q.memcpy(dinp.data(), d_dinp, sizeof(float) * nBTC).wait();
    q.memcpy(dweight.data(), d_dweight, sizeof(float) * C).wait();
    q.memcpy(dbias.data(), d_dbias, sizeof(float) * C).wait();
    fwrite(dinp.data(), 4, nBTC, fout);
    fwrite(dweight.data(), 4, C, fout);
    fwrite(dbias.data(), 4, C, fout);
  }
  fclose(fout);
  sycl::free(d_inp, q);
  sycl::free(d_weight, q);
  sycl::free(d_bias, q);
  sycl::free(d_dout, q);
  sycl::free(d_dinp, q);
  sycl::free(d_dweight, q);
  sycl::free(d_dbias, q);
  sycl::free(d_mean, q);
  sycl::free(d_rstd, q);
  sycl::free(d_scratch, q);
  return 0;
}
