// Idiomatic SYCL 2020 port of matmul_backward_bias (fp32).
// Each variant k in [1,2,3,4,5,7,8,9] is its own SYCL kernel on the device.
// Contract: build/candidate <input> <output> B T OC
#include <sycl/sycl.hpp>

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <vector>

using floatX = float;

namespace {

template <typename T>
T ceil_div(T a, T b) {
  return (a + b - 1) / b;
}

bool is_power_of_two(int n) { return (n > 0) && ((n & (n - 1)) == 0); }

int largest_pow2_le(int n) {
  if (n < 1) return 0;
  while ((n & (n - 1)) > 0) n = n & (n - 1);
  return n;
}

int pick_local(int want, int device_max) {
  int v = largest_pow2_le(std::min(want, device_max));
  if (v < 1) v = 1;
  // keep work-group modest for portability
  if (v > 256) v = 256;
  return v;
}

}  // namespace

// ---- variant 1: one work-group per channel, local tree reduction ----
void run_bias1(sycl::queue &q, float *dbias, const float *dout, int B, int T,
               int OC, int max_wg) {
  const int BT = B * T;
  const int ls = pick_local(256, max_wg);
  const std::size_t groups = static_cast<std::size_t>(OC);
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> scratch(sycl::range<1>(static_cast<std::size_t>(ls)), h);
    h.parallel_for(
        sycl::nd_range<1>(groups * static_cast<std::size_t>(ls),
                          sycl::range<1>(static_cast<std::size_t>(ls))),
        [=](sycl::nd_item<1> it) {
          const int ch = static_cast<int>(it.get_group(0));
          const int lid = static_cast<int>(it.get_local_id(0));
          const int n = static_cast<int>(it.get_local_range(0));
          const float *col = dout + ch;
          float sum = 0.0f;
          for (int r = lid; r < BT; r += n) sum += col[static_cast<std::size_t>(r) * static_cast<std::size_t>(OC)];
          scratch[lid] = sum;
          it.barrier(sycl::access::fence_space::local_space);
          for (int stride = n / 2; stride >= 1; stride /= 2) {
            if (lid < stride) scratch[lid] += scratch[lid + stride];
            it.barrier(sycl::access::fence_space::local_space);
          }
          if (lid == 0) dbias[ch] += scratch[0];
        });
  }).wait();
}

// ---- variant 2: one sub-group per channel ----
void run_bias2(sycl::queue &q, float *dbias, const float *dout, int B, int T,
               int OC, int max_wg, std::size_t sg_hint) {
  const int BT = B * T;
  const int ls = pick_local(128, max_wg);
  if (sg_hint < 1) sg_hint = 8;
  const std::size_t groups =
      ceil_div(static_cast<std::size_t>(OC) * sg_hint, static_cast<std::size_t>(ls));
  q.submit([&](sycl::handler &h) {
    h.parallel_for(
        sycl::nd_range<1>(groups * static_cast<std::size_t>(ls),
                          sycl::range<1>(static_cast<std::size_t>(ls))),
        [=](sycl::nd_item<1> it) {
          auto sg = it.get_sub_group();
          const std::size_t sg_width = sg.get_local_range()[0];
          const std::size_t per_group =
              it.get_local_range(0) / sg_width;
          const std::size_t idx =
              it.get_group(0) * per_group + sg.get_group_linear_id();
          if (idx >= static_cast<std::size_t>(OC)) return;
          const int ch = static_cast<int>(idx);
          float sum = 0.0f;
          for (std::size_t r = sg.get_local_linear_id();
               r < static_cast<std::size_t>(BT);
               r += sg.get_local_linear_range())
            sum += dout[r * static_cast<std::size_t>(OC) + static_cast<std::size_t>(ch)];
          sum = sycl::reduce_over_group(sg, sum, sycl::plus<float>{});
          if (sg.get_local_linear_id() == 0) dbias[ch] += sum;
        });
  }).wait();
}

// ---- variant 3: whole work-group per channel, sub-group reduce + tail ----
void run_bias3(sycl::queue &q, float *dbias, const float *dout, int B, int T,
               int OC, int max_wg) {
  const int BT = B * T;
  const int ls = pick_local(256, max_wg);
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> part(sycl::range<1>(static_cast<std::size_t>(ls)), h);
    h.parallel_for(
        sycl::nd_range<1>(static_cast<std::size_t>(OC) * static_cast<std::size_t>(ls),
                          sycl::range<1>(static_cast<std::size_t>(ls))),
        [=](sycl::nd_item<1> it) {
          const int ch = static_cast<int>(it.get_group(0));
          const int lid = static_cast<int>(it.get_local_id(0));
          const int n = static_cast<int>(it.get_local_range(0));
          float tsum = 0.0f;
          for (int r = lid; r < BT; r += n)
            tsum += dout[static_cast<std::size_t>(r) * static_cast<std::size_t>(OC) + static_cast<std::size_t>(ch)];
          auto sg = it.get_sub_group();
          float wsum = sycl::reduce_over_group(sg, tsum, sycl::plus<float>{});
          if (sg.get_local_linear_id() == 0)
            part[sg.get_group_linear_id()] = wsum;
          it.barrier(sycl::access::fence_space::local_space);
          if (lid == 0) {
            const std::size_t sg_width = it.get_sub_group().get_local_range()[0];
            const int nparts = n / static_cast<int>(sg_width);
            float acc = 0.0f;
            for (int i = 0; i < nparts; ++i) acc += part[i];
            dbias[ch] += acc;
          }
        });
  }).wait();
}

// ---- variant 4: tiled columns, coalesced access ----
void run_bias4(sycl::queue &q, float *dbias, const float *dout, int B, int T,
               int OC, int max_wg) {
  constexpr int kTile = 16;
  constexpr int kStep = 8;
  constexpr int kLocal = kTile * kStep;
  (void)max_wg;
  const int BT = B * T;
  const int groups = OC / kTile;
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> tile(sycl::range<1>(kLocal), h);
    h.parallel_for(
        sycl::nd_range<1>(static_cast<std::size_t>(groups) * kLocal,
                          sycl::range<1>(kLocal)),
        [=](sycl::nd_item<1> it) {
          const int g = static_cast<int>(it.get_group(0));
          const int lid = static_cast<int>(it.get_local_id(0));
          const int col = lid % kTile;
          const int slice = lid / kTile;
          const int base = g * kTile;
          const float *colptr = dout + base + col;
          float s = 0.0f;
          for (int r = slice; r < BT; r += kStep)
            s += colptr[static_cast<std::size_t>(r) * static_cast<std::size_t>(OC)];
          tile[lid] = s;
          it.barrier(sycl::access::fence_space::local_space);
          if (lid < kTile) {
            float a = 0.0f;
            for (int j = 0; j < kStep; ++j) a += tile[static_cast<std::size_t>(col) + static_cast<std::size_t>(j) * kTile];
            dbias[base + col] += a;
          }
        });
  }).wait();
}

// ---- variant 5: strided rows with global atomics ----
void run_bias5(sycl::queue &q, float *dbias, const float *dout, int B, int T,
               int OC, int max_wg) {
  const int BT = B * T;
  constexpr int kParts = 4;
  const int ls = pick_local(256, max_wg);
  const std::size_t total = static_cast<std::size_t>(OC) * kParts;
  const std::size_t padded = ceil_div(total, static_cast<std::size_t>(ls)) * static_cast<std::size_t>(ls);
  q.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(padded), sycl::range<1>(static_cast<std::size_t>(ls))),
      [=](sycl::nd_item<1> it) {
        const std::size_t gid = it.get_global_id(0);
        if (gid >= total) return;
        const int ch = static_cast<int>(gid % static_cast<std::size_t>(OC));
        const int part = static_cast<int>(gid / static_cast<std::size_t>(OC));
        float s = 0.0f;
        for (int r = part; r < BT; r += kParts)
          s += dout[static_cast<std::size_t>(r) * static_cast<std::size_t>(OC) + static_cast<std::size_t>(ch)];
        sycl::atomic_ref<float, sycl::memory_order::relaxed,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            ref(dbias[ch]);
        ref.fetch_add(s);
      }).wait();
}

// helpers for buffered variants
void zero_buf(sycl::queue &q, float *p, std::size_t n) {
  q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { p[i[0]] = 0.0f; }).wait();
}

void cast_add_buf(sycl::queue &q, float *dst, const float *buf, int OC) {
  q.parallel_for(sycl::range<1>(static_cast<std::size_t>(OC)),
                 [=](sycl::id<1> i) {
                   const std::size_t k = i[0];
                   dst[k] += buf[k];
                 }).wait();
}

// ---- variant 7: fp32 scratch buffer + final add ----
void run_bias7(sycl::queue &q, float *dbias, const float *dout, float *scratch,
               int B, int T, int OC, int max_wg) {
  const int BT = B * T;
  constexpr int kParts = 4;
  zero_buf(q, scratch, static_cast<std::size_t>(OC));
  const int ls = pick_local(256, max_wg);
  const std::size_t total = static_cast<std::size_t>(OC) * kParts;
  const std::size_t padded = ceil_div(total, static_cast<std::size_t>(ls)) * static_cast<std::size_t>(ls);
  // accumulate into scratch
  q.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(padded), sycl::range<1>(static_cast<std::size_t>(ls))),
      [=](sycl::nd_item<1> it) {
        const std::size_t gid = it.get_global_id(0);
        if (gid >= total) return;
        const int ch = static_cast<int>(gid % static_cast<std::size_t>(OC));
        const int part = static_cast<int>(gid / static_cast<std::size_t>(OC));
        float s = 0.0f;
        for (int r = part; r < BT; r += kParts)
          s += dout[static_cast<std::size_t>(r) * static_cast<std::size_t>(OC) + static_cast<std::size_t>(ch)];
        sycl::atomic_ref<float, sycl::memory_order::relaxed,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            ref(scratch[ch]);
        ref.fetch_add(s);
      }).wait();
  cast_add_buf(q, dbias, scratch, OC);
}

// ---- variant 8 ----
void run_bias8(sycl::queue &q, float *dbias, const float *dout, float *scratch,
               int B, int T, int OC, int max_wg, int num_cu) {
  const int BT = B * T;
  constexpr int kOcPerGroup = 64;
  const int gx = ceil_div(OC, kOcPerGroup);
  const int ls = pick_local(128, max_wg);
  long fill = static_cast<long>(num_cu) * 1024L / (static_cast<long>(ls) * gx);
  int gy = static_cast<int>(fill);
  if (gy < 1) gy = 1;
  if (gy > 8) gy = 8;
  if (gy == 1) {
    // direct write, one contributor per channel
    q.parallel_for(sycl::range<1>(static_cast<std::size_t>(OC)),
                   [=](sycl::id<1> i) {
                     const int ch = static_cast<int>(i[0]);
                     float s = 0.0f;
                     for (int r = 0; r < BT; ++r)
                       s += dout[static_cast<std::size_t>(r) * static_cast<std::size_t>(OC) + static_cast<std::size_t>(ch)];
                     dbias[ch] += s;
                   }).wait();
  } else {
    zero_buf(q, scratch, static_cast<std::size_t>(OC));
    const std::size_t total = static_cast<std::size_t>(OC) * static_cast<std::size_t>(gy);
    const std::size_t padded =
        ceil_div(total, static_cast<std::size_t>(ls)) * static_cast<std::size_t>(ls);
    q.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(padded), sycl::range<1>(static_cast<std::size_t>(ls))),
        [=](sycl::nd_item<1> it) {
          const std::size_t gid = it.get_global_id(0);
          if (gid >= total) return;
          const int ch = static_cast<int>(gid % static_cast<std::size_t>(OC));
          const int part = static_cast<int>(gid / static_cast<std::size_t>(OC));
          float s = 0.0f;
          for (int r = part; r < BT; r += gy)
            s += dout[static_cast<std::size_t>(r) * static_cast<std::size_t>(OC) + static_cast<std::size_t>(ch)];
          sycl::atomic_ref<float, sycl::memory_order::relaxed,
                           sycl::memory_scope::device,
                           sycl::access::address_space::global_space>
              ref(scratch[ch]);
          ref.fetch_add(s);
        }).wait();
    cast_add_buf(q, dbias, scratch, OC);
  }
}

// ---- variant 9: aux buffer per row-partition + row reduction ----
void run_bias9(sycl::queue &q, float *dbias, const float *dout, float *scratch,
               int B, int T, int OC, int max_wg, int num_cu) {
  const int BT = B * T;
  constexpr int kOcPerGroup = 64;
  const int gx = ceil_div(OC, kOcPerGroup);
  const int ls = pick_local(128, max_wg);
  long fill = static_cast<long>(num_cu) * 1024L / (static_cast<long>(ls) * gx);
  int gy = static_cast<int>(fill);
  if (gy < 1) gy = 1;
  if (gy > 8) gy = 8;
  if (gy == 1) {
    q.parallel_for(sycl::range<1>(static_cast<std::size_t>(OC)),
                   [=](sycl::id<1> i) {
                     const int ch = static_cast<int>(i[0]);
                     float s = 0.0f;
                     for (int r = 0; r < BT; ++r)
                       s += dout[static_cast<std::size_t>(r) * static_cast<std::size_t>(OC) + static_cast<std::size_t>(ch)];
                     dbias[ch] += s;
                   }).wait();
  } else {
    const std::size_t total = static_cast<std::size_t>(OC) * static_cast<std::size_t>(gy);
    const std::size_t padded =
        ceil_div(total, static_cast<std::size_t>(ls)) * static_cast<std::size_t>(ls);
    // each (part, channel) writes its own slot
    q.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(padded), sycl::range<1>(static_cast<std::size_t>(ls))),
        [=](sycl::nd_item<1> it) {
          const std::size_t gid = it.get_global_id(0);
          if (gid >= total) return;
          const int ch = static_cast<int>(gid % static_cast<std::size_t>(OC));
          const int part = static_cast<int>(gid / static_cast<std::size_t>(OC));
          float s = 0.0f;
          for (int r = part; r < BT; r += gy)
            s += dout[static_cast<std::size_t>(r) * static_cast<std::size_t>(OC) + static_cast<std::size_t>(ch)];
          scratch[static_cast<std::size_t>(part) * static_cast<std::size_t>(OC) + static_cast<std::size_t>(ch)] = s;
        }).wait();
    // reduce across partitions
    q.parallel_for(sycl::range<1>(static_cast<std::size_t>(OC)),
                   [=](sycl::id<1> i) {
                     const int ch = static_cast<int>(i[0]);
                     float acc = 0.0f;
                     for (int l = 0; l < gy; ++l)
                       acc += scratch[static_cast<std::size_t>(l) * static_cast<std::size_t>(OC) + static_cast<std::size_t>(ch)];
                     dbias[ch] += acc;
                   }).wait();
  }
}

int main(int argc, char **argv) {
  if (argc != 6) {
    std::fprintf(stderr, "usage: %s <input> <output> B T OC\n", argv[0]);
    return 2;
  }
  const int B = std::atoi(argv[3]);
  const int T = std::atoi(argv[4]);
  const int OC = std::atoi(argv[5]);
  const std::size_t n_dout =
      static_cast<std::size_t>(B) * static_cast<std::size_t>(T) * static_cast<std::size_t>(OC);

  std::FILE *fin = std::fopen(argv[1], "rb");
  if (!fin) {
    std::perror(argv[1]);
    return 2;
  }
  std::vector<float> dout_h(n_dout);
  if (std::fread(dout_h.data(), sizeof(float), n_dout, fin) != n_dout) {
    std::fprintf(stderr, "input too short at dout\n");
    std::fclose(fin);
    return 2;
  }
  std::fclose(fin);
  std::FILE *fout = std::fopen(argv[2], "wb");
  if (!fout) {
    std::perror(argv[2]);
    return 2;
  }

  sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order{}};
  const int max_wg = static_cast<int>(
      q.get_device().get_info<sycl::info::device::max_work_group_size>());
  int num_cu = static_cast<int>(
      q.get_device().get_info<sycl::info::device::max_compute_units>());
  if (num_cu < 1) num_cu = 1;
  std::size_t sg_hint = 8;
  try {
    auto sizes =
        q.get_device().get_info<sycl::info::device::sub_group_sizes>();
    for (auto s : sizes) sg_hint = std::max(sg_hint, s);
  } catch (...) {
  }

  float *d_dout = sycl::malloc_device<float>(n_dout, q);
  float *d_dbias = sycl::malloc_device<float>(static_cast<std::size_t>(OC), q);
  // scratch holds up to 8 row-partitions of OC
  float *d_scratch =
      sycl::malloc_device<float>(static_cast<std::size_t>(OC) * 8, q);
  if (!d_dout || !d_dbias || !d_scratch) {
    std::fprintf(stderr, "device alloc failed\n");
    return 1;
  }
  q.memcpy(d_dout, dout_h.data(), n_dout * sizeof(float)).wait();

  std::vector<float> dbias_h(static_cast<std::size_t>(OC));
  const int variants[8] = {1, 2, 3, 4, 5, 7, 8, 9};
  for (int vi = 0; vi < 8; ++vi) {
    const int k = variants[vi];
    q.memset(d_dbias, 0, static_cast<std::size_t>(OC) * sizeof(float)).wait();
    switch (k) {
      case 1:
        run_bias1(q, d_dbias, d_dout, B, T, OC, max_wg);
        break;
      case 2:
        run_bias2(q, d_dbias, d_dout, B, T, OC, max_wg, sg_hint);
        break;
      case 3:
        run_bias3(q, d_dbias, d_dout, B, T, OC, max_wg);
        break;
      case 4:
        run_bias4(q, d_dbias, d_dout, B, T, OC, max_wg);
        break;
      case 5:
        run_bias5(q, d_dbias, d_dout, B, T, OC, max_wg);
        break;
      case 7:
        run_bias7(q, d_dbias, d_dout, d_scratch, B, T, OC, max_wg);
        break;
      case 8:
        run_bias8(q, d_dbias, d_dout, d_scratch, B, T, OC, max_wg, num_cu);
        break;
      case 9:
        run_bias9(q, d_dbias, d_dout, d_scratch, B, T, OC, max_wg, num_cu);
        break;
      default:
        std::printf("Invalid kernel number\n");
        return 1;
    }
    q.wait();
    q.memcpy(dbias_h.data(), d_dbias, static_cast<std::size_t>(OC) * sizeof(float)).wait();
    std::fwrite(dbias_h.data(), sizeof(float), static_cast<std::size_t>(OC), fout);
  }
  std::fclose(fout);
  sycl::free(d_dout, q);
  sycl::free(d_dbias, q);
  sycl::free(d_scratch, q);
  return 0;
}
