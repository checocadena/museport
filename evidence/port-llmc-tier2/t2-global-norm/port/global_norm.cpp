#include <sycl/sycl.hpp>

#include <cassert>
#include <cstddef>
#include <cstdio>
#include <cstdlib>

// fp32 build: floatX is plain float.
using floatX = float;

namespace {

int occupancy_grid(sycl::queue &q, int block_size) {
  const std::size_t max_wg =
      q.get_device().get_info<sycl::info::device::max_work_group_size>();
  if (block_size > static_cast<int>(max_wg)) block_size = static_cast<int>(max_wg);
  if (block_size <= 0) block_size = 1;
  long cu =
      static_cast<long>(q.get_device().get_info<sycl::info::device::max_compute_units>());
  if (cu <= 0) cu = 1;
  // Aim for roughly 2048 resident threads per compute unit (CUDA heuristic),
  // then clamp so the deterministic k4 tail (single-group aggregate) stays valid.
  long target_threads = cu * 2048L;
  long grid = target_threads / block_size;
  if (grid < 1) grid = 1;
  if (grid > 512) grid = 512;
  return static_cast<int>(grid);
}

class NormKernel1;
class NormKernel2;
class NormKernel3;
class NormKernel4;
class NormAggregate;

void global_norm1(sycl::queue &q, float *out, const floatX *values, std::size_t count,
                  int block_size) {
  const int grid_size = occupancy_grid(q, block_size);
  assert(grid_size > 0);
  const std::size_t local = static_cast<std::size_t>(block_size);
  const sycl::nd_range<1> launch(grid_size * local, local);
  q.submit([&](sycl::handler &h) {
     sycl::local_accessor<float, 1> scratch(sycl::range<1>(local), h);
     h.parallel_for<NormKernel1>(launch, [=](sycl::nd_item<1> it) {
       const std::size_t idx = it.get_global_linear_id();
       const std::size_t stride = it.get_global_range(0);
       float acc = 0.0f;
       for (std::size_t i = idx; i < count; i += stride) {
         float v = static_cast<float>(values[i]);
         acc += v * v;
       }
       sycl::sub_group sg = it.get_sub_group();
       float sg_sum = sycl::reduce_over_group(sg, acc, sycl::plus<float>{});
       const std::size_t sg_id = sg.get_group_linear_id();
       const std::size_t num_sg = sg.get_group_linear_range();
       if (sg.leader()) scratch[sg_id] = sg_sum;
       sycl::group<1> grp = it.get_group();
       sycl::group_barrier(grp);
       const std::size_t lid = it.get_local_linear_id();
       float part = (lid < num_sg) ? scratch[lid] : 0.0f;
       float block_sum = sycl::reduce_over_group(grp, part, sycl::plus<float>{});
       if (lid == 0) {
         sycl::atomic_ref<float, sycl::memory_order::relaxed,
                           sycl::memory_scope::device>
             atom(out[0]);
         atom.fetch_add(block_sum);
       }
     });
   }).wait();
}

void global_norm2(sycl::queue &q, float *out, const floatX *values, std::size_t count,
                  int block_size) {
  const int grid_size = occupancy_grid(q, block_size);
  assert(grid_size > 0);
  const std::size_t local = static_cast<std::size_t>(block_size);
  const sycl::nd_range<1> launch(grid_size * local, local);
  q.submit([&](sycl::handler &h) {
     h.parallel_for<NormKernel2>(launch, [=](sycl::nd_item<1> it) {
       const std::size_t idx = it.get_global_linear_id();
       const std::size_t stride = it.get_global_range(0);
       float acc = 0.0f;
       for (std::size_t i = idx; i < count; i += stride) {
         float v = static_cast<float>(values[i]);
         acc += v * v;
       }
       sycl::sub_group sg = it.get_sub_group();
       float sg_sum = sycl::reduce_over_group(sg, acc, sycl::plus<float>{});
       if (sg.leader()) {
         sycl::atomic_ref<float, sycl::memory_order::relaxed,
                           sycl::memory_scope::device>
             atom(out[0]);
         atom.fetch_add(sg_sum);
       }
     });
   }).wait();
}

void global_norm3(sycl::queue &q, float *out, const floatX *values, std::size_t count,
                  int block_size) {
  const int grid_size = occupancy_grid(q, block_size);
  assert(grid_size > 0);
  const std::size_t local = static_cast<std::size_t>(block_size);
  const sycl::nd_range<1> launch(grid_size * local, local);
  q.submit([&](sycl::handler &h) {
     h.parallel_for<NormKernel3>(launch, [=](sycl::nd_item<1> it) {
       const std::size_t idx = it.get_global_linear_id();
       const std::size_t stride = it.get_global_range(0);
       float acc = 0.0f;
       for (std::size_t i = idx; i < count; i += stride) {
         float v = static_cast<float>(values[i]);
         acc += v * v;
       }
       float block_sum =
           sycl::reduce_over_group(it.get_group(), acc, sycl::plus<float>{});
       if (it.get_local_linear_id() == 0) {
         sycl::atomic_ref<float, sycl::memory_order::relaxed,
                           sycl::memory_scope::device>
             atom(out[0]);
         atom.fetch_add(block_sum);
       }
     });
   }).wait();
}

void global_norm_aggregate(sycl::queue &q, float *out, std::size_t num_blocks) {
  std::size_t max_wg =
      q.get_device().get_info<sycl::info::device::max_work_group_size>();
  std::size_t local = 256;
  if (local > max_wg) local = max_wg;
  if (local == 0) local = 1;
  const sycl::nd_range<1> launch(local, local);
  q.submit([&](sycl::handler &h) {
     h.parallel_for<NormAggregate>(launch, [=](sycl::nd_item<1> it) {
       const std::size_t lid = it.get_local_linear_id();
       const std::size_t lrange = it.get_local_range(0);
       float partial = 0.0f;
       for (std::size_t i = lid; i < num_blocks; i += lrange) partial += out[i];
       float sum =
           sycl::reduce_over_group(it.get_group(), partial, sycl::plus<float>{});
       if (lid == 0) out[0] = sum;
     });
   }).wait();
}

void global_norm4(sycl::queue &q, float *out, const floatX *values, std::size_t count,
                  int block_size) {
  if (block_size <= 64) block_size = 128;
  const std::size_t max_wg =
      q.get_device().get_info<sycl::info::device::max_work_group_size>();
  if (block_size > static_cast<int>(max_wg))
    block_size = static_cast<int>(max_wg);
  const int grid_size = occupancy_grid(q, block_size);
  assert(grid_size > 0);
  assert(grid_size < 1024);
  const std::size_t local = static_cast<std::size_t>(block_size);
  const sycl::nd_range<1> launch(
      static_cast<std::size_t>(grid_size) * local, local);
  q.submit([&](sycl::handler &h) {
     h.parallel_for<NormKernel4>(launch, [=](sycl::nd_item<1> it) {
       const std::size_t idx = it.get_global_linear_id();
       const std::size_t stride = it.get_global_range(0);
       float acc = 0.0f;
       for (std::size_t i = idx; i < count; i += stride) {
         float v = static_cast<float>(values[i]);
         acc += v * v;
       }
       float block_sum =
           sycl::reduce_over_group(it.get_group(), acc, sycl::plus<float>{});
       if (it.get_local_linear_id() == 0) out[it.get_group_linear_id()] = block_sum;
     });
   }).wait();
  global_norm_aggregate(q, out, static_cast<std::size_t>(grid_size));
}

void global_norm(sycl::queue &q, int kernel_num, float *out, const floatX *values,
                 std::size_t count, int block_size) {
  switch (kernel_num) {
    case 1: return global_norm1(q, out, values, count, block_size);
    case 2: return global_norm2(q, out, values, count, block_size);
    case 3: return global_norm3(q, out, values, count, block_size);
    case 4: return global_norm4(q, out, values, count, block_size);
    default: std::fprintf(stderr, "unknown kernel %d\n", kernel_num); std::exit(2);
  }
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 4) {
    std::fprintf(stderr, "usage: %s <input> <output> N\n", argv[0]);
    return 2;
  }
  const std::size_t N = static_cast<std::size_t>(std::atoi(argv[3]));
  FILE *fin = std::fopen(argv[1], "rb");
  if (!fin) { std::perror(argv[1]); return 2; }
  float *inp = static_cast<float *>(std::malloc(sizeof(float) * N));
  if (!inp) { std::fprintf(stderr, "oom\n"); return 2; }
  if (std::fread(inp, sizeof(float), N, fin) != N) {
    std::fprintf(stderr, "input too short at inp\n");
    return 2;
  }
  std::fclose(fin);
  FILE *fout = std::fopen(argv[2], "wb");
  if (!fout) { std::perror(argv[2]); return 2; }
  for (std::size_t i = 0; i < N; ++i) inp[i] *= 1e-3f;

  try {
    sycl::queue q{sycl::default_selector_v,
                  sycl::property::queue::in_order{}};

    float *d_inp = sycl::malloc_device<float>(N, q);
    float *d_out = sycl::malloc_device<float>(1024, q);
    if (!d_inp || !d_out) { std::fprintf(stderr, "device oom\n"); return 2; }

    q.memcpy(d_inp, inp, sizeof(float) * N).wait();

    float host_out = 0.0f;
    const int variants[] = {1, 2, 3, 4};
    for (int vi = 0; vi < 4; ++vi) {
      const int k = variants[vi];
      q.memset(d_out, 0, sizeof(float) * 1024).wait();
      global_norm(q, k, d_out, d_inp, N, 256);
      q.wait();
      q.memcpy(&host_out, d_out, sizeof(float)).wait();
      std::fwrite(&host_out, sizeof(float), 1, fout);
    }
    sycl::free(d_inp, q);
    sycl::free(d_out, q);
  } catch (const sycl::exception &e) {
    std::fprintf(stderr, "SYCL error: %s\n", e.what());
    return 1;
  }
  std::fclose(fout);
  std::free(inp);
  return 0;
}
