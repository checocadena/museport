#include <sycl/sycl.hpp>

#include <cassert>
#include <cstddef>
#include <cstdio>
#include <cstdlib>

using floatX = float;

// Distinct kernel name tags: one SYCL kernel per variant.
class NormKernel1;
class NormKernel2;
class NormKernel3;
class NormKernel4Stage1;
class NormAggregate;

namespace {

int pick_grid_size(sycl::queue &q, int block_size) {
    std::size_t cus = 0;
    try {
        cus = q.get_device().get_info<sycl::info::device::max_compute_units>();
    } catch (...) {
        cus = 0;
    }
    if (cus == 0) cus = 8;
    long g = static_cast<long>(cus) * 16L;
    if (g < 8) g = 8;
    // Keep small enough that the single-group aggregate can cover it
    // with a portable work-group size.
    if (g > 512) g = 512;
    (void)block_size;
    return static_cast<int>(g);
}

int effective_block_size(sycl::queue &q, int want) {
    std::size_t max_wg = 0;
    try {
        max_wg = q.get_device().get_info<sycl::info::device::max_work_group_size>();
    } catch (...) {
        max_wg = 0;
    }
    if (max_wg != 0 && static_cast<std::size_t>(want) > max_wg) {
        return static_cast<int>(max_wg);
    }
    return want;
}

using GlobalAtomic = sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                      sycl::memory_scope::device,
                                      sycl::access::address_space::global_space>;

// Variant 1: grid-stride accumulation, one atomic add per work-group.
void global_norm1(sycl::queue &q, float *out, const floatX *values, std::size_t count,
                  int block_size) {
    const int grid_size = pick_grid_size(q, block_size);
    assert(grid_size > 0);
    const int bs = effective_block_size(q, block_size);
    q.parallel_for<NormKernel1>(
         sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(grid_size) *
                                          static_cast<std::size_t>(bs)),
                           sycl::range<1>(static_cast<std::size_t>(bs))),
         [=](sycl::nd_item<1> it) {
             sycl::group<1> g = it.get_group();
             const std::size_t lid = it.get_local_id(0);
             const std::size_t lrange = it.get_local_range(0);
             const std::size_t gid = it.get_group(0);
             const std::size_t grange = it.get_group_range(0);
             const std::size_t index = lid + lrange * gid;
             const std::size_t stride = lrange * grange;
             float acc = 0.0f;
             for (std::size_t i = index; i < count; i += stride) {
                 float v = static_cast<float>(values[i]);
                 acc += v * v;
             }
             float block_sum =
                 sycl::reduce_over_group(g, acc, sycl::plus<float>{});
             if (lid == 0) {
                 GlobalAtomic atom(out[0]);
                 atom.fetch_add(block_sum);
             }
         })
        .wait();
}

// Variant 2: grid-stride accumulation, one atomic add per sub-group.
void global_norm2(sycl::queue &q, float *out, const floatX *values, std::size_t count,
                  int block_size) {
    const int grid_size = pick_grid_size(q, block_size);
    assert(grid_size > 0);
    const int bs = effective_block_size(q, block_size);
    q.parallel_for<NormKernel2>(
         sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(grid_size) *
                                          static_cast<std::size_t>(bs)),
                           sycl::range<1>(static_cast<std::size_t>(bs))),
         [=](sycl::nd_item<1> it) {
             sycl::sub_group sg = it.get_sub_group();
             const std::size_t lid = it.get_local_id(0);
             const std::size_t lrange = it.get_local_range(0);
             const std::size_t gid = it.get_group(0);
             const std::size_t grange = it.get_group_range(0);
             const std::size_t index = lid + lrange * gid;
             const std::size_t stride = lrange * grange;
             float acc = 0.0f;
             for (std::size_t i = index; i < count; i += stride) {
                 float v = static_cast<float>(values[i]);
                 acc += v * v;
             }
             float sg_sum =
                 sycl::reduce_over_group(sg, acc, sycl::plus<float>{});
             if (sg.leader()) {
                 GlobalAtomic atom(out[0]);
                 atom.fetch_add(sg_sum);
             }
         })
        .wait();
}

// Variant 3: grid-stride accumulation, work-group reduction via sub-group
// partials staged through local memory, one atomic per work-group.
void global_norm3(sycl::queue &q, float *out, const floatX *values, std::size_t count,
                  int block_size) {
    const int grid_size = pick_grid_size(q, block_size);
    assert(grid_size > 0);
    const int bs = effective_block_size(q, block_size);
    const sycl::range<1> global_range(static_cast<std::size_t>(grid_size) *
                                      static_cast<std::size_t>(bs));
    const sycl::range<1> local_range(static_cast<std::size_t>(bs));
    // Upper bound on sub-groups per work-group for the scratch allocation.
    const std::size_t scratch_elems = static_cast<std::size_t>(bs);
    q.submit([&](sycl::handler &cgh) {
         sycl::local_accessor<float, 1> scratch(sycl::range<1>(scratch_elems), cgh);
         cgh.parallel_for<NormKernel3>(
             sycl::nd_range<1>(global_range, local_range),
             [=](sycl::nd_item<1> it) {
                 sycl::group<1> g = it.get_group();
                 sycl::sub_group sg = it.get_sub_group();
                 const std::size_t lid = it.get_local_id(0);
                 const std::size_t lrange = it.get_local_range(0);
                 const std::size_t gid = it.get_group(0);
                 const std::size_t grange = it.get_group_range(0);
                 const std::size_t index = lid + lrange * gid;
                 const std::size_t stride = lrange * grange;
                 float acc = 0.0f;
                 for (std::size_t i = index; i < count; i += stride) {
                     float v = static_cast<float>(values[i]);
                     acc += v * v;
                 }
                 float sg_sum =
                     sycl::reduce_over_group(sg, acc, sycl::plus<float>{});
                 const std::size_t sg_id = sg.get_group_linear_id();
                 const std::size_t num_sgs = sg.get_group_linear_range();
                 const std::size_t sg_lid = sg.get_local_linear_id();
                 if (sg_lid == 0) {
                     scratch[sg_id] = sg_sum;
                 }
                 sycl::group_barrier(g);
                 float gather = 0.0f;
                 // First sub-group gathers the per-sub-group partials.
                 if (sg_id == 0 && sg_lid < num_sgs) {
                     gather = scratch[sg_lid];
                 }
                 float block_sum =
                     sycl::reduce_over_group(sg, gather, sycl::plus<float>{});
                 // Broadcast the leader sub-group's sum to the group.
                 float leader_sum = sycl::group_broadcast(g, block_sum, 0);
                 if (lid == 0) {
                     GlobalAtomic atom(out[0]);
                     atom.fetch_add(leader_sum);
                 }
             });
     })
        .wait();
}

// Variant 4: deterministic two-stage reduction. Stage 1 writes one partial
// per work-group; stage 2 sums the partials in a single work-group.
void global_norm4(sycl::queue &q, float *out, const floatX *values, std::size_t count,
                  int block_size) {
    if (block_size <= 64) {
        block_size = 128;
    }
    const int grid_size = pick_grid_size(q, block_size);
    assert(grid_size > 0);
    assert(grid_size < 1024);
    const int bs = effective_block_size(q, block_size);
    q.parallel_for<NormKernel4Stage1>(
         sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(grid_size) *
                                          static_cast<std::size_t>(bs)),
                           sycl::range<1>(static_cast<std::size_t>(bs))),
         [=](sycl::nd_item<1> it) {
             sycl::group<1> g = it.get_group();
             const std::size_t lid = it.get_local_id(0);
             const std::size_t lrange = it.get_local_range(0);
             const std::size_t gid = it.get_group(0);
             const std::size_t grange = it.get_group_range(0);
             const std::size_t index = lid + lrange * gid;
             const std::size_t stride = lrange * grange;
             float acc = 0.0f;
             for (std::size_t i = index; i < count; i += stride) {
                 float v = static_cast<float>(values[i]);
                 acc += v * v;
             }
             float block_sum =
                 sycl::reduce_over_group(g, acc, sycl::plus<float>{});
             if (lid == 0) {
                 out[gid] = block_sum;
             }
         })
        .wait();

    const int agg_block = effective_block_size(q, 256);
    q.parallel_for<NormAggregate>(
         sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(agg_block)),
                           sycl::range<1>(static_cast<std::size_t>(agg_block))),
         [=](sycl::nd_item<1> it) {
             sycl::group<1> g = it.get_group();
             const std::size_t lid = it.get_local_id(0);
             const std::size_t lrange = it.get_local_range(0);
             float acc = 0.0f;
             for (std::size_t i = lid; i < static_cast<std::size_t>(grid_size);
                  i += lrange) {
                 acc += out[i];
             }
             float sum =
                 sycl::reduce_over_group(g, acc, sycl::plus<float>{});
             if (lid == 0) {
                 out[0] = sum;
             }
         })
        .wait();
}

void global_norm(sycl::queue &q, int kernel_num, float *out, const floatX *values,
                 std::size_t count, int block_size) {
    switch (kernel_num) {
        case 1:
            return global_norm1(q, out, values, count, block_size);
        case 2:
            return global_norm2(q, out, values, count, block_size);
        case 3:
            return global_norm3(q, out, values, count, block_size);
        case 4:
            return global_norm4(q, out, values, count, block_size);
        default:
            break;
    }
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s <input> <output> N\n", argv[0]);
        return 2;
    }
    const int N = std::atoi(argv[3]);
    if (N <= 0) {
        std::fprintf(stderr, "bad N\n");
        return 2;
    }
    FILE *fin = std::fopen(argv[1], "rb");
    if (!fin) {
        std::perror(argv[1]);
        return 2;
    }
    float *inp = static_cast<float *>(std::malloc(sizeof(float) * static_cast<std::size_t>(N)));
    if (!inp) return 2;
    if (std::fread(inp, sizeof(float), static_cast<std::size_t>(N), fin) !=
        static_cast<std::size_t>(N)) {
        std::fprintf(stderr, "input too short at inp\n");
        return 2;
    }
    std::fclose(fin);
    FILE *fout = std::fopen(argv[2], "wb");
    if (!fout) {
        std::perror(argv[2]);
        return 2;
    }
    for (std::size_t i = 0; i < static_cast<std::size_t>(N); ++i) {
        inp[i] *= 1e-3f;
    }

    sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order{}};

    float *d_inp = sycl::malloc_device<float>(static_cast<std::size_t>(N), q);
    float *d_out = sycl::malloc_device<float>(1024, q);
    if (!d_inp || !d_out) return 2;
    q.memcpy(d_inp, inp, sizeof(float) * static_cast<std::size_t>(N)).wait();

    float host_out[1] = {0.0f};
    const int variants[] = {1, 2, 3, 4};
    for (int vi = 0; vi < 4; ++vi) {
        const int k = variants[vi];
        q.fill(d_out, 0.0f, 1).wait();
        global_norm(q, k, d_out, d_inp, static_cast<std::size_t>(N), 256);
        q.wait();
        q.memcpy(host_out, d_out, sizeof(float)).wait();
        std::fwrite(host_out, sizeof(float), 1, fout);
    }
    std::fclose(fout);
    sycl::free(d_inp, q);
    sycl::free(d_out, q);
    std::free(inp);
    return 0;
}
