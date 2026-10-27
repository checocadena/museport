/*
Kernels for a global norm.
Global norm in this context means that we want to calculate a single norm cooperatively using all avalailable SMs, instead
 of multiple norms that can be handled by separate blocks.

Compile example:
nvcc -O3 --use_fast_math global_norm.cu -o global_norm
*/

#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <assert.h>

// turn on bf16 as default, done up here for now
// museport: fp32 build (#define ENABLE_BF16 removed)
#include "common.h"

dpct::device_info deviceProp;

// museport: CPU reference global_norm_cpu() removed (it is the hidden check)


template<class T>
void norm_kernel1(float* out, const T* data, size_t count, float *block_result) {
    // we want as few atomics as possible, so each block tries to do
    // the maximum amount of work (so no fixed chunk, but instead iterating
    // until we run out of data), and then we reduce inside the block
    // and finally have just one atomic per block.
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    sycl::group<3> block =
        sycl::ext::oneapi::this_work_item::get_work_group<3>();
    sycl::sub_group warp = sycl::ext::oneapi::this_work_item::get_sub_group();

    // out will be updated atomically from all thread blocks
    size_t index = item_ct1.get_local_id(2) +
                   item_ct1.get_local_range(2) * item_ct1.get_group(2);
    size_t grid_width =
        item_ct1.get_local_range(2) * item_ct1.get_group_range(2);
    float accumulator = 0.f;
    for(size_t i = index; i < count; i += grid_width) {
        accumulator += (float)data[i] * (float)data[i];
    }
    // warp-level reduce
    float warp_result = sycl::reduce_over_group(
        sycl::ext::oneapi::this_work_item::get_sub_group(), accumulator,
        sycl::plus<float>{});
    block_result[sycl::ext::oneapi::this_work_item::get_sub_group()
                     .get_group_linear_id()] = warp_result;
    /*
    DPCT1065:1: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (sycl::ext::oneapi::this_work_item::get_sub_group()
            .get_group_linear_id() == 0) {
        float gather =
            sycl::ext::oneapi::this_work_item::get_sub_group()
                        .get_local_linear_id() <
                    sycl::ext::oneapi::this_work_item::get_sub_group()
                        .get_group_linear_range()
                ? block_result
                      [sycl::ext::oneapi::this_work_item::get_sub_group()
                           .get_local_linear_id()]
                : 0.f;
        float block_sum = sycl::reduce_over_group(
            sycl::ext::oneapi::this_work_item::get_sub_group(), gather,
            sycl::plus<float>{});
        if (sycl::ext::oneapi::this_work_item::get_sub_group()
                .get_local_linear_id() == 0) {
            dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
                out, block_sum);
        }
    }
}

template<class T>
void norm_kernel2(float* out, const T* data, size_t count) {
    // concrete example for an A100 GPU (108 SMs, 2048 max threads each)
    // so there are 2048 * 108 = 221,184 threads total
    // say the block_size is 512, then we would launch 432 blocks in total
    // say num_params is ~100M, each thread will process ~500 elements
    // warps reduce with warp-level reduce, we have 221,184/32 = 6,912 warps
    // and then each warp atomicAdd's to global memory, total of 6,912 atomics

    // no shared memory; but one atomic per warp instead of per block
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    sycl::group<3> block =
        sycl::ext::oneapi::this_work_item::get_work_group<3>();
    sycl::sub_group warp = sycl::ext::oneapi::this_work_item::get_sub_group();

    // out will be updated atomically from all thread blocks
    size_t index = item_ct1.get_local_id(2) +
                   item_ct1.get_local_range(2) * item_ct1.get_group(2);
    size_t grid_width =
        item_ct1.get_local_range(2) * item_ct1.get_group_range(2);
    float accumulator = 0.f;
    for(size_t i = index; i < count; i += grid_width) {
        accumulator += (float)data[i] * (float)data[i];
    }

    // warp-level reduce
    float warp_result = sycl::reduce_over_group(
        sycl::ext::oneapi::this_work_item::get_sub_group(), accumulator,
        sycl::plus<float>{});
    // and atomic in global buffer
    if (sycl::ext::oneapi::this_work_item::get_sub_group()
            .get_local_linear_id() == 0) {
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            out, warp_result);
    }
}

template<class T>
void norm_kernel3(float* out, const T* data, size_t count, float *shared_val) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    size_t index = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                   item_ct1.get_local_id(2);
    size_t grid_width =
        item_ct1.get_local_range(2) * item_ct1.get_group_range(2);
    float accumulator = 0.f;
    for(size_t i = index; i < count; i += grid_width) {
        accumulator += (float)data[i] * (float)data[i];
    }
    // block-level reduce
    float block_sum = blockReduce<warpReduceSum>(accumulator, shared_val);
    if (item_ct1.get_local_id(2) == 0) {
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            out, block_sum);
    }
}

// Same as kernel3 but without atomic adds -> this allows us to have determinism due to the
// non associativity of floating point operations. Roughly same performance as kernel3.
template<class T>
void norm_kernel4(float* out, const T* data, size_t count, float *shared_val) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    size_t index = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                   item_ct1.get_local_id(2);
    size_t grid_width =
        item_ct1.get_local_range(2) * item_ct1.get_group_range(2);
    float accumulator = 0.f;
    for(size_t i = index; i < count; i += grid_width) {
        accumulator += (float)data[i] * (float)data[i];
    }
    // block-level reduce
    float block_sum = blockReduce<warpReduceSum>(accumulator, shared_val);
    // each block accumulates its partial sum to out[blockIdx.x]
    // we want to avoid using atomic add here so we combine this kernel with the aggregate kernel call
    // that sums up the partial block sums
    if (item_ct1.get_local_id(2) == 0) {
        out[item_ct1.get_group(2)] = block_sum;
    }
}

void global_norm_aggregate_kernel(float* out, size_t count, float *shared_val) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    size_t index = item_ct1.get_local_id(2);
    // grab block sums from the previous kernel, use 0. as the neutral sum element
    float block_sum = (index < count) ? out[index] : 0.f;
    float sum = blockReduce<warpReduceSum>(block_sum, shared_val);
    if (item_ct1.get_local_id(2) == 0) {
        out[0] = sum;  // out[0] ends up with the final norm squared
    }
}

// ----------------------------------------------------------------------------
// kernel launchers

template<typename T>
void global_norm1(float* out, const T* values, size_t count, int block_size) {
    // launch just enough blocks to fill the grid. deliberately no DIV_CEIL.
    // having one block less than possible is a tiny performance hit, having
    // one block too many is catastrophic, since it only can start once all the other
    // blocks finish. anyway, I think cuda_threads_per_SM should be a multiple of 512
    // on all gpus, so the division really is going to be exact.
    const int grid_size = cuda_threads_per_SM * cuda_num_SMs / block_size;
    assert(grid_size > 0);      // gives a better error than letting the call below fail
    /*
    DPCT1049:2: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> block_result_acc_ct1(sycl::range<1>(32),
                                                            cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                                  sycl::range<3>(1, 1, block_size),
                              sycl::range<3>(1, 1, block_size)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                norm_kernel1(out, values, count,
                             block_result_acc_ct1
                                 .get_multi_ptr<sycl::access::decorated::no>()
                                 .get());
            });
    });
    /*
    DPCT1010:21: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    cudaCheck(0);
}

template<typename T>
void global_norm2(float* out, const T* values, size_t count, int block_size) {
    // ditto
    const int grid_size = cuda_threads_per_SM * cuda_num_SMs / block_size;
    assert(grid_size > 0);      // gives a better error than letting the call below fail
    /*
    DPCT1049:3: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                              sycl::range<3>(1, 1, block_size),
                          sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
            norm_kernel2(out, values, count);
        });
    /*
    DPCT1010:22: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    cudaCheck(0);
}

template<typename T>
void global_norm3(float* out, const T* values, size_t count, int block_size) {
    // launch just enough blocks to fill the grid. deliberately no DIV_CEIL.
    // having one block less than possible is a tiny performance hit, having
    // one block too many is catastrophic, since it only can start once all the other
    // blocks finish. anyway, I think cuda_threads_per_SM should be a multiple of 512
    // on all gpus, so the division really is going to be exact.
    const int grid_size = deviceProp.get_max_work_items_per_compute_unit() *
                          deviceProp.get_max_compute_units() / block_size;
    assert(grid_size > 0);  // gives a better error than letting the call below fail
    /*
    DPCT1049:4: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
        /*
        DPCT1101:27: 'WARP_SIZE' expression was replaced with a value. Modify
        the code to use the original expression, provided in comments, if it is
        correct.
        */
        sycl::local_accessor<float, 1> shared_val_acc_ct1(
            sycl::range<1>(32 /*WARP_SIZE*/), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                                  sycl::range<3>(1, 1, block_size),
                              sycl::range<3>(1, 1, block_size)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                norm_kernel3(out, values, count,
                             shared_val_acc_ct1
                                 .get_multi_ptr<sycl::access::decorated::no>()
                                 .get());
            });
    });
    /*
    DPCT1010:23: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    cudaCheck(0);
}

template<typename T>
void global_norm4(float* out, const T* values, size_t count, int block_size) {
    if (block_size <= 64) {
        block_size = 128;  // to avoid triggering the assert below
    }
    // launch just enough blocks to fill the grid. deliberately no DIV_CEIL.
    // having one block less than possible is a tiny performance hit, having
    // one block too many is catastrophic, since it only can start once all the other
    // blocks finish. anyway, I think cuda_threads_per_SM should be a multiple of 512
    // on all gpus, so the division really is going to be exact.
    const int grid_size = deviceProp.get_max_work_items_per_compute_unit() *
                          deviceProp.get_max_compute_units() / block_size;
    assert(grid_size > 0);      // gives a better error than letting the call below fail
    assert(grid_size < 1024);  // we want to later accumulate the block sums in a single block
    /*
    DPCT1049:5: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
        /*
        DPCT1101:29: 'WARP_SIZE' expression was replaced with a value. Modify
        the code to use the original expression, provided in comments, if it is
        correct.
        */
        sycl::local_accessor<float, 1> shared_val_acc_ct1(
            sycl::range<1>(32 /*WARP_SIZE*/), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                                  sycl::range<3>(1, 1, block_size),
                              sycl::range<3>(1, 1, block_size)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                norm_kernel4(out, values, count,
                             shared_val_acc_ct1
                                 .get_multi_ptr<sycl::access::decorated::no>()
                                 .get());
            });
    });
    /*
    DPCT1010:24: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    cudaCheck(0);
    /*
    DPCT1049:6: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
        /*
        DPCT1101:30: 'WARP_SIZE' expression was replaced with a value. Modify
        the code to use the original expression, provided in comments, if it is
        correct.
        */
        sycl::local_accessor<float, 1> shared_val_acc_ct1(
            sycl::range<1>(32 /*WARP_SIZE*/), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, 1024),
                              sycl::range<3>(1, 1, 1024)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                global_norm_aggregate_kernel(
                    out, grid_size,
                    shared_val_acc_ct1
                        .get_multi_ptr<sycl::access::decorated::no>()
                        .get());
            });
    });
    /*
    DPCT1010:25: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    cudaCheck(0);
}

void global_norm(int kernel_num, float* out, const floatX* values, size_t count, int block_size) {
    switch (kernel_num) {
        case 1:
            return global_norm1(out, values, count, block_size);
        case 2:
            return global_norm2(out, values, count, block_size);
        case 3:
            return global_norm3(out, values, count, block_size);
        case 4:
            return global_norm4(out, values, count, block_size);
    }
}

// ----------------------------------------------------------------------------
// museport contract main: <input> <output> sizes...; runs every kernel variant and
// writes each variant's outputs, in order, as raw float32.
int main(int argc, char** argv) {
    setup_main();
    if (argc != 4) {
        fprintf(stderr, "usage: %s <input> <output> N\n", argv[0]);
        return 2;
    }
    const int N = atoi(argv[3]);
    FILE* fin = fopen(argv[1], "rb");
    if (!fin) { perror(argv[1]); return 2; }
    float* inp = (float*)malloc(sizeof(float) * (size_t)(N));
    if (fread(inp, sizeof(float), (size_t)(N), fin) != (size_t)(N)) {
        fprintf(stderr, "input too short at inp\n"); return 2;
    }
    fclose(fin);
    FILE* fout = fopen(argv[2], "wb");
    if (!fout) { perror(argv[2]); return 2; }
    for (size_t i = 0; i < (size_t)N; ++i) { inp[i] *= 1e-3; }
    float* d_inp;
    cudaCheck(DPCT_CHECK_ERROR(d_inp = sycl::malloc_device<float>(
                                   (size_t)(N), dpct::get_in_order_queue())));
    cudaCheck(
        DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                             .memcpy(d_inp, inp, sizeof(float) * (size_t)(N))
                             .wait()));
    float* out = (float*)malloc(sizeof(float) * (size_t)(1));
    float* d_out;
    cudaCheck(
        DPCT_CHECK_ERROR(d_out = sycl::malloc_device<float>(
                             (size_t)(1024), dpct::get_in_order_queue())));
    cudaCheck(
        DPCT_CHECK_ERROR(dpct::get_device(0).get_device_info(deviceProp)));
    const int variants[] = {1, 2, 3, 4};
    for (int vi = 0; vi < 4; vi++) {
        const int k = variants[vi];
        cudaCheck(
            DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                 .memset(d_out, 0, sizeof(float) * (size_t)(1))
                                 .wait()));
        global_norm(k, d_out, d_inp, (size_t)N, 256);
        /*
        DPCT1010:26: SYCL uses exceptions to report errors and does not use the
        error codes. The cudaGetLastError function call was replaced with 0. You
        need to rewrite this code.
        */
        cudaCheck(0);
        cudaCheck(DPCT_CHECK_ERROR(
            dpct::get_current_device().queues_wait_and_throw()));
        cudaCheck(DPCT_CHECK_ERROR(
            dpct::get_in_order_queue()
                .memcpy(out, d_out, sizeof(float) * (size_t)(1))
                .wait()));
        fwrite(out, sizeof(float), (size_t)(1), fout);
    }
    fclose(fout);
    return 0;
}
