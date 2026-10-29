/*
AdamW optimizer. SYCL 2020 port of llmc/adamw.cuh. Single device.
*/

// llmc internal imports
#include "sycl_common.h"
#include "sycl_utils.h"

// ----------------------------------------------------------------------------
// SYCL kernels

// Implements linear interpolation using only two floating-point operations (as opposed to three in a naive implementation).
// Reference: https://developer.nvidia.com/blog/lerp-faster-cuda
inline float adamw_lerp(float start, float end, float weight) {
    return sycl::fma(weight, end, sycl::fma(-weight, start, start));
}

template <typename Tp, typename Tg>
inline void adamw_update_one(Tp* params_memory, float* master_params_memory, Tg* grads_memory,
                             float* m_memory, float* v_memory, size_t num_parameters,
                             float learning_rate, float beta1, float beta2,
                             float beta1_correction, float beta2_correction,
                             float eps, float weight_decay,
                             float grad_scale, unsigned int seed,
                             int gx, int gy, ptrdiff_t w_stride, ptrdiff_t g_stride, ptrdiff_t s_stride) {
    if (num_parameters == 0) { return; }
    constexpr size_t BLOCK = 512;
    size_t total = num_parameters * (size_t)(gy > 0 ? gy : 1);
    sycl_queue.parallel_for(sycl::nd_range<1>(sycl::range<1>(CEIL_DIV(total, BLOCK) * BLOCK),
                                              sycl::range<1>(BLOCK)),
        [=](sycl::nd_item<1> it) {
            size_t lin = it.get_global_linear_id();
            if (lin >= total) { return; }
            size_t idx = lin % num_parameters;
            size_t slice = lin / num_parameters;
            Tp* pm = params_memory + slice * (size_t)w_stride;
            float* mm = master_params_memory ? master_params_memory + slice * (size_t)s_stride : nullptr;
            Tg* gm = grads_memory + slice * (size_t)g_stride;
            float* m_mem = m_memory + slice * (size_t)s_stride;
            float* v_mem = v_memory + slice * (size_t)s_stride;
            // get the gradient, m, and v for this parameter
            float grad = grad_scale * (float)gm[idx];
            float m = m_mem[idx];
            float v = v_mem[idx];
            // update the first moment (momentum)
            m = adamw_lerp(grad, m, beta1);
            m_mem[idx] = m;
            // update the second moment (RMSprop)
            v = adamw_lerp(grad * grad, v, beta2);
            v_mem[idx] = v;
            m /= beta1_correction;  // m_hat
            v /= beta2_correction;  // v_hat
            // fetch the old value of this parameter as a float, from either source
            float old_param = (mm != nullptr) ? mm[idx] : (float)pm[idx];
            // update this parameter
            float param = old_param - (learning_rate * (m / (sycl::sqrt(v) + eps) + weight_decay * old_param));
            // update our low precision version of the parameters using stochastic rounding
            // this will be used in the next forward pass
            // NOTE: CUDA indexes RNG by (threadIdx.x [0,512), blockIdx.x*blockDim.x + blockIdx.y);
            // here global id == idx + slice*num_parameters may exceed that scheme, but the
            // requirement is only determinism + unique-ish seeds; replicate CUDA exactly for
            // the common single-slice case: pos_x = idx % 512, pos_y = idx/512 (+ slice grid dim)
            int pos_x = (int)(idx % 512);
            int pos_y = (int)(idx / 512 + slice * ((num_parameters + 511) / 512));
            stochastic_rounding(param, &pm[idx], seed, pos_x, pos_y);
            // write the full, float version of the param into our master copy, if we maintain one
            if (mm != nullptr) { mm[idx] = param; }
        }).wait();
    (void)gx;
}

template <typename Tp>
inline void init_from_master_one(Tp* params_memory, float* master_params_memory, size_t num_parameters,
                                 ptrdiff_t w_stride, ptrdiff_t s_stride, int num_slices, unsigned int seed) {
    if (num_parameters == 0) { return; }
    constexpr size_t BLOCK = 512; // must match block size of adamw_update so that RNG also matches
    size_t total = num_parameters * (size_t)(num_slices > 0 ? num_slices : 1);
    sycl_queue.parallel_for(sycl::nd_range<1>(sycl::range<1>(CEIL_DIV(total, BLOCK) * BLOCK),
                                              sycl::range<1>(BLOCK)),
        [=](sycl::nd_item<1> it) {
            size_t lin = it.get_global_linear_id();
            if (lin >= total) { return; }
            size_t idx = lin % num_parameters;
            size_t slice = lin / num_parameters;
            Tp* pm = params_memory + slice * (size_t)w_stride;
            float* mm = master_params_memory + slice * (size_t)s_stride;
            int pos_x = (int)(idx % 512);
            int pos_y = (int)(idx / 512 + slice * ((num_parameters + 511) / 512));
            stochastic_rounding(mm[idx], &pm[idx], seed, pos_x, pos_y);
        }).wait();
}

template <typename Tp, typename Tg>
inline void adamw_update(Tp* params_memory, float* master_params_memory, Tg* grads_memory,
                         float* m_memory, float* v_memory, size_t num_parameters,
                         ptrdiff_t w_stride, ptrdiff_t g_stride, ptrdiff_t s_stride, int num_slices,
                         float learning_rate, float beta1, float beta2, int t,
                         float eps, float weight_decay,
                         float grad_scale, unsigned int seed) {
    // AdamW update
    float beta1_correction = 1.0f - powf(beta1, t);
    float beta2_correction = 1.0f - powf(beta2, t);
    adamw_update_one(params_memory, master_params_memory, grads_memory, m_memory, v_memory,
                     num_parameters, learning_rate, beta1, beta2, beta1_correction, beta2_correction,
                     eps, weight_decay, grad_scale, seed, 0, num_slices, w_stride, g_stride, s_stride);
}

template <typename Tp>
inline void init_from_master(Tp* params_memory, float* master_params_memory, size_t num_parameters,
                             ptrdiff_t w_stride, ptrdiff_t s_stride, int num_slices, unsigned int seed) {
    init_from_master_one(params_memory, master_params_memory, num_parameters, w_stride, s_stride,
                         num_slices, seed);
}
