# Tier 1 run-2 ports on an AMD GPU

- **Date:** 2026-10-26
- **Device:** AMD Instinct MI350X VF, gfx950 (`AMD Instinct MI350X VF gfx950:sramecc+:xnack- [HIP 71460.85]`, SYCL selector `hip:0`)
- **Machine:** a DigitalOcean GPU droplet (`gpu-mi350x1-288gb-spot`, $3.50/h), one droplet per port, created and deleted by `oneapi-agent remote-test --provider digitalocean --vendor amd --max-usd 1`, run from a laptop. Every run logged `deleted digitalocean machine <id>: confirmed`, and the DigitalOcean API showed 0 droplets and 0 SSH keys after each run.
- **Toolkit:** oneapi-agent-toolkit `e260526` (branch `hipify-local-headers`, PR #22). Run 1 of gelu_forward used the version museport pins (`e693601`).
- **SYCL compiler:** `sycl-hip-nightly-2026-10-14-rocm7.0-r3` (sha256 `e9bf82ee…c8fc3`)
- **Check:** each SYCL port is checked against the hidden C++ reference only (see [Native CUDA comparison on AMD](#native-cuda-comparison-on-amd)).

## Results

| Port | sycl_verify_vs_reference | sycl_compile | Minutes | est_cost_usd |
|---|---|---|---|---|
| t1-adamw | PASS 1/1 (worst max abs err 5.96e-08, atol 1e-4) | ok (0 warnings) | 2.3 | 0.133 |
| t1-crossentropy-forward | PASS 1/1 (worst max abs err 1.19e-07, atol 1e-5) | ok (0 warnings) | 2.2 | 0.126 |
| t1-crossentropy-softmax-backward | PASS 1/1 (worst max abs err 0, atol 1e-5) | ok (0 warnings) | 2.4 | 0.138 |
| t1-encoder-backward | PASS 1/1 (worst max abs err 4.77e-07, atol 1e-5) | ok (0 warnings) | 2.2 | 0.127 |
| t1-encoder-forward | PASS 1/1 (worst max abs err 0, atol 1e-5) | ok (0 warnings) | 2.2 | 0.126 |
| t1-fused-residual-forward | PASS 1/1 (worst max abs err 1.67e-06, atol 1e-5) | ok (12 warnings) | 2.4 | 0.137 |
| t1-gelu-backward | PASS 1/1 (worst max abs err 2.38e-07, atol 1e-5) | ok (0 warnings) | 2.2 | 0.126 |
| t1-gelu-forward | PASS 1/1 (worst max abs err 1.19e-07, atol 1e-5) | ok (0 warnings) | 2.0 | 0.118 |
| t1-permute | PASS 1/1 (worst max abs err 0, atol 1e-5) | ok (0 warnings) | 2.1 | 0.123 |
| t1-residual-forward | PASS 1/1 (worst max abs err 0, atol 1e-5) | ok (0 warnings) | 2.4 | 0.138 |

**10/10 ports PASS against the reference.**

## Cost

`est_cost_usd` is the toolkit's estimate (price per hour × minutes from create to delete), not a DigitalOcean invoice.

- The 10 runs in the table: **$1.292**
- Plus the first gelu_forward run (replaced by the one above): $0.127
- **Total: $1.419**

## Native CUDA comparison on AMD

The toolkit also tries to build llm.c's original CUDA for the AMD GPU (hipify, then hipcc) and to compare the SYCL port against it. On AMD that build failed, so `sycl_verify_vs_native` never ran. `native_build` reported `fail` in all 10 runs; the errors below are from gelu_forward.

**First run (toolkit `e693601`):** only the `.cu` file was hipified, not the `common.h` it includes:

```
/tmp/tmp.c5mHhD2qX0/task/common.h:3:10: fatal error: 'cuda_runtime.h' file not found
    3 | #include <cuda_runtime.h>
```

Fixed by toolkit PR #22, which also hipifies the source's local headers.

**Second run (toolkit `e260526`):** `common.h` is hipified, but the hipified code doesn't compile under ROCm 7.14 (`3 warnings and 3 errors generated when compiling for gfx950`):

```
/tmp/tmp.FaiUvhtfyc/out/work/common.h:150:35: error: use of undeclared identifier '__ldcs'; did you mean '__ldg'?
  150 |     return Packed128<ElementType>{__ldcs(reinterpret_cast<const int4*>(address))};
```
```
/tmp/tmp.FaiUvhtfyc/out/work/common.h:238:5: error: no matching function for call to 'memset'
  238 |     memset(arr, 0, N * sizeof(float)); // all zero
/opt/rocm/core-7.14/include/hip/amd_detail/amd_device_functions.h:892:32: note: candidate function not viable: call to __device__ function from __host__ function
```
```
/opt/rocm/core-7.14/include/hip/amd_detail/amd_warp_sync_functions.h:298:62: error: static assertion failed due to requirement 'sizeof(unsigned int) == 8': The mask must be a 64-bit integer. Implicitly promoting a smaller integer is almost always an error.
/tmp/tmp.FaiUvhtfyc/out/work/common.h:17:16: note: in instantiation of function template specialization '__shfl_xor_sync<unsigned int, float>' requested here
   17 |         val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
```

**Conclusion:** llm.c's CUDA doesn't convert to HIP automatically, so on AMD the ports are checked against the hidden C++ reference only. The comparison with llm.c's CUDA was done on NVIDIA.
