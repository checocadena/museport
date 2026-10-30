# GPU test: onemath_cublas_bf16_gemm_unit_tests

- **Result:** `ok`: PASS on NVIDIA L4 (cuda:0, nvidia): sycl_verify_vs_reference: PASS: 1/1 case(s) match (worst max abs err 0, atol 0)
- **Device:** NVIDIA L4 (`cuda:0`), vendor `nvidia`
- **SYCL target flags:** `-fsycl-targets=nvidia_gpu_sm_89`

| step | status | summary |
|---|---|---|
| reference_build | `ok` | exit 0: true |
| native_build | `fail` | exit 1: /usr/local/cuda/bin/nvcc -O2 -std=c++17 -arch=native /tmp/tmp.Jxl4sHUZMc/task/write_one.cpp -o build/native |
| sycl_compile | `ok` | compiled 1 source(s) -> /tmp/tmp.Jxl4sHUZMc/out/work/build/sycl (0 warning(s)) |
| sycl_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 0, atol 0) |

Limitations:
- Benchmarks are single-process wall times plus program-reported kernel times; treat them as rough.
