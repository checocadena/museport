# GPU test: llmc_test_gpt2_bf16

- **Result:** `fail`: FAIL on NVIDIA L4 (cuda:0, nvidia): sycl_verify_vs_reference: PASS: 1/1 case(s) match (worst max abs err 0.0403, atol 0.05); sycl_verify_vs_native: FAIL: 0/1 case(s) match; first failure: 1 of 12 element(s) outside tolerance
- **Device:** NVIDIA L4 (`cuda:0`), vendor `nvidia`
- **SYCL target flags:** `-fsycl-targets=nvidia_gpu_sm_89`

| step | status | summary |
|---|---|---|
| reference_build | `ok` | exit 0: true |
| native_build | `ok` | exit 0: /usr/local/cuda/bin/nvcc -O2 -std=c++17 -arch=native -I/tmp/tmp.Gkq64zdIQQ/task /tmp/tmp.Gkq64zdIQQ/task/test_gpt2.cu --use_fast_math -DENABLE_BF16 -lcublas -lcublasLt -L/usr/local/cuda/lib64/stubs -lnvidia-ml -o build/native |
| native_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 0.0272, atol 0.05) |
| sycl_compile | `ok` | compiled 1 source(s) -> /tmp/tmp.Gkq64zdIQQ/out/work/build/sycl (0 warning(s)) |
| sycl_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 0.0403, atol 0.05) |
| sycl_verify_vs_native | `fail` | FAIL: 0/1 case(s) match; first failure: 1 of 12 element(s) outside tolerance |

Limitations:
- Benchmarks are single-process wall times plus program-reported kernel times; treat them as rough.
