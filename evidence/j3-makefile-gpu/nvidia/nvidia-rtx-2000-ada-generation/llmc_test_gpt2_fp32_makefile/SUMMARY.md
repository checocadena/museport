# GPU test: llmc_test_gpt2_fp32_makefile

- **Result:** `ok`: PASS on NVIDIA RTX 2000 Ada Generation (cuda:0, nvidia): sycl_verify_vs_reference: PASS: 1/1 case(s) match (worst max abs err 0.000126, atol 0.01); sycl_verify_vs_native: PASS: 1/1 case(s) match (worst max abs err 5.01e-06, atol 0.01)
- **Device:** NVIDIA RTX 2000 Ada Generation (`cuda:0`), vendor `nvidia`
- **SYCL target flags:** `-fsycl-targets=nvidia_gpu_sm_89`

| step | status | summary |
|---|---|---|
| reference_build | `ok` | exit 0: true |
| native_build | `ok` | exit 0: /usr/local/cuda/bin/nvcc -O2 -std=c++17 -arch=native -I/tmp/tmp.v0I4ZAYi3I/task /tmp/tmp.v0I4ZAYi3I/task/test_gpt2_fp32.cu -lcublas -o build/native |
| native_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 0.000121, atol 0.01) |
| sycl_compile | `ok` | compiled 1 source(s) -> /tmp/tmp.v0I4ZAYi3I/out/work/build/sycl (0 warning(s)) |
| sycl_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 0.000126, atol 0.01) |
| sycl_verify_vs_native | `ok` | PASS: 1/1 case(s) match (worst max abs err 5.01e-06, atol 0.01) |
| sycl_benchmark | `ok` | 10 run(s) each; cuda:0: median 2639.51 ms, p95 2706.98 ms |
| native_benchmark | `ok` | 10 run(s) each; none: median 2533.35 ms, p95 2563.93 ms |

Limitations:
- Benchmarks are single-process wall times plus program-reported kernel times; treat them as rough.
