# GPU test: llmc_global_norm

- **Result:** `ok`: PASS on NVIDIA RTX A4500 (cuda:0, nvidia): sycl_verify_vs_reference: PASS: 1/1 case(s) match (worst max abs err 5.96e-07, atol 0.01); sycl_verify_vs_native: PASS: 1/1 case(s) match (worst max abs err 1.55e-06, atol 0.01)
- **Device:** NVIDIA RTX A4500 (`cuda:0`), vendor `nvidia`
- **SYCL target flags:** `-fsycl-targets=nvidia_gpu_sm_86`

| step | status | summary |
|---|---|---|
| reference_build | `ok` | exit 0: g++ -O2 -std=c++17 /tmp/tmp.lbyN18z6tB/task/global_norm_ref.cpp -o build/reference |
| native_build | `ok` | exit 0: /usr/local/cuda/bin/nvcc -O2 -std=c++17 -arch=native -I/tmp/tmp.lbyN18z6tB/task /tmp/tmp.lbyN18z6tB/task/global_norm_harness.cu -o build/native |
| native_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 1.31e-06, atol 0.01) |
| sycl_compile | `ok` | compiled 1 source(s) -> /tmp/tmp.lbyN18z6tB/out/work/build/sycl (0 warning(s)) |
| sycl_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 5.96e-07, atol 0.01) |
| sycl_verify_vs_native | `ok` | PASS: 1/1 case(s) match (worst max abs err 1.55e-06, atol 0.01) |
| sycl_benchmark | `ok` | 10 run(s) each; cuda:0: median 344.10 ms, p95 421.58 ms |
| native_benchmark | `ok` | 10 run(s) each; none: median 393.38 ms, p95 433.28 ms |

Limitations:
- Benchmarks are single-process wall times plus program-reported kernel times; treat them as rough.
