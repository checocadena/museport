# GPU test: llmc_fused_residual_forward

- **Result:** `ok`: PASS on NVIDIA RTX A4500 (cuda:0, nvidia): sycl_verify_vs_reference: PASS: 1/1 case(s) match (worst max abs err 1.43e-06, atol 1e-05); sycl_verify_vs_native: PASS: 1/1 case(s) match (worst max abs err 1.43e-06, atol 1e-05)
- **Device:** NVIDIA RTX A4500 (`cuda:0`), vendor `nvidia`
- **SYCL target flags:** `-fsycl-targets=nvidia_gpu_sm_86`

| step | status | summary |
|---|---|---|
| reference_build | `ok` | exit 0: g++ -O2 -std=c++17 /tmp/tmp.JRoQE1ZsMO/task/fused_residual_forward_ref.cpp -o build/reference |
| native_build | `ok` | exit 0: /usr/local/cuda/bin/nvcc -O2 -std=c++17 -arch=native -I/tmp/tmp.JRoQE1ZsMO/task /tmp/tmp.JRoQE1ZsMO/task/fused_residual_forward_harness.cu -o build/native |
| native_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 1.43e-06, atol 1e-05) |
| sycl_compile | `ok` | compiled 1 source(s) -> /tmp/tmp.JRoQE1ZsMO/out/work/build/sycl (12 warning(s)) |
| sycl_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 1.43e-06, atol 1e-05) |
| sycl_verify_vs_native | `ok` | PASS: 1/1 case(s) match (worst max abs err 1.43e-06, atol 1e-05) |
| sycl_benchmark | `ok` | 10 run(s) each; cuda:0: median 316.12 ms, p95 376.74 ms |
| native_benchmark | `ok` | 10 run(s) each; none: median 336.38 ms, p95 384.46 ms |

Limitations:
- Benchmarks are single-process wall times plus program-reported kernel times; treat them as rough.
