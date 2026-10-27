# GPU test: llmc_layernorm_forward

- **Result:** `ok`: PASS on NVIDIA RTX A4500 (cuda:0, nvidia): sycl_verify_vs_reference: PASS: 1/1 case(s) match (worst max abs err 9.54e-07, atol 1e-05); sycl_verify_vs_native: PASS: 1/1 case(s) match (worst max abs err 3.58e-07, atol 1e-05)
- **Device:** NVIDIA RTX A4500 (`cuda:0`), vendor `nvidia`
- **SYCL target flags:** `-fsycl-targets=nvidia_gpu_sm_86`

| step | status | summary |
|---|---|---|
| reference_build | `ok` | exit 0: g++ -O2 -std=c++17 /tmp/tmp.ZDAEDL8DCr/task/layernorm_forward_ref.cpp -o build/reference |
| native_build | `ok` | exit 0: /usr/local/cuda/bin/nvcc -O2 -std=c++17 -arch=native -I/tmp/tmp.ZDAEDL8DCr/task /tmp/tmp.ZDAEDL8DCr/task/layernorm_forward_harness.cu -o build/native |
| native_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 1.19e-06, atol 1e-05) |
| sycl_compile | `ok` | compiled 1 source(s) -> /tmp/tmp.ZDAEDL8DCr/out/work/build/sycl (30 warning(s)) |
| sycl_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 9.54e-07, atol 1e-05) |
| sycl_verify_vs_native | `ok` | PASS: 1/1 case(s) match (worst max abs err 3.58e-07, atol 1e-05) |
| sycl_benchmark | `ok` | 10 run(s) each; cuda:0: median 341.01 ms, p95 385.97 ms |
| native_benchmark | `ok` | 10 run(s) each; none: median 342.67 ms, p95 382.26 ms |

Limitations:
- Benchmarks are single-process wall times plus program-reported kernel times; treat them as rough.
