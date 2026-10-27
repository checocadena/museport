# GPU test: llmc_layernorm_backward

- **Result:** `ok`: PASS on NVIDIA L4 (cuda:0, nvidia): sycl_verify_vs_reference: PASS: 1/1 case(s) match (worst max abs err 2.48e-05, atol 0.001); sycl_verify_vs_native: PASS: 1/1 case(s) match (worst max abs err 2.1e-05, atol 0.001)
- **Device:** NVIDIA L4 (`cuda:0`), vendor `nvidia`
- **SYCL target flags:** `-fsycl-targets=nvidia_gpu_sm_89`

| step | status | summary |
|---|---|---|
| reference_build | `ok` | exit 0: g++ -O2 -std=c++17 /tmp/tmp.wZmlkuELk4/task/layernorm_backward_ref.cpp -o build/reference |
| native_build | `ok` | exit 0: /usr/local/cuda/bin/nvcc -O2 -std=c++17 -arch=native -I/tmp/tmp.wZmlkuELk4/task /tmp/tmp.wZmlkuELk4/task/layernorm_backward_harness.cu -o build/native |
| native_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 1.72e-05, atol 0.001) |
| sycl_compile | `ok` | compiled 1 source(s) -> /tmp/tmp.wZmlkuELk4/out/work/build/sycl (0 warning(s)) |
| sycl_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 2.48e-05, atol 0.001) |
| sycl_verify_vs_native | `ok` | PASS: 1/1 case(s) match (worst max abs err 2.1e-05, atol 0.001) |
| sycl_benchmark | `ok` | 10 run(s) each; cuda:0: median 353.78 ms, p95 378.19 ms |
| native_benchmark | `ok` | 10 run(s) each; none: median 354.04 ms, p95 390.27 ms |

Limitations:
- Benchmarks are single-process wall times plus program-reported kernel times; treat them as rough.
