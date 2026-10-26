# GPU test: llmc_layernorm_forward

- **Result:** `ok`: PASS on NVIDIA RTX 2000 Ada Generation (cuda:0, nvidia): sycl_verify_vs_reference: PASS: 1/1 case(s) match (worst max abs err 9.54e-07, atol 1e-05); sycl_verify_vs_native: PASS: 1/1 case(s) match (worst max abs err 3.58e-07, atol 1e-05)
- **Device:** NVIDIA RTX 2000 Ada Generation (`cuda:0`), vendor `nvidia`
- **SYCL target flags:** `-fsycl-targets=nvidia_gpu_sm_89`

| step | status | summary |
|---|---|---|
| reference_build | `ok` | exit 0: g++ -O2 -std=c++17 /tmp/tmp.rVHhKs79t7/task/layernorm_forward_ref.cpp -o build/reference |
| native_build | `ok` | exit 0: /usr/local/cuda/bin/nvcc -O2 -std=c++17 -arch=native -I/tmp/tmp.rVHhKs79t7/task /tmp/tmp.rVHhKs79t7/task/layernorm_forward_harness.cu -o build/native |
| native_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 1.19e-06, atol 1e-05) |
| sycl_compile | `ok` | compiled 1 source(s) -> /tmp/tmp.rVHhKs79t7/out/work/build/sycl (0 warning(s)) |
| sycl_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 9.54e-07, atol 1e-05) |
| sycl_verify_vs_native | `ok` | PASS: 1/1 case(s) match (worst max abs err 3.58e-07, atol 1e-05) |
| sycl_benchmark | `ok` | 10 run(s) each; cuda:0: median 265.09 ms, p95 282.82 ms |
| native_benchmark | `ok` | 10 run(s) each; none: median 264.41 ms, p95 346.49 ms |

Limitations:
- Benchmarks are single-process wall times plus program-reported kernel times; treat them as rough.
