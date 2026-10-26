# GPU test: llmc_trimat_forward_harnesscheck

- **Result:** `ok`: PASS on NVIDIA RTX 4000 Ada Generation (cuda:0, nvidia): sycl_verify_vs_reference: PASS: 1/1 case(s) match (worst max abs err 0, atol 0.0001); sycl_verify_vs_native: PASS: 1/1 case(s) match (worst max abs err 5.96e-07, atol 0.0001)
- **Device:** NVIDIA RTX 4000 Ada Generation (`cuda:0`), vendor `nvidia`
- **SYCL target flags:** `-fsycl-targets=nvidia_gpu_sm_89`

| step | status | summary |
|---|---|---|
| reference_build | `ok` | exit 0: g++ -O2 -std=c++17 /tmp/tmp.zSg9iOmwhl/task/trimat_forward_ref.cpp -o build/reference |
| native_build | `ok` | exit 0: /usr/local/cuda/bin/nvcc -O2 -std=c++17 -arch=native -I/tmp/tmp.zSg9iOmwhl/task /tmp/tmp.zSg9iOmwhl/task/trimat_forward_harness.cu -lcublas -o build/native |
| native_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 5.96e-07, atol 0.0001) |
| sycl_compile | `ok` | compiled 1 source(s) -> /tmp/tmp.zSg9iOmwhl/out/work/build/sycl (0 warning(s)) |
| sycl_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 0, atol 0.0001) |
| sycl_verify_vs_native | `ok` | PASS: 1/1 case(s) match (worst max abs err 5.96e-07, atol 0.0001) |
| sycl_benchmark | `ok` | 10 run(s) each; cuda:0: median 29.39 ms, p95 30.29 ms |
| native_benchmark | `ok` | 10 run(s) each; none: median 541.01 ms, p95 558.87 ms |

Limitations:
- Benchmarks are single-process wall times plus program-reported kernel times; treat them as rough.
