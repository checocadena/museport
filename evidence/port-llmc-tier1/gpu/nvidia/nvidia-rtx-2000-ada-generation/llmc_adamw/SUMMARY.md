# GPU test: llmc_adamw

- **Result:** `ok`: PASS on NVIDIA RTX 2000 Ada Generation (cuda:0, nvidia): sycl_verify_vs_reference: PASS: 1/1 case(s) match (worst max abs err 5.96e-08, atol 0.0001); sycl_verify_vs_native: PASS: 1/1 case(s) match (worst max abs err 5.96e-08, atol 0.0001)
- **Device:** NVIDIA RTX 2000 Ada Generation (`cuda:0`), vendor `nvidia`
- **SYCL target flags:** `-fsycl-targets=nvidia_gpu_sm_89`

| step | status | summary |
|---|---|---|
| reference_build | `ok` | exit 0: g++ -O2 -std=c++17 /tmp/tmp.94uOYh1RO8/task/adamw_ref.cpp -o build/reference |
| native_build | `ok` | exit 0: /usr/local/cuda/bin/nvcc -O2 -std=c++17 -arch=native -I/tmp/tmp.94uOYh1RO8/task /tmp/tmp.94uOYh1RO8/task/adamw_harness.cu -o build/native |
| native_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 5.96e-08, atol 0.0001) |
| sycl_compile | `ok` | compiled 1 source(s) -> /tmp/tmp.94uOYh1RO8/out/work/build/sycl (0 warning(s)) |
| sycl_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 5.96e-08, atol 0.0001) |
| sycl_verify_vs_native | `ok` | PASS: 1/1 case(s) match (worst max abs err 5.96e-08, atol 0.0001) |
| sycl_benchmark | `ok` | 10 run(s) each; cuda:0: median 302.27 ms, p95 348.90 ms |
| native_benchmark | `ok` | 10 run(s) each; none: median 322.73 ms, p95 356.55 ms |

Limitations:
- Benchmarks are single-process wall times plus program-reported kernel times; treat them as rough.
