# GPU test: llmc_classifier_fused

- **Result:** `ok`: PASS on NVIDIA GeForce RTX 4090 (cuda:0, nvidia): sycl_verify_vs_reference: PASS: 1/1 case(s) match (worst max abs err 3.81e-06, atol 0.0001); sycl_verify_vs_native: PASS: 1/1 case(s) match (worst max abs err 1.14e-05, atol 0.0001)
- **Device:** NVIDIA GeForce RTX 4090 (`cuda:0`), vendor `nvidia`
- **SYCL target flags:** `-fsycl-targets=nvidia_gpu_sm_89`

| step | status | summary |
|---|---|---|
| reference_build | `ok` | exit 0: g++ -O2 -std=c++17 /tmp/tmp.1uGolSK0PL/task/classifier_fused_ref.cpp -o build/reference |
| native_build | `ok` | exit 0: /usr/local/cuda/bin/nvcc -O2 -std=c++17 -arch=native -I/tmp/tmp.1uGolSK0PL/task /tmp/tmp.1uGolSK0PL/task/classifier_fused_harness.cu -o build/native |
| native_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 1.34e-05, atol 0.0001) |
| sycl_compile | `ok` | compiled 1 source(s) -> /tmp/tmp.1uGolSK0PL/out/work/build/sycl (0 warning(s)) |
| sycl_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 3.81e-06, atol 0.0001) |
| sycl_verify_vs_native | `ok` | PASS: 1/1 case(s) match (worst max abs err 1.14e-05, atol 0.0001) |
| sycl_benchmark | `ok` | 10 run(s) each; cuda:0: median 416.63 ms, p95 420.70 ms |
| native_benchmark | `ok` | 10 run(s) each; none: median 379.08 ms, p95 388.89 ms |

Limitations:
- Benchmarks are single-process wall times plus program-reported kernel times; treat them as rough.
