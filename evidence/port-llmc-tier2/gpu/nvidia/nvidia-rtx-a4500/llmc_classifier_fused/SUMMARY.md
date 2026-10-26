# GPU test: llmc_classifier_fused

- **Result:** `ok`: PASS on NVIDIA RTX A4500 (cuda:0, nvidia): sycl_verify_vs_reference: PASS: 1/1 case(s) match (worst max abs err 1.34e-05, atol 0.0001); sycl_verify_vs_native: PASS: 1/1 case(s) match (worst max abs err 3.81e-06, atol 0.0001)
- **Device:** NVIDIA RTX A4500 (`cuda:0`), vendor `nvidia`
- **SYCL target flags:** `-fsycl-targets=nvidia_gpu_sm_86`

| step | status | summary |
|---|---|---|
| reference_build | `ok` | exit 0: g++ -O2 -std=c++17 /tmp/tmp.oECKP30I6M/task/classifier_fused_ref.cpp -o build/reference |
| native_build | `ok` | exit 0: /usr/local/cuda/bin/nvcc -O2 -std=c++17 -arch=native -I/tmp/tmp.oECKP30I6M/task /tmp/tmp.oECKP30I6M/task/classifier_fused_harness.cu -o build/native |
| native_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 1.34e-05, atol 0.0001) |
| sycl_compile | `ok` | compiled 1 source(s) -> /tmp/tmp.oECKP30I6M/out/work/build/sycl (24 warning(s)) |
| sycl_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 1.34e-05, atol 0.0001) |
| sycl_verify_vs_native | `ok` | PASS: 1/1 case(s) match (worst max abs err 3.81e-06, atol 0.0001) |
| sycl_benchmark | `ok` | 10 run(s) each; cuda:0: median 410.81 ms, p95 440.81 ms |
| native_benchmark | `ok` | 10 run(s) each; none: median 356.93 ms, p95 434.49 ms |

Limitations:
- Benchmarks are single-process wall times plus program-reported kernel times; treat them as rough.
