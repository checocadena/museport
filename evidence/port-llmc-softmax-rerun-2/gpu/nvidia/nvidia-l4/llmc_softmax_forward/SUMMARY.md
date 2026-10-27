# GPU test: llmc_softmax_forward

- **Result:** `ok`: PASS on NVIDIA L4 (cuda:0, nvidia): sycl_verify_vs_reference: PASS: 1/1 case(s) match (worst max abs err 2.86e-05, atol 0.0001); sycl_verify_vs_native: PASS: 1/1 case(s) match (worst max abs err 5.96e-08, atol 0.0001)
- **Device:** NVIDIA L4 (`cuda:0`), vendor `nvidia`
- **SYCL target flags:** `-fsycl-targets=nvidia_gpu_sm_89`

| step | status | summary |
|---|---|---|
| reference_build | `ok` | exit 0: g++ -O2 -std=c++17 /tmp/tmp.ShsoHcpjQI/task/softmax_forward_ref.cpp -o build/reference |
| native_build | `ok` | exit 0: /usr/local/cuda/bin/nvcc -O2 -std=c++17 -arch=native -I/tmp/tmp.ShsoHcpjQI/task /tmp/tmp.ShsoHcpjQI/task/softmax_forward_harness.cu -o build/native |
| native_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 2.86e-05, atol 0.0001) |
| sycl_compile | `ok` | compiled 1 source(s) -> /tmp/tmp.ShsoHcpjQI/out/work/build/sycl (0 warning(s)) |
| sycl_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 2.86e-05, atol 0.0001) |
| sycl_verify_vs_native | `ok` | PASS: 1/1 case(s) match (worst max abs err 5.96e-08, atol 0.0001) |
| sycl_benchmark | `ok` | 10 run(s) each; cuda:0: median 349.89 ms, p95 415.15 ms |
| native_benchmark | `ok` | 10 run(s) each; none: median 343.33 ms, p95 380.65 ms |

Limitations:
- Benchmarks are single-process wall times plus program-reported kernel times; treat them as rough.
