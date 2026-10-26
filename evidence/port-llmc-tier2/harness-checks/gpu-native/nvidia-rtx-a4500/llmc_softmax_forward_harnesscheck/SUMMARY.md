# GPU test: llmc_softmax_forward_harnesscheck

- **Result:** `ok`: PASS on NVIDIA RTX A4500 (cuda:0, nvidia): sycl_verify_vs_reference: PASS: 1/1 case(s) match (worst max abs err 0, atol 0.0001); sycl_verify_vs_native: PASS: 1/1 case(s) match (worst max abs err 2.86e-05, atol 0.0001)
- **Device:** NVIDIA RTX A4500 (`cuda:0`), vendor `nvidia`
- **SYCL target flags:** `-fsycl-targets=nvidia_gpu_sm_86`

| step | status | summary |
|---|---|---|
| reference_build | `ok` | exit 0: g++ -O2 -std=c++17 /tmp/tmp.F65q5JGa30/task/softmax_forward_ref.cpp -o build/reference |
| native_build | `ok` | exit 0: /usr/local/cuda/bin/nvcc -O2 -std=c++17 -arch=native -I/tmp/tmp.F65q5JGa30/task /tmp/tmp.F65q5JGa30/task/softmax_forward_harness.cu -o build/native |
| native_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 2.86e-05, atol 0.0001) |
| sycl_compile | `ok` | compiled 1 source(s) -> /tmp/tmp.F65q5JGa30/out/work/build/sycl (0 warning(s)) |
| sycl_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 0, atol 0.0001) |
| sycl_verify_vs_native | `ok` | PASS: 1/1 case(s) match (worst max abs err 2.86e-05, atol 0.0001) |
| sycl_benchmark | `ok` | 10 run(s) each; cuda:0: median 86.97 ms, p95 101.39 ms |
| native_benchmark | `ok` | 10 run(s) each; none: median 443.68 ms, p95 480.60 ms |

Limitations:
- Benchmarks are single-process wall times plus program-reported kernel times; treat them as rough.
