# GPU test: llmc_matmul_backward_bias

- **Result:** `ok`: PASS on NVIDIA RTX 2000 Ada Generation (cuda:0, nvidia): sycl_verify_vs_reference: PASS: 1/1 case(s) match (worst max abs err 5.72e-06, atol 0.005); sycl_verify_vs_native: PASS: 1/1 case(s) match (worst max abs err 1.14e-05, atol 0.005)
- **Device:** NVIDIA RTX 2000 Ada Generation (`cuda:0`), vendor `nvidia`
- **SYCL target flags:** `-fsycl-targets=nvidia_gpu_sm_89`

| step | status | summary |
|---|---|---|
| reference_build | `ok` | exit 0: g++ -O2 -std=c++17 /tmp/tmp.DTR7FcXqfS/task/matmul_backward_bias_ref.cpp -o build/reference |
| native_build | `ok` | exit 0: /usr/local/cuda/bin/nvcc -O2 -std=c++17 -arch=native -I/tmp/tmp.DTR7FcXqfS/task /tmp/tmp.DTR7FcXqfS/task/matmul_backward_bias_harness.cu -o build/native |
| native_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 1.34e-05, atol 0.005) |
| sycl_compile | `ok` | compiled 1 source(s) -> /tmp/tmp.DTR7FcXqfS/out/work/build/sycl (0 warning(s)) |
| sycl_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 5.72e-06, atol 0.005) |
| sycl_verify_vs_native | `ok` | PASS: 1/1 case(s) match (worst max abs err 1.14e-05, atol 0.005) |
| sycl_benchmark | `ok` | 10 run(s) each; cuda:0: median 277.79 ms, p95 299.71 ms |
| native_benchmark | `ok` | 10 run(s) each; none: median 275.95 ms, p95 300.79 ms |

Limitations:
- Benchmarks are single-process wall times plus program-reported kernel times; treat them as rough.
