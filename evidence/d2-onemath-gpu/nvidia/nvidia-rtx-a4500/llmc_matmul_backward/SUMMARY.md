# GPU test: llmc_matmul_backward

- **Result:** `fail`: FAIL on NVIDIA RTX A4500 (cuda:0, nvidia): sycl_verify_vs_reference: candidate exited with code -6; sycl_verify_vs_native: candidate exited with code -6
- **Device:** NVIDIA RTX A4500 (`cuda:0`), vendor `nvidia`
- **SYCL target flags:** `-fsycl-targets=nvidia_gpu_sm_86`

| step | status | summary |
|---|---|---|
| reference_build | `ok` | exit 0: g++ -O2 -std=c++17 /tmp/tmp.uO5ucYofHN/task/matmul_backward_ref.cpp -o build/reference |
| native_build | `ok` | exit 0: /usr/local/cuda/bin/nvcc -O2 -std=c++17 -arch=native -I/tmp/tmp.uO5ucYofHN/task /tmp/tmp.uO5ucYofHN/task/matmul_backward_harness.cu -lcublas -o build/native |
| native_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 0.00021, atol 0.001) |
| sycl_compile | `ok` | compiled 1 source(s) -> /tmp/tmp.uO5ucYofHN/out/work/build/sycl (0 warning(s)) |
| sycl_verify_vs_reference | `fail` | candidate exited with code -6 |
| sycl_verify_vs_native | `fail` | candidate exited with code -6 |
| sycl_benchmark | `fail` | command failed on all 1 device(s) |
| native_benchmark | `ok` | 10 run(s) each; none: median 521.63 ms, p95 546.94 ms |

Limitations:
- Benchmarks are single-process wall times plus program-reported kernel times; treat them as rough.
