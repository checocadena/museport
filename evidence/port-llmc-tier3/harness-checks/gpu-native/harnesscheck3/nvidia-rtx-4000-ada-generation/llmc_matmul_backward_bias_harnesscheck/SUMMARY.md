# GPU test: llmc_matmul_backward_bias_harnesscheck

- **Result:** `ok`: PASS on NVIDIA RTX 4000 Ada Generation (cuda:0, nvidia): sycl_verify_vs_reference: PASS: 1/1 case(s) match (worst max abs err 0, atol 0.005); sycl_verify_vs_native: PASS: 1/1 case(s) match (worst max abs err 1.72e-05, atol 0.005)
- **Device:** NVIDIA RTX 4000 Ada Generation (`cuda:0`), vendor `nvidia`
- **SYCL target flags:** `-fsycl-targets=nvidia_gpu_sm_89`

| step | status | summary |
|---|---|---|
| reference_build | `ok` | exit 0: g++ -O2 -std=c++17 /tmp/tmp.SWOvTRk2dd/task/matmul_backward_bias_ref.cpp -o build/reference |
| native_build | `ok` | exit 0: /usr/local/cuda/bin/nvcc -O2 -std=c++17 -arch=native -I/tmp/tmp.SWOvTRk2dd/task /tmp/tmp.SWOvTRk2dd/task/matmul_backward_bias_harness.cu -o build/native |
| native_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 1.91e-05, atol 0.005) |
| sycl_compile | `ok` | compiled 1 source(s) -> /tmp/tmp.SWOvTRk2dd/out/work/build/sycl (0 warning(s)) |
| sycl_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 0, atol 0.005) |
| sycl_verify_vs_native | `ok` | PASS: 1/1 case(s) match (worst max abs err 1.72e-05, atol 0.005) |
| sycl_benchmark | `ok` | 10 run(s) each; cuda:0: median 22.33 ms, p95 22.86 ms |
| native_benchmark | `ok` | 10 run(s) each; none: median 364.09 ms, p95 395.22 ms |

Limitations:
- Benchmarks are single-process wall times plus program-reported kernel times; treat them as rough.
