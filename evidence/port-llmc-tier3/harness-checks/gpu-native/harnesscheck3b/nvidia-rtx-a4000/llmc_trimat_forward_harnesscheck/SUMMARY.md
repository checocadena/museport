# GPU test: llmc_trimat_forward_harnesscheck

- **Result:** `fail`: FAIL on NVIDIA RTX A4000 (cuda:0, nvidia): sycl_verify_vs_reference: PASS: 1/1 case(s) match (worst max abs err 0, atol 0.0001); sycl_verify_vs_native: FAIL: 0/1 case(s) match; first failure: 49484 of 1966080 element(s) outside tolerance
- **Device:** NVIDIA RTX A4000 (`cuda:0`), vendor `nvidia`
- **SYCL target flags:** `-fsycl-targets=nvidia_gpu_sm_86`

| step | status | summary |
|---|---|---|
| reference_build | `ok` | exit 0: g++ -O2 -std=c++17 /tmp/tmp.Ra8IKpon7h/task/trimat_forward_ref.cpp -o build/reference |
| native_build | `ok` | exit 0: /usr/local/cuda/bin/nvcc -O2 -std=c++17 -arch=native -I/tmp/tmp.Ra8IKpon7h/task /tmp/tmp.Ra8IKpon7h/task/trimat_forward_harness.cu -lcublas -o build/native |
| native_verify_vs_reference | `fail` | FAIL: 0/1 case(s) match; first failure: 49484 of 1966080 element(s) outside tolerance |
| sycl_compile | `ok` | compiled 1 source(s) -> /tmp/tmp.Ra8IKpon7h/out/work/build/sycl (0 warning(s)) |
| sycl_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 0, atol 0.0001) |
| sycl_verify_vs_native | `fail` | FAIL: 0/1 case(s) match; first failure: 49484 of 1966080 element(s) outside tolerance |
| sycl_benchmark | `ok` | 10 run(s) each; cuda:0: median 16.78 ms, p95 21.22 ms |
| native_benchmark | `ok` | 10 run(s) each; none: median 537.04 ms, p95 747.88 ms |

Limitations:
- Benchmarks are single-process wall times plus program-reported kernel times; treat them as rough.
