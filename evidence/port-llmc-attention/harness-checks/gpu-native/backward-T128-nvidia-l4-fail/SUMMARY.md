# GPU test: llmc_attention_backward_harnesscheck

- **Result:** `fail`: FAIL on NVIDIA L4 (cuda:0, nvidia): sycl_verify_vs_reference: PASS: 1/1 case(s) match (worst max abs err 0, atol 0.001); sycl_verify_vs_native: FAIL: 0/1 case(s) match; first failure: 266609 of 4718592 element(s) outside tolerance
- **Device:** NVIDIA L4 (`cuda:0`), vendor `nvidia`
- **SYCL target flags:** `-fsycl-targets=nvidia_gpu_sm_89`

| step | status | summary |
|---|---|---|
| reference_build | `ok` | exit 0: g++ -O2 -std=c++17 /tmp/tmp.fZX37In9a1/task/attention_backward_ref.cpp -o build/reference |
| native_build | `ok` | exit 0: /usr/local/cuda/bin/nvcc -O2 -std=c++17 -arch=native -I/tmp/tmp.fZX37In9a1/task /tmp/tmp.fZX37In9a1/task/attention_backward_harness.cu -lcublas -o build/native |
| native_verify_vs_reference | `fail` | FAIL: 0/1 case(s) match; first failure: 267029 of 4718592 element(s) outside tolerance |
| sycl_compile | `ok` | compiled 1 source(s) -> /tmp/tmp.fZX37In9a1/out/work/build/sycl (0 warning(s)) |
| sycl_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 0, atol 0.001) |
| sycl_verify_vs_native | `fail` | FAIL: 0/1 case(s) match; first failure: 266609 of 4718592 element(s) outside tolerance |
| sycl_benchmark | `ok` | 10 run(s) each; cuda:0: median 122.51 ms, p95 124.81 ms |
| native_benchmark | `ok` | 10 run(s) each; none: median 418.01 ms, p95 563.45 ms |

Limitations:
- Benchmarks are single-process wall times plus program-reported kernel times; treat them as rough.
