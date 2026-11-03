# GPU test: llmc_attention_forward_harnesscheck

- **Result:** `ok`: PASS on NVIDIA L4 (cuda:0, nvidia): sycl_verify_vs_reference: PASS: 1/1 case(s) match (worst max abs err 0, atol 0.001); sycl_verify_vs_native: PASS: 1/1 case(s) match (worst max abs err 2.38e-07, atol 0.001)
- **Device:** NVIDIA L4 (`cuda:0`), vendor `nvidia`
- **SYCL target flags:** `-fsycl-targets=nvidia_gpu_sm_89`

| step | status | summary |
|---|---|---|
| reference_build | `ok` | exit 0: g++ -O2 -std=c++17 /tmp/tmp.xHkXZEB2pd/task/attention_forward_ref.cpp -o build/reference |
| native_build | `ok` | exit 0: /usr/local/cuda/bin/nvcc -O2 -std=c++17 -arch=native -I/tmp/tmp.xHkXZEB2pd/task /tmp/tmp.xHkXZEB2pd/task/attention_forward_harness.cu -lcublas -o build/native |
| native_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 2.38e-07, atol 0.001) |
| sycl_compile | `ok` | compiled 1 source(s) -> /tmp/tmp.xHkXZEB2pd/out/work/build/sycl (2 warning(s)) |
| sycl_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 0, atol 0.001) |
| sycl_verify_vs_native | `ok` | PASS: 1/1 case(s) match (worst max abs err 2.38e-07, atol 0.001) |
| sycl_benchmark | `ok` | 10 run(s) each; cuda:0: median 17.49 ms, p95 18.17 ms |
| native_benchmark | `ok` | 10 run(s) each; none: median 409.69 ms, p95 418.95 ms |

Limitations:
- Benchmarks are single-process wall times plus program-reported kernel times; treat them as rough.
