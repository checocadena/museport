# GPU test: llmc_attention_forward

- **Result:** `ok`: PASS on NVIDIA L4 (cuda:0, nvidia): sycl_verify_vs_reference: PASS: 1/1 case(s) match (worst max abs err 2.98e-07, atol 0.001); sycl_verify_vs_native: PASS: 1/1 case(s) match (worst max abs err 2.98e-07, atol 0.001)
- **Device:** NVIDIA L4 (`cuda:0`), vendor `nvidia`
- **SYCL target flags:** `-fsycl-targets=nvidia_gpu_sm_89`

| step | status | summary |
|---|---|---|
| reference_build | `ok` | exit 0: g++ -O2 -std=c++17 /tmp/tmp.UJtJKN26Pc/task/attention_forward_ref.cpp -o build/reference |
| native_build | `ok` | exit 0: /usr/local/cuda/bin/nvcc -O2 -std=c++17 -arch=native -I/tmp/tmp.UJtJKN26Pc/task /tmp/tmp.UJtJKN26Pc/task/attention_forward_harness.cu -lcublas -o build/native |
| native_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 2.38e-07, atol 0.001) |
| sycl_compile | `ok` | compiled 1 source(s) -> /tmp/tmp.UJtJKN26Pc/out/work/build/sycl (6 warning(s)) |
| sycl_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 2.98e-07, atol 0.001) |
| sycl_verify_vs_native | `ok` | PASS: 1/1 case(s) match (worst max abs err 2.98e-07, atol 0.001) |
| sycl_benchmark | `ok` | 10 run(s) each; cuda:0: median 564.51 ms, p95 578.28 ms |
| native_benchmark | `ok` | 10 run(s) each; none: median 532.70 ms, p95 543.53 ms |

Limitations:
- Benchmarks are single-process wall times plus program-reported kernel times; treat them as rough.
