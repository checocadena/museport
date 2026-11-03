# GPU test: llmc_attention_backward

- **Result:** `ok`: PASS on NVIDIA RTX 2000 Ada Generation (cuda:0, nvidia): sycl_verify_vs_reference: PASS: 1/1 case(s) match (worst max abs err 4.77e-07, atol 0.001); sycl_verify_vs_native: PASS: 1/1 case(s) match (worst max abs err 4.77e-07, atol 0.001)
- **Device:** NVIDIA RTX 2000 Ada Generation (`cuda:0`), vendor `nvidia`
- **SYCL target flags:** `-fsycl-targets=nvidia_gpu_sm_89`

| step | status | summary |
|---|---|---|
| reference_build | `ok` | exit 0: g++ -O2 -std=c++17 /tmp/tmp.142mqNNZFT/task/attention_backward_ref.cpp -o build/reference |
| native_build | `ok` | exit 0: /usr/local/cuda/bin/nvcc -O2 -std=c++17 -arch=native -I/tmp/tmp.142mqNNZFT/task /tmp/tmp.142mqNNZFT/task/attention_backward_harness.cu -lcublas -o build/native |
| native_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 7.15e-07, atol 0.001) |
| sycl_compile | `ok` | compiled 1 source(s) -> /tmp/tmp.142mqNNZFT/out/work/build/sycl (18 warning(s)) |
| sycl_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 4.77e-07, atol 0.001) |
| sycl_verify_vs_native | `ok` | PASS: 1/1 case(s) match (worst max abs err 4.77e-07, atol 0.001) |
| sycl_benchmark | `ok` | 10 run(s) each; cuda:0: median 769.29 ms, p95 829.68 ms |
| native_benchmark | `ok` | 10 run(s) each; none: median 901.78 ms, p95 1017.75 ms |

Limitations:
- Benchmarks are single-process wall times plus program-reported kernel times; treat them as rough.
