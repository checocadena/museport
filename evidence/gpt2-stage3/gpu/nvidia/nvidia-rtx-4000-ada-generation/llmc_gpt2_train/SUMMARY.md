# GPU test: llmc_gpt2_train

- **Result:** `ok`: PASS on NVIDIA RTX 4000 Ada Generation (cuda:0, nvidia): sycl_verify_vs_reference: PASS: 1/1 case(s) match (worst max abs err 0.000135, atol 0.01); sycl_verify_vs_native: PASS: 1/1 case(s) match (worst max abs err 5.9e-06, atol 0.01)
- **Device:** NVIDIA RTX 4000 Ada Generation (`cuda:0`), vendor `nvidia`
- **SYCL target flags:** `-fsycl-targets=nvidia_gpu_sm_89`

| step | status | summary |
|---|---|---|
| reference_build | `ok` | exit 0: bash -c 'mkdir -p build && ln -sf /tmp/tmp.cTJZTHqvF8/task/gpt2_124M.bin build/gpt2_124M.bin && g++ -O2 -std=c++17 /tmp/tmp.cTJZTHqvF8/task/gpt2_ref.cpp -o build/reference' |
| native_build | `ok` | exit 0: /usr/local/cuda/bin/nvcc -O2 -std=c++17 -arch=native -I/tmp/tmp.cTJZTHqvF8/task /tmp/tmp.cTJZTHqvF8/task/gpt2_train_harness.cu -lcublas -o build/native |
| native_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 0.000128, atol 0.01) |
| sycl_compile | `ok` | compiled 1 source(s) -> /tmp/tmp.cTJZTHqvF8/out/work/build/sycl (0 warning(s)) |
| sycl_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 0.000135, atol 0.01) |
| sycl_verify_vs_native | `ok` | PASS: 1/1 case(s) match (worst max abs err 5.9e-06, atol 0.01) |
| sycl_benchmark | `ok` | 10 run(s) each; cuda:0: median 1310.10 ms, p95 1390.35 ms |
| native_benchmark | `ok` | 10 run(s) each; none: median 1299.63 ms, p95 1396.57 ms |

Limitations:
- Benchmarks are single-process wall times plus program-reported kernel times; treat them as rough.
