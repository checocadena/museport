# GPU test: llmc_train_gpt2_bf16_speed

- **Result:** `fail`: FAIL on NVIDIA RTX 2000 Ada Generation (cuda:0, nvidia): sycl_verify_vs_reference: candidate exited with code 1; sycl_verify_vs_native: reference exited with code 1
- **Device:** NVIDIA RTX 2000 Ada Generation (`cuda:0`), vendor `nvidia`
- **SYCL target flags:** `-fsycl-targets=nvidia_gpu_sm_89`

| step | status | summary |
|---|---|---|
| reference_build | `ok` | exit 0: true |
| native_build | `ok` | exit 0: /usr/local/cuda/bin/nvcc -O2 -std=c++17 -arch=native -I/tmp/tmp.m6wULwePLA/task /tmp/tmp.m6wULwePLA/task/train_gpt2.cu --use_fast_math -DENABLE_BF16 -lcublas -lcublasLt -L/usr/local/cuda/lib64/stubs -lnvidia-ml -o build/native |
| native_verify_vs_reference | `fail` | candidate exited with code 1 |
| sycl_compile | `ok` | compiled 1 source(s) -> /tmp/tmp.m6wULwePLA/out/work/build/sycl (0 warning(s)) |
| sycl_verify_vs_reference | `fail` | candidate exited with code 1 |
| sycl_verify_vs_native | `error` | reference exited with code 1 |
| sycl_benchmark | `fail` | command failed on all 1 device(s) |
| native_benchmark | `fail` | command failed on all 1 device(s) |

Limitations:
- Benchmarks are single-process wall times plus program-reported kernel times; treat them as rough.
