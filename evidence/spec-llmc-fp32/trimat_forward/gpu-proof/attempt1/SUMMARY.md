# GPU test: llmc_trimat_forward

- **Result:** `ok`: recorded 1 output(s) of the original CUDA on NVIDIA RTX A4500 (nvidia); two runs identical: PASS: 1/1 case(s) match (worst max abs err 0, atol 0)
- **Device:** NVIDIA RTX A4500 (`cuda:0`), vendor `nvidia`
- **SYCL target flags:** `(JIT)`

| step | status | summary |
|---|---|---|
| reference_build | `ok` | exit 0: true |
| native_build | `ok` | exit 0: /usr/local/cuda/bin/nvcc -O2 -std=c++17 -arch=native -I/tmp/tmp.WROa1ELEXa/task /tmp/tmp.WROa1ELEXa/task/trimat_forward_harness.cu -lcublas -o build/native |
| native_repeat | `ok` | PASS: 1/1 case(s) match (worst max abs err 0, atol 0) |

Limitations:
- A recorded reference is the original program's own output on one GPU: it is only as right as that program, and a non-deterministic program needs a tolerance at least as wide as its run-to-run difference.
