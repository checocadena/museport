# GPU test: llama2_story_record

- **Result:** `ok`: recorded 2 output(s) of the original CUDA on NVIDIA RTX 2000 Ada Generation (nvidia); two runs identical: PASS: 2/2 case(s) match (worst max abs err 0, atol 0)
- **Device:** NVIDIA RTX 2000 Ada Generation (`cuda:0`), vendor `nvidia`
- **SYCL target flags:** `(JIT)`

| step | status | summary |
|---|---|---|
| reference_build | `ok` | exit 0: true |
| native_build | `ok` | exit 0: /usr/local/cuda/bin/nvcc -O2 -std=c++17 -arch=native -I/tmp/tmp.rJcqeXc4tn/task /tmp/tmp.rJcqeXc4tn/task/llama2.cu -o build/native |
| native_repeat | `ok` | PASS: 2/2 case(s) match (worst max abs err 0, atol 0) |

Limitations:
- A recorded reference is the original program's own output on one GPU: it is only as right as that program, and a non-deterministic program needs a tolerance at least as wide as its run-to-run difference.
