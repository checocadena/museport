# L1 speed: Muse's bf16 trainer against llm.c's CUDA trainer

llm.c's trainer command line, `-b 4 -t 1024 -x 30 -v 1000 -m 2 -s 0 -n 0 -h 0`, runs 30 training
steps on tiny-shakespeare from GPT-2 124M. Time per step is the median of llm.c's own step timer
over steps 6-30 (`museport.speed`, `run_speed.py`), so model loading and warm-up are not
counted. The losses of the 30 steps are also compared, with a tolerance of 0.1.

**The CUDA reference:** llm.c's trainer built with `nvcc` on an NVIDIA L4. Two runs gave
identical losses: 141 ms per step ([cuda-record](cuda-record)), for $0.01.

## Run 3's port (Muse's final files, unchanged)

| Device | ms per step | Losses (worst difference over 30 steps) |
|---|---|---|
| NVIDIA RTX 2000 Ada, llm.c's CUDA trainer (native, same machine) | **183** | 0.0131 vs the L4 recording |
| NVIDIA RTX 2000 Ada, Muse's SYCL port (oneMath patched) | **523** (2.9× CUDA's time) | 0.0153 vs llm.c's CUDA live, 0.0156 vs the recording |
| AMD Instinct MI350X, Muse's SYCL port | 867 | 0.0162 vs the recording |

- **Correct in real training:** at full size (T=1024) on both GPUs, the port's losses follow
  llm.c's CUDA within 0.016 for 30 steps.
- **Not tuned:** on NVIDIA it takes 2.9 times as long per step as llm.c's CUDA. On the MI350X,
  a much larger GPU, it is slower still, so the port is far from what the hardware can do.
  - **A visible cause:** oneMath has no bf16 batched gemm, so the attention matmuls loop one
    gemm per (batch, head): 48 calls where llm.c makes one batched cuBLAS call.
  - **Not done:** the kernels have not been profiled.
- **Cost:** NVIDIA $0.015, AMD $0.26.
