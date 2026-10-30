# L1 pure-oneAPI tuning run: SYCL and oneMath only, with the test running training's code

**The question:** the GPU-tuned port ([../gtune](../gtune)) reached 1.13-1.21x llm.c's CUDA time
on NVIDIA. It got there two ways that weaken the claim:

- it called NVIDIA's cuBLAS directly through SYCL interop;
- it sent llm.c's small test shapes to different kernels than training's.

Can Muse do without both?

**Setup:** Muse started from that port with [this goal](goal.txt). It had the same tools, and the
GPU broker ran with `--pure-oneapi`: any port containing `dlopen`, `get_native`, `cublas`,
`rocblas` and the like was refused before a GPU was rented. Caps: Muse $15, GPU $5.

## Result

**The run itself did not succeed.** It reached its 300-turn limit at $14.72
([outcome.json](outcome.json)). The final source's CPU check timed out after 600 s.

**One version inside the run is the result.** Muse called `gpu_test` once, at turn 209, on a
version with no vendor interop and no size thresholds, so llm.c's test runs the same code paths
as training. That version, in [gpu-calls/](gpu-calls) with the exact files it built:

| Check | Result |
|---|---|
| llm.c's test, CPU, rebuilt outside the loop, no time limit ([grade](independent-grade-turn209)) | **PASS**: `overall okay: 1`, logits OK, 16/16 gradients, losses within 0.05 of PyTorch's |
| llm.c's test, NVIDIA L4 (Muse's `gpu_test`) | **PASS**, worst loss 0.0333; also against llm.c's CUDA test on the same GPU (0.0324) |
| llm.c's test, AMD Instinct MI300X (ours, [turn209-gpu/test-amd](turn209-gpu/test-amd)) | **PASS**, worst loss 0.0325 |
| Speed, NVIDIA RTX A4000 (`gpu_test`, 2 runs) | 181.1 ms per step vs llm.c's CUDA 116.7: **1.55x** |
| Speed, AMD MI300X (ours, 3 runs, [turn209-gpu/speed-amd](turn209-gpu/speed-amd)) | 190.8 ms per step, 30 losses within 0.015 of CUDA's recording |
| Vendor interop or size thresholds in the source | none |

**Timed again on Oct 30** ([speed/](speed), 3 runs each, the same files):

| GPU | Port ms/step | llm.c CUDA ms/step | Ratio | Losses vs CUDA's recording |
|---|---|---|---|---|
| NVIDIA L4 | 199.9 | 142.4 | **1.40x** | within 0.023 |
| NVIDIA RTX A4000 | 181.6 | 117.7 | **1.54x** | within 0.014 |
| AMD Instinct MI350X | **63.3** | no CUDA on AMD | | within 0.016 |

So the portable version runs at **1.40-1.55x** llm.c's CUDA time on NVIDIA. On the MI350X it
matches the GPU-tuned port's 63.5 ms, which on AMD also ran without vendor code.

**The final version** ([port/](port)) differs from turn 209's in `train_gpt2.cpp` and five
headers: Muse kept changing rounding to match the CPU check. Graded outside the loop
([grade](independent-grade-final)), it fails llm.c's logit check, with `overall okay: 0`.

## What it means

- **Portable, and the test checks what trains.** SYCL + oneMath alone reaches 1.40-1.55x CUDA's
  step time. The GPU-tuned port's 1.36-1.52x without interop used size thresholds; this one
  has none, and llm.c's test passes on a CPU, an NVIDIA GPU and an AMD GPU.
- **The remaining gap is the one in the oneMath feature request.** The profile shows cuBLAS
  gemms called 13,824 times in 10 steps: Muse's per-(batch, head) loop of oneMath calls, which
  a bf16 `gemm_batch` would replace with one call per layer
  ([issue draft](../../../upstream/onemath/ISSUE-bf16-gemm-forms.md)).
- **Why the loop's check failed:** on the CPU, the training-size paths that the test now runs
  are slow. The test took 105 minutes outside the loop, with two grades sharing the CPU, while
  the check allows 10 minutes.
- **Different GPUs:** the first AMD numbers come from an MI300X (no MI350X was available then) and cannot be
  compared with the earlier 63.5 ms on an MI350X.
- **Runs per number:** Muse's own NVIDIA timing is 2 runs; the Oct 30 timings are 3 runs each.

## Checks and cost

| | |
|---|---|
| Task folder before and after | unchanged ([task-before.sha256](task-before.sha256), [task-after.sha256](task-after.sha256)) |
| Secrets | none |
| Muse | $14.72 |
| GPU | $0.03 (Muse's one call) + $0.35 (our AMD test and speed) + $0.97 (the Oct 30 timings) |
