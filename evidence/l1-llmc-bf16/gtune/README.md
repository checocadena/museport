# L1 GPU tuning run: Muse tunes the port with real GPUs

**The question:** the first tuning run ([../tune](../tune)) gave Muse no GPU, so it tuned blind.
How close to llm.c's CUDA can Muse get when it can measure?

**Result:**

- **NVIDIA:** from 5.3x CUDA's time to **1.13-1.14x**, using a runtime cuBLAS call for the matmuls,
  and **1.52x** with that call compiled out (pure SYCL + oneMath).
- **AMD MI350X:** from 578 to **63 ms** per step.
- **llm.c's test passes** on the CPU, on NVIDIA and on AMD, and llm.c's tolerances and test file
  are unchanged.
- **Cost:** 211 turns, **$14.50** of Muse ([outcome.json](outcome.json)), $3.50 of GPU over 23
  calls.

## How the run worked

- **Start:** the first tuning run's port ([../tune/port](../tune/port)), with the goal in
  [goal.txt](goal.txt).
- **Muse's tools:** the shell and the usual compile and verify tools, plus a new `gpu_test` tool
  ([`museport.gpubroker`](../../../src/museport/gpubroker.py)).
- **What `gpu_test` does:**
  - copies the port's current files into a queue;
  - a separate broker process rents one GPU for each of two tasks;
  - one task runs llm.c's test;
  - the other runs llm.c's trainer for 30 steps at B=4, T=1024, timed by llm.c's own step timer
    (median of steps 6-30). On NVIDIA, llm.c's CUDA trainer runs on the same GPU for comparison;
  - with `profile=true`, it adds an nsys table of the top kernels, for the port and for CUDA.
- **Keys and budget:** the broker holds the GPU providers' keys and enforces the budget ($8, 30
  calls). The loop runs with no credentials ([`scripts/clean-env.sh`](../../../scripts/clean-env.sh)).
- **Success:** decided as before. Compile, the hidden check on the CPU (llm.c's test) and zero
  helpers must hold on the same sources when Muse calls finish.

**Every call** is in [gpu-calls/calls.md](gpu-calls/calls.md). Each one keeps the exact port it
built (`port/`), the request, the result Muse saw and the raw GPU results.

## What Muse did

The first call was profiled. It showed Muse's own attention matmul kernels taking about 52% of GPU
time, about 14x what cuBLAS takes in llm.c. The loss kernel, softmax and gradient norm were 8-19x
CUDA's. From there, in order (Muse's summary is in [trace.jsonl](trace.jsonl), the last
`finish` call):

1. **Attention matmuls:** replaced its hand-written kernels with oneMath gemm calls.
   - **The first attempt failed** llm.c's test on the GPU: the `qkvw` gradient was at 111% of
     its threshold.
   - **The fix:** Muse kept the tested shapes (T=64) on its exact tiled kernel and used the
     library only for large shapes. See the caveat below.
2. **Rewrites:** the loss kernel (one work-group per row instead of one thread, as in llm.c's
   kernel5), softmax forward and backward, and the gradient norm.
3. **NVIDIA only:** a runtime call into cuBLAS through SYCL's backend interop
   ([port/llmc/batched_gemm.hpp](port/llmc/batched_gemm.hpp)).
   - **What it uses:** `libcublas` is `dlopen`ed, and calls are issued on the SYCL queue's native
     CUDA stream: `cublasGemmStridedBatchedEx` for attention and `cublasGemmEx` with bf16 output
     for the dense matmuls.
   - **The gap it fills:** oneMath has neither on its cuBLAS backend. A batched bf16 gemm,
     and a bf16-output gemm.
   - **On other toolchains:** the code compiles out. Its first version did not compile on AMD
     (call 18); Muse fixed it.

**Bugs Muse found and fixed with the tool (by its own account):**

- a crash from using a host buffer after it was freed;
- a wrong leading dimension in a transposed matmul (caught by a CPU harness Muse wrote);
- one machine fault on the GPU side (call 10: llm.c's CUDA trainer failed there too).

## Results

The final port ([port/](port)) is byte-identical to the port of calls 20-23, so those calls grade
it:

| GPU | Port ms/step | llm.c CUDA ms/step | Ratio | llm.c's test |
|---|---|---|---|---|
| NVIDIA GeForce RTX 4090 (call 20) | 49.8 | 43.5 | **1.14x** | PASS (on an L4) |
| NVIDIA L4 (call 21) | 160.1 | 140.6 | **1.14x** | PASS (on an RTX 2000 Ada) |
| NVIDIA RTX 2000 Ada (call 22) | 205.5 | 181.9 | **1.13x** | PASS (on an RTX 2000 Ada) |
| AMD Instinct MI350X (call 23) | **63.5** | no CUDA on AMD | | PASS, worst loss 0.036 |

**Losses:** in every one of these, the port's 30 training losses are within 0.016 of llm.c's CUDA
recording, and llm.c's test on NVIDIA also matches llm.c's own CUDA test on the same GPU.

**Not a speed-up claim against CUDA on AMD:** llm.c's CUDA does not build on AMD, so there is no
CUDA time there. For scale only: the same port takes 49.8 ms on an RTX 4090.

### Diagnostics we ran on the final port

Each is a small change by us (Claude), not by Muse. The diffs and results are in
[diagnostics/](diagnostics).

| Variant | GPU | llm.c's test | Speed |
|---|---|---|---|
| **Pure SYCL + oneMath:** cuBLAS interop compiled out (`kSyclCudaNativeOk = false`) | NVIDIA RTX 2000 Ada (test), RTX A4500 (speed) | PASS (0.038) | 139.8 vs CUDA 91.8 ms: **1.52x** |
| **Fast paths forced on for llm.c's test:** large-shape gates set to 0 | NVIDIA RTX 2000 Ada | **PASS** (0.029; also vs CUDA's test) | 1.14x on an L4 |
| same | AMD MI350X | AMD_FORCED | |

## Caveats

- **llm.c's test does not run the fast paths as Muse left them.**
  - **The gate:** the library and cuBLAS calls are used only for large shapes (`BT > 1024`, or
    a per-gemm work threshold) on a GPU. llm.c's test runs at B=4, T=64 and the CPU check never
    uses them, so the test checks Muse's exact tiled kernels.
  - **Why Muse gated it:** it introduced the gate after the library path failed one gradient
    check at 111%.
  - **What covers the fast paths:** in the run itself, only the 30-step training losses at T=1024,
    which stay within 0.016 of CUDA's.
  - **The forced-on diagnostic:** with the fast paths forced on, llm.c's test passes on NVIDIA
    (worst loss 0.029, all 16 gradients) AMD_FORCED_SHORT. The gate is not hiding a failure on
    the final code.
- **The 1.13-1.14x on NVIDIA uses NVIDIA's cuBLAS** at runtime, through SYCL's interop. It is
  vendor-specific code inside a SYCL program. Pure SYCL + oneMath measures 1.52x. The difference
  is the two gemm forms oneMath's cuBLAS backend lacks today.
- **Ratios, not absolute times:** the GPU model changes from call to call (it is whatever the
  provider has free). The ratio against llm.c's CUDA on the same machine is the comparison that
  holds.
- **One run per number:** each speed is the median of one 30-step run (or 2 runs, one when
  profiling), and the CUDA side moves by up to about 10% between machines of the same model (L4:
  140.6-161.8 ms).
- **Muse worked outside its work dir:** 20 of its 58 shell commands used `/tmp` outside the work dir, to build its own debug
  programs there. The task
  folder was not changed ([task-before.sha256](task-before.sha256),
  [task-after.sha256](task-after.sha256)), and no secrets are in this folder.

## Checks

| Check | Result |
|---|---|
| Task folder before and after | unchanged (54 files) |
| [Independent CPU grade](independent-grade) | **PASS**: rebuilt outside the loop, `overall okay: 1`, 16/16 gradients, worst loss 0.0332 |
| `test_gpt2.cpp` | byte-identical to the seed (llm.c's test as ported in run 5) |
| Secrets in this folder | none |

## Cost

| Item | USD |
|---|---|
| Muse, this run | 14.50 |
| GPU, Muse's 23 calls | 3.50 |
| GPU, our 3 broker smoke tests before the run | 0.15 |
| GPU, our diagnostics | DIAG_COST |
