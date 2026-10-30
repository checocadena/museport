# L1: llm.c's main bf16 trainer, ported and tuned by Muse

llm.c's production trainer at `f1e2ace`: `train_gpt2.cu`, `test_gpt2.cu` and the CUDA headers
in `llmc/`. That is 12 CUDA files, 4,884 lines and 25 kernels, in bf16 mixed precision, with
cuBLASLt matmuls, warp reductions and stochastic rounding. The scope is a single device without
cuDNN, which are llm.c's defaults. Plan: [`docs/L1-PLAN.md`](../../docs/L1-PLAN.md).

## Result

1. **Port, in one run** ([run5](run5)): 125 turns, $5.67, from llm.c's CUDA alone, with a shell.
   A repeat with the same setup ([run6](run6/NOTES.md)) failed within $10, so this is 1 of 2.
   - **On the CPU:** llm.c's own test passes. That covers logits, all 16 gradient tensors, 10
     training losses against PyTorch, and a bit-exact save, reload and replay, with llm.c's own
     bf16 tolerances.
   - **On GPUs:** it also passes on an AMD MI350X (worst loss 0.044) and an NVIDIA RTX 4000 Ada
     (0.030). On an RTX 2000 Ada, one of llm.c's checks failed, and which one was not recorded
     ([run5/gpu](run5/gpu)); the GPU-tuned port passes there.
   - **Speed:** it took 36x CUDA's time per training step (RTX A5000).
2. **Tuning without a GPU** ([tune](tune/README.md)): 192 turns, $10.39. The port moved to device
   memory and its own attention kernels: AMD went from 1274 to 578 ms per step, and NVIDIA to
   5.7x CUDA's time.
3. **Tuning with GPUs** ([gtune](gtune/README.md)): 211 turns, $14.50, plus $3.50 of GPU over 23
   calls. Muse got a `gpu_test` tool that builds its current code on a rented GPU and returns
   llm.c's test result, ms per step next to llm.c's CUDA, and a per-kernel profile.

| Final port, llm.c's trainer, B=4, T=1024 | ms per step | vs llm.c's CUDA, same GPU | llm.c's test |
|---|---|---|---|
| **Portable SYCL + oneMath (no vendor code, test runs training's code), NVIDIA L4** | 199.9 | **1.40x** | PASS |
| **Portable, NVIDIA RTX A4000** | 181.6 | **1.54x** | PASS |
| **Portable, AMD Instinct MI350X** | **63.3** | (llm.c's CUDA does not build on AMD) | PASS |
| *With NVIDIA's cuBLAS through SYCL interop (the GPU tuning run):* | | | |
| NVIDIA RTX 4090 | 49.8 | **1.14x** | PASS |
| NVIDIA L4 | 160.1 | **1.14x** | PASS |
| NVIDIA RTX 2000 Ada | 205.5 | **1.13x** | PASS |
| NVIDIA RTX A4500 (our 3-run repeat) | 113.3 | **1.21x** | PASS |
| NVIDIA RTX 2000 Ada, pure SYCL + oneMath (cuBLAS interop compiled out, our diagnostic) | 250.6 | **1.36x** | PASS |
| NVIDIA RTX A4500, pure SYCL + oneMath (our diagnostic) | 139.8 | **1.52x** | PASS |
| AMD Instinct MI350X | **63.5** | no CUDA on AMD | PASS |

**Read with these caveats** (details in [gtune/README.md](gtune/README.md#caveats)):

- **The NVIDIA 1.13-1.21x calls NVIDIA's cuBLAS at runtime** through SYCL's backend interop, for
  two bf16 gemm forms oneMath's API does not have on any backend: batched bf16, and bf16 output
  (it has both for fp16). Without them, the same code is 1.36x (RTX 2000 Ada) to 1.52x (RTX A4500).
- **llm.c's test does not run the fast paths as Muse left them.** Muse gates them to large
  shapes (B·T > 1024) on a GPU. Forced on, llm.c's test still passes: on NVIDIA (0.029) and on
  AMD (0.046, against a tolerance of 0.05).
- **Rented GPUs vary** from call to call, so the ratio to CUDA on the same machine is the
  comparison. Each number comes from one run.

**Muse cost, L1 in all:** $75.50 (of the $80 cap): plan $0.18, runs 1-3 $14.35, run 4 $5.57, run 5
$5.67, run 6 $10.13, tuning $10.39, $14.50 and $14.72.

| Step | Result | Cost |
|---|---|---|
| `museport plan` (Muse) | 12 items; the grounding check caught one kernel name Muse made up ([plan](plan/plan.md)) | $0.18 |
| Prove the check (NVIDIA RTX 2000 Ada) | llm.c's CUDA test passes; a planted GELU bug fails ([check-proof.json](check-proof.json)) | $0.008 GPU |
| [Run 1](run1) | stopped by the toolkit's 2M-token limit after 34 turns of reading; SYCLomatic could not parse the trainer | $0.70 |
| [Run 2](run2) | 200 turns: the whole port written and compiling; forward pass right; training NaN | $11.79 |
| [Run 3](run3), continuing run 2's files | success in 65 turns, worst loss 0.0326 from PyTorch's | $1.86 |
| [Run 4](run4), from scratch, with a shell | stopped by the loop's stuck rule at turn 132, mid-debugging: three different shell commands failed in a row. Failed shell commands no longer count | $5.57 |
| [Run 5](run5), from scratch, with a shell | **success in one run**, 125 turns; rebuilt outside the loop: PASS (16/16 gradients) | $5.67 |
| [Tuning, CPU only](tune/README.md) | success, 192 turns; AMD 1274 to 578 ms per step | $10.39 |
| [Run 6](run6/NOTES.md), a repeat of run 5 | **failed**: $10 cap at turn 197; graded outside the loop, llm.c's test fails | $10.13 |
| [Pure-oneAPI tuning](ptune/README.md), no vendor interop, no size thresholds | the run failed (300 turns), but its turn-209 version passes llm.c's test on CPU, an NVIDIA L4 and an AMD MI300X, at 1.55x CUDA on an RTX A4000 | $14.72 + $0.38 GPU |
| [Tuning with `gpu_test`](gtune/README.md) | success, 211 turns; NVIDIA 1.13-1.21x CUDA (1.36-1.52x pure oneMath), AMD 63 ms | $14.50 + $3.50 GPU |

**Integrity, runs 4 onward:**

- **No credentials:** the loop runs with no credentials in its environment
  ([`scripts/clean-env.sh`](../../scripts/clean-env.sh)).
- **Task folder:** fingerprinted before and after each run, and unchanged.
- **Secrets:** evidence is checked for secrets before commit. Run 4's shell had printed a key
  into its trace; it was redacted before any commit, and the key was reported for rotation.

The sections below describe the hidden check, and runs 1-3 and their GPU results as first
recorded.

## The hidden check

llm.c's own test, as Muse ports it (`test_gpt2.cpp`), run by a wrapper outside the work dir
([`task/run_test.py`](task/run_test.py)). It writes three things, compared with a reference at
0.05, llm.c's bf16 loss tolerance:

1. 1.0 if the test exits 0 and prints `overall okay: 1`. That covers llm.c's logit check, its
   16 gradient checks, its 10 losses against PyTorch, and its save, reload and replay.
2. 1.0 if the test's source keeps llm.c's tolerances: logits 25, loss 0.05 and the 16 gradient
   thresholds. Every assignment to them is read, so a loosened test fails (tests in
   `tests/test_bf16.py`).
3. The 10 step losses, checked by the wrapper itself against PyTorch's.

**Proven before any model ran, on an NVIDIA RTX 2000 Ada:**

| Candidate | okay | Worst loss vs PyTorch | Result |
|---|---|---|---|
| llm.c's own CUDA test, bf16, `nvcc` | 1 | 0.0254 | PASS |
| planted bug: GELU constant 0.04715 ([diff](planted/gelu.diff)) | 0 | within 0.05 | FAIL |

The planted bug's losses stay within tolerance; only llm.c's own logit and gradient checks catch
it, which is why the check needs `overall okay`.

## Runs 1-3: the first port, over three runs

- **Run 1** (34 turns, $0.70) ran with SYCLomatic, which failed to parse the trainer:
  `nvtx3/nvToolsExt.h`, NVIDIA's profiling header, is not in its CUDA headers. Muse read the
  sources for 30 turns, and the toolkit's per-run limit of 2M tokens, which counts cached
  re-reads, stopped the run before it wrote anything. `museport loop` gained `--max-total-tokens`
  and `--no-migrate` for the next run.
- **Run 2** (200 turns, $11.79) had no SYCLomatic, and the USD cap set the limit.
  - **Writing:** Muse read everything and checked bf16 and oneMath's matmul layout in small
    test programs. It then wrote the port in turns 48-65: 12 SYCL kernel headers, a trainer in
    three parts, and the test.
  - **Debugging:** with no tool that runs a program, Muse compiled small debug programs as the
    candidate and read their printout in the check's failure message. That way it found and
    fixed an invalid oneMath argument and an attention bug, and a layernorm bug: a reduction
    over a 32-wide sub-group, CUDA's warp, which is narrower on the CPU.
  - **Where it stopped:** the step-1 loss reached 5.2505 against PyTorch's 5.2700, so the
    forward pass was right. It then traced the training NaNs to an out-of-bounds write into the
    gradient buffer, located in attention's backward pass, when the run reached its 200 turns.
- **Run 3** (65 turns, $1.86) started from run 2's files.
  - **What it was given:** the goal added a note, taken from run 2's own trace, on where run 2
    stopped. When the test failed, the check now also showed the test's own printout.
  - **Fixes:** Muse fixed the out-of-bounds write (turn 27: finite losses, but too slow), then
    the gradient scale (turn 47: all 10 losses right; one gradient, `lnfw`, at 103% of its
    threshold), then that gradient. From turn 52 the test passed.
  - **Finish:** Muse deleted its debug programs, and the run stopped on success at turn 65:
    compile, verify, and 0 helpers in the work dir, on the same sources.

## The independent grade (run 3)

The check trusts what the compiled program prints, and the loop does not force that program to
be built from `test_gpt2.cpp`. J3's check has the same gap. So the result above is not taken
from the loop alone ([`independent-grade/`](independent-grade)):

- **Rebuild:** Muse's final files were copied to a clean folder and `test_gpt2.cpp` was built
  with `icpx -fsycl -O2 -DENABLE_BF16` and oneMath, with no task or SYCLomatic include path:
  14 s, no warnings.
- **The check on that build:** PASS. `overall okay: 1`, tolerances kept, and the worst loss
  0.0326 from PyTorch's ([grade.json](independent-grade/grade.json)).
- **llm.c's test output** ([test-output.txt](independent-grade/test-output.txt)):
  - logits OK, with the largest difference 9.46 against a tolerance of 25;
  - 16 of 16 gradient tensors OK;
  - 10 losses OK against PyTorch, and the 10 replayed losses bit-identical;
  - `overall okay: 1`, exit 0. The test took about 108 s on the CPU.
- **The code, read:**
  - `test_gpt2.cpp` is llm.c's test, with the same checks, thresholds and expected losses. It
    differs from `test_gpt2.cu` only in the trainer include, comments and small C++ changes
    (`std::max`, `fabsf`).
  - The model runs on the SYCL device: 32 SYCL kernels in `llmc/*.h`, and 18 oneMath BLAS calls
    for the matmuls.

## GPUs (run 3's port)

Muse's final files, unchanged, built with the toolkit's `gpu-test`, with oneMath built on each
machine. Each run costs under $0.20.

| GPU | Result |
|---|---|
| NVIDIA RTX 2000 Ada, oneMath as released | **aborts** in oneMath: its cuBLAS backend has no bf16 gemm ("not implemented"); llm.c's own CUDA test passes on the same GPU (worst loss 0.0254) ([attempt1](gpu/attempt1)) |
| NVIDIA RTX 2000 Ada, oneMath with [our patch](../../patches/onemath-cublas-bf16-gemm.patch) | **PASS**: `overall okay: 1`; worst loss 0.0358 vs PyTorch and 0.0319 vs llm.c's own CUDA test on the same GPU ([nvidia](gpu/nvidia)) |
| AMD Instinct MI350X, oneMath on rocBLAS | **FAIL by one tensor**: all 10 losses within 0.05 of PyTorch's, determinism OK, and 15 of 16 gradient tensors OK; `wpe`'s gradient is at 105% of llm.c's tightest threshold (4.27e-3 vs 4e-3), where the CPU run is at 83% ([amd](gpu/amd)) |

- **The oneMath gap:** upstream oneMath's cuBLAS backend declares `gemm` with bf16 inputs and
  an fp32 output, but throws "unimplemented" for it, for buffers and USM alike, while the
  half-precision version calls `cublasGemmEx`. The patch routes bf16 to the same helper with
  `CUDA_R_16BF`: two lines. It was written by Claude, the assistant running this project, not
  by Muse; Muse's port found the gap. It applies to the oneMath commit museport builds and to
  upstream `develop`.
- **AMD:** llm.c's test comments say its gradient thresholds were set by hand, and that "the
  actual errors can be hardware specific". The `wpe` gradient sums over only 4 sequences per
  position, and on the MI350X rocBLAS accumulates in a different order. That is the likely
  cause, but it is not proven, and by the check's rule this is a fail. AMD attempt 2 lost the
  failing line to a wrapper that kept only the last 3,000 characters
  ([attempt2](gpu/attempt2)); the committed wrapper now lists every failing check.
- **Speed, rough:** one training step of the test (B=4, T=64) took about 82 ms on the MI350X
  and about 2.5 s on the CPU. Not compared with CUDA.

## What else to know (runs 1-3)

- **Not one pass:** the port took three runs. Run 1 was stopped by our setup before Muse wrote
  any code; run 3 continued run 2's files.
- **The plain-C headers:** Muse also wrote its own copies of llm.c's plain-C headers
  (`utils.h`, `tokenizer.h`, `dataloader.h`, ...) instead of using them unchanged. Some are
  trimmed: `mfu.h`, for example, drops NVIDIA's monitoring library.
- **Two fixes to our check, recorded in [run2/NOTES.md](run2/NOTES.md):**
  - The wrapper crashed on a `-nan` loss. It was fixed mid-run, and since a NaN fails either way,
    no verdict changed.
  - From run 3, a failing test also shows its own printout.
- **Scope:** one run of the test per device. bf16 stochastic rounding and
  llm.c's tolerances are as in llm.c. Speed was not compared.

**Muse cost, runs 1-3:** $14.52 (plan $0.18 and runs $0.70, $11.79 and $1.86).
**GPU, runs 1-3:** $0.53 (the proof $0.008 and five GPU runs, $0.52).
