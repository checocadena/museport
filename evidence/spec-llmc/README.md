# Harness specs written by Muse, then proven (plan item J2)

Every hidden check in this project is built from a short spec per CUDA file:
- the CPU reference functions;
- the kernel variants the port must keep;
- the inputs and sizes, the two calls, and the tolerance.

`harness_source` and `reference_source` then build the CUDA contract harness and the hidden
reference from llm.c's file mechanically. Until now I wrote each spec by hand (`llmc.py`).
Here `muse-spark-1.3` wrote the spec for 17 of llm.c's `dev/cuda` files from the CUDA file
alone. `gelu_forward` and `global_norm` were its worked examples and are not graded.

## How a spec is checked (`museport spec`, `src/museport/spec.py`)

1. **Static checks, free:**
   - the CPU functions exist in the file;
   - the variants are the dispatcher's `case` labels, and any left out need a reason;
   - sizes and counts evaluate (arithmetic only), and the buffers fit in 512 MB.
2. **CPU proof, free:** SYCLomatic migrates the generated harness, icpx compiles the raw output,
   and it must match the reference built from llm.c's own CPU code. This is how the hand-written
   specs were proven.
3. **GPU proof, about $0.003–0.007 (`--gpu-proof`).** SYCLomatic fails on 5 of these files even
   with my hand-written specs. When the CPU proof fails at migrate, compile or verify, llm.c's
   harness is built with `nvcc` on a rented NVIDIA GPU, its outputs are recorded (J1), and they
   are compared here with the same CPU reference.

The spec is wrong only when every proof fails. Muse then gets the errors as feedback, for up to
3 attempts.

## Results ([SUMMARY.md](SUMMARY.md), [summary.json](summary.json))

| | Files |
|---|---|
| Spec proven | **16 of 17** |
| ...by the loop, CPU proof | 11 (10 on the first attempt; `layernorm_backward` on the third, fixing its own spec from the feedback) |
| ...by the loop, GPU proof | 3 (`matmul_forward`, `matmul_backward_bias`: first attempt; `matmul_backward` needed no GPU) |
| ...by a GPU proof after the run | 2 (`softmax_forward`, `classifier_fused`: Muse's first spec, unchanged; see below) |
| Not proven | 1: `trimat_forward` |
| Proven and as strict as llm.c's own check (its fp32 tolerance, every fp32 variant) | **13 of 16** |

**Every proven spec uses llm.c's own backward or forward CPU function as the reference.**
Two names differ from my lists:
- `layernorm_backward`: Muse computes the `mean`/`rstd` inputs with its own loop in `host_prep`
  (the same layernorm formula), instead of keeping `layernorm_forward_cpu` as I do.
- `softmax_forward`: Muse does not list `softmax_forward_online_cpu`, which the check does not
  call.

Sizes are Muse's own reductions and usually differ from mine.

## What went wrong, and what it shows

- **Three specs prove but are looser than llm.c.** llm.c's mains pick the tolerance with
  `#if defined(ENABLE_BF16) ... #else`. For `gelu_backward` and `residual_forward`, Muse took the
  bf16 value (1e-2) instead of fp32's 1e-5. For `matmul_backward_bias` it took 1.0 instead of
  5e-3. It also dropped variant 5 as "fp32-only (guarded by `#ifndef ENABLE_BF16`)", but this is
  the fp32 build, so variant 5 belongs in. A proof shows llm.c's GPU and CPU code agree under the
  spec; it cannot show the spec is as strict as llm.c's. Telling Muse plainly that the build is
  fp32 fixed all three: see the rerun in [`../spec-llmc-fp32`](../spec-llmc-fp32/README.md).
- **`trimat_forward` is not proven** (two runs, $0.32): 2 attempts, then 3 once an API error
  no longer ended a file's attempts.
  - Both runs' first specs fail on the GPU with about half the values off: 4,112,984 of
    7,864,320 in the first run. That is the causal-masked triangle. llm.c's CPU code writes NaN
    there and its `validate_result` skips them; my spec zeroes them in both programs, and Muse's
    does not.
  - The GPU proof caught it, as it should.
  - In both runs the next answer was empty (`finish_reason=stop`, no content). The second run's
    third spec did not compile.
- **The CPU proof can mislead.** Before the GPU proof was in the loop (first run), the
  SYCLomatic failures on `softmax_forward` and `classifier_fused` went back to Muse as if its
  spec were wrong. On `softmax_forward`, Muse then dropped a correct variant to "fix" a compile
  failure that was SYCLomatic's. Its first specs for both files were right. A GPU proof of every
  spec Muse wrote for them, made after the run, passes:
  - `softmax_forward`: worst error 8.3e-07, atol 1e-4;
  - `classifier_fused`: worst error 1.9e-06, atol 1e-4.

  The proofs are in [`softmax_forward/gpu-proof/`](softmax_forward/gpu-proof/) and
  [`classifier_fused/gpu-proof/`](classifier_fused/gpu-proof/). That run is why the GPU proof
  now comes before any feedback.
- **`classifier_fused`'s spec injects llm.c's x20 outliers with `rand()`** in both programs. That
  is reproducible here (glibc on both machines), but less portable than a seeded generator.
- **Three bugs of mine, each fixed with a test:**
  - A relative `--out` made the toolkit refuse migrate's output, so the first run's first proof
    failed before Muse's spec was even tested. That run was stopped.
  - The first GPU comparison rebuilt the inputs from my hand-written spec instead of using the
    task's own. That crashed `classifier_fused`'s reference, and `softmax_forward`'s first pass was
    a lucky match. Both were rerun with the fix. J1's results are unaffected: their tasks came
    from the hand-written specs.
  - An API error ended a file's attempts instead of moving on to the next one.

## Cost

| | USD |
|---|---|
| Muse, the attempts in `summary.json` | 1.133 |
| Muse, outside it: the smoke test (0.026), the stopped first run (0.0245), the stopped `matmul_forward` attempt (0.034) and the first `trimat_forward` run (0.1195) | 0.204 |
| Muse, possibly in flight when two runs were stopped | at most about 0.18 |
| **Muse total** | **about 1.34, at most 1.52** |
| GPU proofs: in the loop 0.010; first `trimat_forward` run 0.007; after-run proofs 0.012; one test of `gpu_proof` on my softmax spec 0.005 | **0.034** |

An attempt with an empty answer reports no usage and is counted at its worst case.

## Files

- **Each `<file>/attempts.json`:** every answer, its static errors, its CPU proof and its GPU
  proof.
- **`spec.json`:** the proven spec.
- **`task/`:** the generated harness, reference and `task.json`. `input.bin` is left out; it is
  regenerated from the spec.
- **`gpu-proof/`:** the GPU proofs' logs and results. The recorded outputs are left out; their
  sha256 hashes are kept.
