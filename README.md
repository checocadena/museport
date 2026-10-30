# Muse Port

**Meta's Muse models plan and port real CUDA projects to the open SYCL standard, and prove
every step.**

Muse Port reads a whole CUDA repository, plans its migration off NVIDIA (kernels, NVIDIA
library calls and their oneAPI equivalents, CUDA-only assumptions), then drives an agent that
ports it piece by piece. Nothing counts as done until a check the model never sees confirms
the output matches the original. The ports are then run on real NVIDIA GPUs against the
original CUDA, and on an AMD MI350X against the hidden reference.

The largest result: Muse ported **llm.c's production bf16 GPT-2 trainer** in one run. Then, with a
tool that runs its code on rented GPUs, Muse tuned it to **1.13-1.21x llm.c's CUDA time on
NVIDIA**, with llm.c's own test passing on NVIDIA and AMD
([below](#llmcs-production-bf16-trainer-ported-in-one-run-then-tuned-on-real-gpus)).

> **Status: built during the Meta Global AI Developer Hackathon (submissions due Nov 4, 2026).**
> This repository was created on Oct 26, 2026, after the event's kickoff. Every commit in it
> is hackathon work. See [What is new](#what-is-new-and-what-it-builds-on).

## What is new, and what it builds on

- **Built during the hackathon (this repository):**
  - `museport scan`, `plan`, `loop`, `port`, `gpu-tasks` and `report`;
  - the hidden-check harness for llm.c's kernels, the variant check, and the streaming Muse
    client;
  - `museport loop --tune --gpu-queue` and `museport gpu-broker`: Muse's `gpu_test` tool,
    which runs its code on rented NVIDIA and AMD GPUs with a profiler, with the keys and the
    budget held outside the model's process;
  - every result under `evidence/`.

  This repository was created at the event and contains no earlier code or git history.
- **Pre-existing (the author's own, not in this repository):**
  [oneapi-agent-toolkit](https://github.com/checocadena/oneapi-agent-toolkit) (AGPL-3.0),
  installed by `pip` as a pinned dependency. It provides the evidence-returning tools
  (migrate, compile, scan_helpers, verify, benchmark), the agent loop and its model client,
  and the GPU test pipeline. Its pre-hackathon state is the tag
  [`pre_hackathon`](https://github.com/checocadena/oneapi-agent-toolkit/tree/pre_hackathon).
  Changes made to it during the event are listed in
  [`docs/CHANGES-TO-TOOLKIT.md`](docs/CHANGES-TO-TOOLKIT.md) and live in that repository.
- **Third-party:** the code ported is from
  [karpathy/llm.c](https://github.com/karpathy/llm.c) and
  [ankan-ban/llama2.cu](https://github.com/ankan-ban/llama2.cu) (both MIT); see
  [`THIRD_PARTY.md`](THIRD_PARTY.md).

## llm.c's production bf16 trainer, ported in one run, then tuned on real GPUs

llm.c's main trainer, `train_gpt2.cu` with `test_gpt2.cu` and the CUDA headers in `llmc/`, is
the code llm.c trains GPT-2 with. It is 12 files and 4,884 lines, in bf16 mixed precision, with
cuBLASLt, warp reductions and stochastic rounding. The hidden check is llm.c's own test, as
Muse ports it, with llm.c's bf16 tolerances. A guard fails any port that loosens them.
[Full write-up](evidence/l1-llmc-bf16/README.md).

1. **Port, in one run:** 125 turns, $5.67, from the CUDA alone, with a shell.
   - **On the CPU:** llm.c's test passes. That covers logits, all 16 gradient tensors, 10
     losses against PyTorch, and a bit-exact checkpoint replay.
   - **On GPUs:** it also passes on an AMD MI350X and an NVIDIA RTX 4000 Ada.
   - **Speed:** it was 36x slower than CUDA.
2. **Tuning on the CPU:** $10.39. NVIDIA reached 5.7x CUDA's time, and AMD went from 1274 to
   578 ms per step ([tune](evidence/l1-llmc-bf16/tune/README.md)).
3. **Tuning with GPUs:** 211 turns, $14.50, plus $3.50 of GPU over 23 calls.
   - **The tool:** each `gpu_test` call builds Muse's current code on a rented GPU. It returns
     llm.c's test result, ms per step next to llm.c's CUDA on the same GPU, and an nsys kernel
     profile.
   - **What Muse did:** it profiled, rewrote the slow kernels, measured each change, and fixed
     two crashes the GPU exposed ([gtune](evidence/l1-llmc-bf16/gtune/README.md)).

| Final port: llm.c's trainer, B=4, T=1024, 30 steps | ms per step | vs llm.c's CUDA on the same GPU | llm.c's test |
|---|---|---|---|
| NVIDIA RTX 4090 | 49.8 | **1.14x** | PASS |
| NVIDIA L4 | 160.1 | **1.14x** | PASS |
| NVIDIA RTX 2000 Ada | 205.5 | **1.13x** | PASS |
| NVIDIA RTX A4500 (our 3-run repeat) | 113.3 | **1.21x** | PASS |
| NVIDIA RTX 2000 Ada, cuBLAS interop compiled out (pure SYCL + oneMath; ours) | 250.6 | **1.36x** | PASS |
| NVIDIA RTX A4500, cuBLAS interop compiled out (ours) | 139.8 | **1.52x** | PASS |
| AMD Instinct MI350X | **63.5** | (llm.c's CUDA does not build on AMD) | PASS |

**What to know:**

- **cuBLAS on NVIDIA:** the NVIDIA 1.13-1.21x uses NVIDIA's cuBLAS at runtime, through SYCL's
  backend interop, for two bf16 gemm forms that oneMath's API does not have on any backend:
  batched bf16, and bf16 output. oneMath has both for fp16. Pure SYCL + oneMath is 1.36x on the
  RTX 2000 Ada (1.13x with cuBLAS) and 1.52x on the RTX A4500 (1.21x). That
  gap is a concrete upstream ask, like the bf16 gemm patch below.
- **Fast paths and llm.c's test:** Muse uses the fast paths only for large shapes, so llm.c's
  small test does not run them. We forced them on as a diagnostic, and llm.c's test passes on
  NVIDIA and on AMD (worst loss 0.046, against a tolerance of 0.05).
- **Not one pass overall:** the first port took three runs (runs 1-3), and run 4 stopped on the
  loop's stuck rule. Run 5 is the one-run port; a repeat with the same setup (run 6) failed within
  $10, so a one-run port succeeded in 1 of 2 tries.
- **Pure oneAPI, with the test running training's code:** a further run banned vendor interop and
  size thresholds. Its own loop did not succeed, but one version it built passes llm.c's test on
  the CPU, an NVIDIA L4 and an AMD MI300X, at 1.55x CUDA on an RTX A4000
  ([ptune](evidence/l1-llmc-bf16/ptune/README.md)).
- **Measurement:** rented GPUs vary between calls, so compare ratios. Each number comes from one
  run.
- **A oneMath fix:** oneMath's cuBLAS backend throws "unimplemented" for bf16 gemm. The port
  found it, and a two-line patch by us, not Muse, routes it to `cublasGemmEx`
  ([patch](patches/onemath-cublas-bf16-gemm.patch)).
- **Cost:** Muse $75.50 for all of L1 (of the $80 cap), including the failed repeat and the
  pure-oneAPI run; GPU about $7.20.

## Results (llm.c `dev/cuda`, `muse-spark-1.3`)

Muse ported 19 of llm.c's CUDA kernel files with the fixed agent loop: one run per file, each
graded by a hidden check built from llm.c's own CPU code and tolerance.

| | Tier 1: plain kernels | Tier 2: warp/sub-group, shared memory | Tier 3: cuBLAS to oneMath | All |
|---|---|---|---|---|
| Items ported | 10 | 5 | 4 (of 7) | 19 |
| Correct with 0 helpers (hidden grade) | 10 | 4 | 4 | **18** |
| Also met the loop's strict success rule | 9 | 3 | 3 | 15 |
| Pass on an NVIDIA GPU (vs reference and llm.c's CUDA) | 10 | 4 | 3 | **17** |
| Pass on an AMD GPU (vs reference) | 10 | 4 | 4 | **18** |
| Muse cost | $1.01 | $1.69 | $1.06 | **$3.75** |

- **Evidence:** [Tier 1](evidence/port-llmc-tier1-run2/README.md),
  [Tiers 2 and 3](evidence/port-llmc-tier23-run2/README.md). Every number comes from a
  committed file, and a test checks this table against them.
- **What "correct" means:** the final sources compile, match the hidden reference within
  llm.c's fp32 tolerance, and no SYCLomatic (dpct) helper is left anywhere in the tree. The
  loop's own rule is stricter: verify and the helper scan must pass on the same compiled
  sources inside the run.
- **Open issues, measured and reported:**
  - `softmax_forward` fails the hidden check (36,435 of 25.7 million values outside
    tolerance) in this run. Muse found the failing variant but ran out of turns without a fix.
    With the stricter goal, 30 turns and a $3 cap, a later run **passes**: loop success in 23
    turns for $0.91, 8 distinct variant kernels, and a pass on NVIDIA against llm.c's CUDA
    ([evidence](evidence/port-llmc-softmax-rerun-2/README.md)). A run in between, at a $0.70
    cap, fixed the copied variants but ran out of budget
    ([evidence](evidence/port-llmc-softmax-rerun/README.md)).
  - `softmax_forward` and `layernorm_backward` copy one implementation into several kernel
    variants; the variant check reports it. A stricter goal fixed `layernorm_backward` in an
    earlier rerun ([round 2](evidence/port-llmc-reruns-2/README.md)).
  - `global_norm` and `trimat_forward` are correct but ended one step short of the loop's
    rule.
- **GPUs:**
  - **NVIDIA:** RTX 2000 Ada and RTX A4500 on Runpod, against both the hidden reference and
    llm.c's own CUDA built with `nvcc`
    ([Tier 1](evidence/port-llmc-tier1-run2/gpu/NVIDIA.md),
    [Tiers 2 and 3](evidence/port-llmc-tier23-run2/gpu/NVIDIA.md)).
  - **AMD:** the same ports pass on an AMD Instinct MI350X against the hidden reference
    ([Tier 1](evidence/port-llmc-tier1-run2/gpu/AMD.md),
    [Tiers 2 and 3](evidence/port-llmc-tier23-run2/gpu/AMD.md)), and so does the
    `softmax_forward` rerun ([evidence](evidence/port-llmc-softmax-rerun-2/gpu/AMD.md)). Tier 1
    ran on DigitalOcean droplets started from the author's laptop; Tiers 2 and 3 ran from the
    cloud session, once `remote-test --public-tls` made that possible ($0.72). AMD's automatic CUDA-to-HIP conversion can't build
    llm.c's own CUDA (`__ldcs`, a host `memset`, 32-bit warp masks), so the comparison with
    llm.c's CUDA was done on NVIDIA.
  - **Ports that call oneMath** (the 3 Tier 3 files that call cuBLAS) run on GPUs with oneMath
    built on the machine, with cuBLAS or rocBLAS ([evidence](evidence/d2-onemath-gpu/README.md)).
    - All 3 pass on AMD.
    - On NVIDIA, 2 pass. `matmul_backward` aborts there: it calls row-major gemm, which
      oneMath's cuBLAS backend does not implement.
    - A column-major re-port of `matmul_backward` passes on NVIDIA, identical to llm.c's CUDA,
      for $0.09. It is the one `llm.c-sycl` uses. The table counts run 2's port, as for every
      item.

### GPT-2: llm.c's real training program

Muse ported llm.c's fp32 GPT-2 training file (`train_gpt2_fp32.cu`) to SYCL in three stages,
with the real GPT-2 124M weights and llm.c's debug batch. Each stage is checked against
PyTorch's numbers at llm.c's own test tolerance, and each check was first proven to pass
llm.c's CPU code and to fail a planted bug.

| Stage | What is checked | Result | Turns | Muse cost |
|---|---|---|---|---|
| [1. Forward](evidence/gpt2-stage1/README.md) | logits and loss, 1e-2 | worst error 0.0012 | 28 | $0.74 |
| [2. Backward](evidence/gpt2-stage2/README.md) | all 124,475,904 gradients, 2e-2 | worst error 0.0124 | 27 | $1.21 |
| [3. Training](evidence/gpt2-stage3/README.md) | the loss of 10 AdamW steps, 1e-2 | worst error 0.00015 | 29 | $1.15 |

- All three met the loop's success rule with 0 helpers.
- **Stage 3:** Muse wrote the 907-line port in one turn. The first verify failed on all 10
  steps; Muse found its own matmul transpose mistake ("Losses are far too high — my
  non-batched oneMath transpose mapping is wrong") and fixed it.
- **Stage 2's tolerance:** 2e-2 is llm.c's CPU-test tolerance for gradients. At the CUDA
  test's 1e-2, llm.c's own CPU code fails one gradient.
- **On an NVIDIA GPU:** stage 3's port, unchanged, trains on an RTX 4000 Ada with oneMath on
  cuBLAS. Its 10 losses match PyTorch's (worst error 0.000135) and llm.c's own CUDA build
  (worst 0.0000059), for $0.01 of GPU time ([evidence](evidence/gpt2-stage3/gpu/NVIDIA.md)).
- **On an AMD GPU:** the same port, unchanged, trains on an AMD Instinct MI350X with oneMath on
  rocBLAS. Its 10 losses match PyTorch's (worst error 0.0000246). It took five runs ($0.65)
  to supply the development files the cloud image lacked
  ([evidence](evidence/gpt2-stage3/gpu/AMD.md)). One source now trains GPT-2 on a CPU, an
  NVIDIA GPU and an AMD GPU.
- **Limits:** the ports are lean rewrites with simple kernels, not llm.c's tuned kernels.
  Stages 1 and 2 ran on the CPU device only, and kernel performance was not compared.

### `llm.c-sycl`: the ports put back together as a repo

[`llm.c-sycl/`](llm.c-sycl/README.md) is llm.c laid out as llm.c is, in SYCL. Its programs run
on a CPU, an NVIDIA GPU and an AMD GPU. From a fresh clone, `./dev/download_starter_pack.sh`,
`make` and `./test_gpt2fp32sycl` print `overall okay: 1` on the CPU
([evidence](evidence/j3-fresh-clone/README.md)).

- **`dev/sycl`:** the 19 kernel ports, unchanged from their graded runs. They pass 19 of 19
  hidden checks with 0 helpers ([evidence](evidence/j3-assemble/README.md)).
- **`train_gpt2_fp32.cpp`:** llm.c's fp32 trainer with its real `main` (options, dataloaders,
  validation, sampling). Muse wrapped its proven stage 3 port in it, keeping 94% of the lines,
  in 19 turns for $0.71 ([evidence](evidence/j3-trainer/README.md)).
- **`test_gpt2_fp32.cpp`:** llm.c's own test, ported by Muse. It checks logits, loss and every
  gradient against PyTorch, then 10 training steps, and prints `overall okay: 1`
  ([evidence](evidence/j3-test/README.md)).
  - It took three runs ($1.62). The third succeeded once the toolkit gained an `edit_file`
    tool: Muse guarded the trainer's `main` with `#ifndef TESTING` in two edits.

| Program | CPU | NVIDIA RTX 2000 Ada | AMD MI350X |
|---|---|---|---|
| `test_gpt2fp32sycl`: `overall okay: 1`, worst error vs PyTorch | 0.000154 | 0.000127 | 0.0000236 |
| `train_gpt2fp32sycl`: worst error vs llm.c's CPU trainer | 0.00227 | 0.0023 | 0.00242 |

On NVIDIA, both also match llm.c's own CUDA programs: 7.03e-06 for the test and 0.00147 for the
trainer. The checks were first proven on llm.c's own CPU programs and on planted bugs: a
learning-rate bug and an unshuffled dataloader fail the trainer check, and a GELU-constant bug
fails the test. Muse cost $2.33, and the GPU runs $0.43.

### A second CUDA repo (one 772-line file), end to end: `llama2-sycl`

[ankan-ban/llama2.cu](https://github.com/ankan-ban/llama2.cu), Llama 2 inference in one CUDA
file (772 lines, 8 kernels, fp16, CUB reductions; the rest of the repo is C and Python, not
CUDA), went through the same pipeline. Only the check's wrapper and a Makefile target were
written by hand for it ([evidence](evidence/j4-llama2/README.md)):

- **Plan:** Muse planned it as one item, grounded against the scan, for $0.029.
- **Check:** greedy stories from the `stories15M` model (two prompts, 64 steps) must match
  llama2.cu's own output byte for byte. The reference is llama2.cu, recorded on an NVIDIA GPU.
  Proven first: llama2.c's `run.c` passes, and two planted bugs fail.
- **Port:** Muse wrote a 737-line SYCL `llama2.cpp` in one run of 20 turns, for $0.45: 2 of 2
  stories identical, 0 helpers.
- **GPUs:** identical on an NVIDIA RTX 2000 Ada, against the recording and against llama2.cu run
  live on the same GPU, and on an AMD MI350X against the recording. GPU cost $0.13.
- **Repo:** [`llama2-sycl/`](llama2-sycl/README.md) is llama2.cu's repo with the port in place
  of `llama2.cu`. From a fresh clone, `make llama2` and the run reproduce both stories on the
  CPU ([evidence](evidence/j4-fresh-clone/README.md)).

### How much the tooling helps, and Muse as a reviewer

Three experiments take a piece away, or add a second agent. Each is one run per item.

- **Without SYCLomatic** (`museport port --no-migrate`, the 10 Tier 1 files,
  [evidence](evidence/port-llmc-tier1-nomigrate/README.md)):
  - **10 of 10** met the loop's success rule and are correct with 0 helpers, in 5.3 turns per
    item, for **$0.48** (with SYCLomatic: 9 of 10, 8.2 turns, $1.01).
  - On these simple kernels SYCLomatic did not help. The one file with warp-level CUDA
    (`fused_residual_forward`) got correct but plain kernels, without its shared-memory and
    warp structure.
- **Without feedback** (`--baseline`: no `verify`, `scan_helpers` or `benchmark`; Muse never
  runs its port, and the hidden grade checks it after):
  - **Tier 1, three runs per item in each arm:** 28 of 30 correct with 0 helpers without
    feedback, against 29 of 30 with it. Every miss is `fused_residual_forward` running out of
    turns with dpct helpers left (2 of 3 runs without feedback, 1 of 3 with). The other 9 files
    are correct in every run of both arms. Without feedback, runs take 9.2 turns per item
    against 7.8, and cost $3.57 against $2.98 for the three runs
    ([evidence](evidence/e1-c6-repeats/README.md)).
  - **GPT-2 training (stage 3):** the port passes (worst error 0.00149), for $1.34. With
    feedback, Muse's first version had a transpose bug that only the check caught; this time
    its first version was right ([evidence](evidence/gpt2-stage3-baseline/README.md)).
- **Muse as a reviewer** (`museport review`): a second Muse call reviews each of 20 finished
  ports from the code alone, with no test results, for $0.59
  ([evidence](evidence/review-muse/README.md)).
  - It caught both ports that copy variants, and flagged no other.
  - It judged the one failing port "likely correct": a code review is not a correctness check.
  - It found a real latent bug the tests miss: divergent `group_barrier`s in the passing
    `softmax_forward` fix, at sizes the hidden check does not use.
  - It flagged 4 correct ports whose variants drop the CUDA variant's parallel structure,
    which neither check can see.

### Hidden checks without a hand-written reference

Each kernel's hidden reference was a C++ program written from llm.c's CPU code. A differential
check replaces it with **the original CUDA's own output**, recorded on an NVIDIA GPU
([evidence](evidence/diff-llmc-tier1/README.md)):

- **Recording:** all 10 Tier 1 kernels were recorded for **$0.034**. Each program's two runs
  were bit-identical.
- **Recordings vs llm.c's CPU references:** they agree on 10 of 10.
- **Muse's ports vs the recordings:** they pass on 10 of 10, the same verdicts as before.
- **Planted bugs:** a dropped-digit typo in gelu fails. A typo smaller than the tolerance passes
  both references; that is the limit of any tolerance-based check.

### Hidden checks written by Muse

The specs those checks are built from (CPU reference, variants, inputs, sizes, tolerance) were
written by hand. Muse now writes them from the CUDA file alone, and each one is proven
([evidence](evidence/spec-llmc/README.md)):
- **Proof:** llm.c's own GPU code and CPU code must agree under the spec, checked through
  SYCLomatic on the CPU, or on an NVIDIA GPU where SYCLomatic fails.
- **Proven:** 16 of llm.c's 17 files (2 were worked examples, not graded). **All 16 are as strict
  as llm.c's own check**, once the prompt says the build is fp32: in the first run, 3 specs took
  llm.c's bf16 tolerance, and one dropped an fp32 variant
  ([rerun](evidence/spec-llmc-fp32/README.md)).
- **Not proven:** `trimat_forward`. The GPU proof caught its masked-triangle mistake each time.
- **Cost:** about $1.64–1.82 of Muse and $0.04 of GPU.

**Limits of these results:** fp32 only, reduced sizes for the kernels, one run per item
(the bf16 trainer above is full size).

## How we got here: the first runs

The first runs used the same tasks and hidden checks, before the loop fixes. Most ports were
already correct, but the runs scored badly on the loop's own rule:

| | Tier 1 | Tier 2 | Tier 3 | All |
|---|---|---|---|---|
| Items ported | 10 | 5 | 4 | 19 |
| Correct with 0 helpers (hidden grade) | 9 | 4 | 4 | 17 |
| Also met the loop's strict success rule | 4 | 2 | 2 | 8 |
| Pass on an NVIDIA GPU (vs reference and llm.c's CUDA) | 9 | 4 | 1 | 14 |
| Muse cost | $1.67 | $2.45 | $1.40 | $5.51 |

Why:

- **No way to delete files.** The success rule needs zero helpers anywhere in the tree,
  including SYCLomatic's own output. Muse emptied those files by overwriting each with a stub,
  one turn per file: 22 of Tier 1's 36 file writes.
- **Reading long files whole,** which burned tokens and turns.
- **Limits:** the toolkit stopped any run at 600,000 tokens (4 runs ended there, under their
  USD cap), and a turn could write only 16,000 tokens, which cut off large ports.
- **Connection problems** with Meta's API: empty answers and dropped connections could end a
  run.

The fixes: `delete_file` and ranged `read_file` in the toolkit, 32,000 output tokens per turn,
no token limit below the USD cap, and lost turns marked instead of fatal. Two earlier rounds of
reruns with a stricter goal cost $2.85 ([round 1](evidence/port-llmc-reruns/README.md),
[round 2](evidence/port-llmc-reruns-2/README.md)).

First-run evidence: [Tier 1](evidence/port-llmc-tier1/README.md),
[Tier 2](evidence/port-llmc-tier2/README.md), [Tier 3](evidence/port-llmc-tier3/README.md).

## Try it

```bash
git clone https://github.com/checocadena/museport && cd museport
pip install -e .
museport scan path/to/cuda/repo          # Markdown inventory; --json for machine-readable
```

`museport scan` is deterministic (no model, no compiler): it lists every CUDA file and kernel,
the NVIDIA libraries used with their oneAPI counterparts, and the CUDA-only constructs that
need a deliberate SYCL design, following local `#include`s. It is the ground truth a Muse
plan is checked against. Example: [llm.c](docs/examples/llm-c-scan.md).

```bash
museport plan path/to/repo --dry-run     # offline: a scripted server returns the mechanical plan
museport plan path/to/repo               # Muse Spark writes the plan (Meta Model API key needed)
```

`museport plan` sends the CUDA sources and the scan to Muse, which must answer with a plan in
a fixed JSON schema: ordered items with files, kernels, tier, approach, oneAPI library
mappings, risks, a verification method and dependencies. The plan is then **grounded**:
every file and kernel it names must exist, every CUDA file must be planned or explicitly left
out, every NVIDIA library needs a oneAPI mapping, every CUDA-only construct must appear as a
risk, tiers must match the scan, and dependencies must form a valid order. Requests whose
worst-case cost exceeds `--max-cost` (default $1) are refused before they are sent.

```bash
museport port plan.json --cuda-dir path/to/llm.c/dev/cuda --tier 1 --out ports   # Muse ports each item
museport port plan.json --cuda-dir ... --tier 1 --tasks-only --check             # no model: prove each check
museport loop task.json --max-cost 0.5     # the agent loop on one toolkit task, driven by Muse
museport gpu-tasks ports                   # remote-test tasks for the ports that pass the hidden grade
```

`museport port` turns each plan item of a tier into a task for the toolkit's agent loop, which
Muse drives over a streaming connection with a USD cap per item and per batch. The model
ports a CUDA harness: the original file, with its CPU reference removed and a `main` that
runs every kernel variant. A hidden reference built from that CPU code checks the result
with llm.c's own tolerance. Each check is proved before any model runs, in one of two ways:
SYCLomatic's raw migration passes it, or llm.c's own CUDA passes it on a GPU. After the
loop, the final sources are graded again by a check the model never sees, and a variant
check reports any port that copies one implementation into several variants.

**Requirements:**
- **`plan`, `loop`, `port`:** a Meta Model API key, or `ONEAPI_AGENT_AUTH_FROM_NETWORK=1`
  where a proxy adds it.
- **`loop` and `port`:** the toolkit's toolchain (`scripts/install-toolchain.sh`: `icpx`,
  SYCLomatic, CUDA headers), activated with `source scripts/oneapi-env.sh`.
- **Tier 3:** oneMath as well (`WITH_ONEMATH=1`).
- **GPU runs:** a Runpod, Vast, Vultr or DigitalOcean key for `oneapi-agent remote-test`.

```bash
museport report evidence --out site      # a static site built only from the committed evidence
```

`museport report` turns `evidence/` into a browsable site: the results by tier, every item, the
plan with its grounding checks, and one page per port. Each port page has the hidden grade,
the variant check, the GPU steps, the CUDA harness beside Muse's SYCL port with a diff, and a
turn-by-turn replay of the loop (what Muse wrote, each tool call and its result). No number is
typed in by hand, and a test checks this README's results table against the same evidence.

## Plan

See [`docs/PLAN.md`](docs/PLAN.md) (the original plan) and
[`docs/MASTER-PLAN.md`](docs/MASTER-PLAN.md) (what remains until submission).

## License

Copyright (C) 2026 Checo Cadena. [AGPL-3.0](LICENSE), like the toolkit it builds on.
