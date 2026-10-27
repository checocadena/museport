# Muse Port

**Meta's Muse models plan and port real CUDA projects to the open SYCL standard, and prove
every step.**

Muse Port reads a whole CUDA repository, plans its migration off NVIDIA (kernels, NVIDIA
library calls and their oneAPI equivalents, CUDA-only assumptions), then drives an agent that
ports it piece by piece. Nothing counts as done until a check the model never sees confirms
the output matches the original. The ports are then run on real NVIDIA GPUs against the
original CUDA, and on an AMD MI350X against the hidden reference.

> **Status: built during the Meta Global AI Developer Hackathon (submissions due Nov 4, 2026).**
> This repository was created on Oct 26, 2026, after the event's kickoff. Every commit in it
> is hackathon work. See [What is new](#what-is-new-and-what-it-builds-on).

## What is new, and what it builds on

- **Built during the hackathon (this repository):**
  - `museport scan`, `plan`, `loop`, `port`, `gpu-tasks` and `report`;
  - the hidden-check harness for llm.c's kernels, the variant check, and the streaming Muse
    client;
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
- **Third-party:** the kernels ported are from
  [karpathy/llm.c](https://github.com/karpathy/llm.c) (MIT); see
  [`THIRD_PARTY.md`](THIRD_PARTY.md).

## Results (llm.c `dev/cuda`, `muse-spark-1.3`)

Muse ported 19 of llm.c's CUDA kernel files with the fixed agent loop: one run per file, each
graded by a hidden check built from llm.c's own CPU code and tolerance.

| | Tier 1: plain kernels | Tier 2: warp/sub-group, shared memory | Tier 3: cuBLAS to oneMath | All |
|---|---|---|---|---|
| Items ported | 10 | 5 | 4 (of 7) | 19 |
| Correct with 0 helpers (hidden grade) | 10 | 4 | 4 | **18** |
| Also met the loop's strict success rule | 9 | 3 | 3 | 15 |
| Pass on an NVIDIA GPU (vs reference and llm.c's CUDA) | 10 | 4 | 1 (3 call oneMath: CPU-verified) | **15** |
| Pass on an AMD GPU (vs reference) | 10 | 0 (not run) | 0 (not run) | **10** |
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
    tolerance). Muse found the failing variant but ran out of turns without a fix. A rerun
    with the stricter goal and 20 turns fixed the copied variants but still fails the check
    ([evidence](evidence/port-llmc-softmax-rerun/README.md), $0.75).
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
  - **AMD:** the 10 Tier 1 ports pass on an AMD Instinct MI350X against the hidden reference
    ([evidence](evidence/port-llmc-tier1-run2/gpu/AMD.md)). They ran on DigitalOcean droplets
    started from the author's laptop. AMD's automatic CUDA-to-HIP conversion can't build
    llm.c's own CUDA (`__ldcs`, a host `memset`, 32-bit warp masks), so the comparison with
    llm.c's CUDA was done on NVIDIA.
  - Ports that call oneMath are verified on the CPU device: oneMath is not built for the GPU
    hosts yet.

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
- **Limits:** the ports are lean rewrites with simple kernels, not llm.c's tuned kernels.
  Stages 1 and 2 ran on the CPU device only, and kernel performance was not compared.

### How much SYCLomatic helps

The same 10 Tier 1 files, with the `migrate` tool (SYCLomatic) taken away
(`museport port --no-migrate`), so Muse writes each port from the CUDA source
([evidence](evidence/port-llmc-tier1-nomigrate/README.md)):

- **10 of 10** met the loop's success rule and are correct with 0 helpers, in 5.3 turns per
  item, for **$0.48** (with SYCLomatic: 9 of 10, 8.2 turns, $1.01).
- On these simple kernels SYCLomatic did not help. The one file with warp-level CUDA
  (`fused_residual_forward`) got correct but plain kernels, without its shared-memory and warp
  structure.

**Limits of all results:** fp32 only, reduced sizes for the kernels, one run per item.

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
