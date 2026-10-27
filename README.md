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

## Results so far (llm.c `dev/cuda`, `muse-spark-1.3`)

| | Tier 1: plain kernels | Tier 2: warp/sub-group, shared memory | Tier 3: cuBLAS to oneMath | All |
|---|---|---|---|---|
| Items ported | 10 | 5 | 4 (of 7) | 19 |
| Correct with 0 helpers (hidden grade) | 9 | 4 | 4 | **17** |
| Also met the loop's strict success rule | 4 | 2 | 2 | 8 |
| Pass on an NVIDIA GPU (vs reference and llm.c's CUDA) | 9 | 4 | 1 (3 call oneMath: CPU-verified) | **14** |
| Muse cost | $1.67 | $2.45 | $1.40 | **$5.51** |

- **Three first runs had open issues.** In Tier 1, `fused_residual_forward` is correct but left
  c2s helpers. In Tier 2, `softmax_forward` is also correct with helpers left, and
  `layernorm_backward` copies one implementation into 8 of its 9 variants.
- **Reruns** of these three, with a stricter goal, are reported beside the first runs; the table
  above counts first runs only. `layernorm_backward` was fixed: a rerun is correct, has 0 helpers
  and keeps a distinct kernel for every variant. The other two still have issues: one rerun
  left 45 helpers and the other fails verify. Two rounds of reruns cost $2.85 in total
  ([round 1](evidence/port-llmc-reruns/README.md), [round 2](evidence/port-llmc-reruns-2/README.md)).
- **Loop fixes, then Tier 1 again:** two changes went in after the reruns. Muse can now delete
  SYCLomatic's output in one call and read long files in parts, and each turn can write up to
  32,000 tokens. With them, all 10 Tier 1 files ran again with the same tasks and turn limit
  ([evidence](evidence/port-llmc-tier1-run2/README.md)).
  - Loop success rose from 4 to **9 of 10**.
  - **10 of 10** are correct with 0 helpers.
  - Cost fell from $1.67 to **$1.01**, and turns per item from 11.5 to 8.2.
  - This is one run per item, with several changes made at once.
- **AMD:** all 10 of those Tier 1 ports **pass on an AMD Instinct MI350X** against the hidden
  reference, for $1.42 of GPU time ([evidence](evidence/port-llmc-tier1-run2/gpu/AMD.md)).
  - They ran on DigitalOcean droplets started from the author's laptop.
  - On AMD they are checked against the reference only. AMD's automatic CUDA-to-HIP
    conversion can't build llm.c's own CUDA (`__ldcs`, a host `memset`, 32-bit warp masks),
    so the comparison with llm.c's CUDA was done on NVIDIA.
- **Limits:** fp32 only, reduced sizes, one run per item. AMD has been tested for Tier 1 only.
- **Evidence:** [plan](evidence/plan-llmc-devcuda-stream/README.md),
  [Tier 1](evidence/port-llmc-tier1/README.md), [Tier 2](evidence/port-llmc-tier2/README.md),
  [Tier 3](evidence/port-llmc-tier3/README.md); every number above comes from a committed
  file.

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

**Tier 1** ([evidence](evidence/port-llmc-tier1/README.md)): Muse (`muse-spark-1.3`)
ported llm.c's 10 Tier 1 kernel files for $1.67 in total. 9 final ports are correct and
helper-free by the hidden grade (4 also met the loop's strict success rule within 12 turns).
All 9 pass on NVIDIA GPUs against both the reference and llm.c's original CUDA. fp32 only,
reduced sizes, one run each. The Tier 1 rerun's ports also pass on an AMD MI350X against the
reference (see above).

**Tier 2** ([evidence](evidence/port-llmc-tier2/README.md)): warp shuffles, cooperative groups,
shared memory. Muse ported llm.c's 5 Tier 2 files for $2.45 in total. 4 final ports are
correct and helper-free by the hidden grade, and all 4 pass on NVIDIA GPUs.
`classifier_fused` passes every variant, including the one SYCLomatic mistranslates. One
caveat, measured and reported: Muse's `layernorm_backward` is correct but copies one
implementation into 8 of its 9 variants.

**Tier 3** ([evidence](evidence/port-llmc-tier3/README.md)): cuBLAS and cuBLASLt to oneMath.
Muse ported 4 of llm.c's Tier 3 files for $1.40; all 4 are correct and helper-free by the
hidden grade, and each cuBLAS call became a oneMath call. Verified on the CPU device;
the one port with no library calls also passes on NVIDIA. The attention files and NCCL
were not attempted.

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
