# Submission text (draft, plan item I4)

All numbers below link to committed evidence. Re-check them against `demo/data.json` and the
README before pasting.

## Name

**Muse Port:** Meta's Muse ports CUDA code off NVIDIA, and proves every step.

## One line

Muse Port has Meta's Muse Spark plan and port a real CUDA codebase (Karpathy's llm.c) to the
open SYCL standard. A port counts as done only when a check the model never sees agrees. The
ports then run on NVIDIA and AMD GPUs.

## Opening line of the 3-minute video (required)

> "Pre-existing: my open-source oneapi-agent-toolkit, the SYCL tools and agent loop, used here
> as a pinned dependency. Built this week: Muse Port. Meta's Muse plans and ports CUDA code
> against hidden checks, with reports, NVIDIA and AMD runs, and a port of GPT-2."

## What it does

1. **`museport scan`** inventories a CUDA repository without any model: kernels, NVIDIA
   library calls and their oneAPI equivalents, and CUDA-only constructs.
2. **`museport plan`:** Muse Spark writes a migration plan in a fixed schema. Every file,
   kernel, library mapping and risk it names is checked against the scan. On llm.c's
   `dev/cuda` the plan has 23 items, places 22 of 22 files and has 0 grounding issues, for
   $0.26 ([plan](../evidence/plan-llmc-devcuda-stream/README.md)).
3. **`museport port`:** Muse ports each plan item inside an agent loop: migrate, rewrite,
   compile, verify, remove helpers. The **hidden check** is built from llm.c's own CPU
   reference and tolerance. Before any model runs, each check is proven to pass a known-good
   port. After the run, the final sources are graded again.
4. **GPU runs:** each passing port runs unchanged on real GPUs.
   - **NVIDIA:** checked against the reference and against llm.c's original CUDA built with
     `nvcc`.
   - **AMD:** checked against the reference.
5. **`museport report`:** a static site built only from the committed evidence: results,
   every port, and a turn-by-turn replay of Muse's work.

## How Meta's models are used

- **Planning:** `muse-spark-1.3` plans the whole migration from the CUDA sources and the scan,
  over a streaming connection with a cost cap.
- **Porting:** `muse-spark-1.3` drives the agent loop. It decides what to read and what to
  write, fixes compile errors, and deletes the migration tool's leftovers.
- **Problems found in Muse's API, and their fixes:**
  - Non-streamed calls fail at about 30 s; the fix is a streaming client.
  - Turns can come back empty when reasoning uses the whole output budget; the fix marks them
    and carries on.
  - Usage is missing on some streams; the fix charges the worst case.

  All three are documented in the evidence.

## Results (measured; evidence in the repository)

**llm.c kernels, with the fixed agent loop** (one run per file):
- 19 kernel files ported.
- **18 correct with 0 helpers**, by the hidden check; 15 also met the loop's stricter rule.
- **15 pass on NVIDIA** against llm.c's own CUDA; the 10 Tier 1 ports also pass on an
  **AMD MI350X**.
- $3.75 in total ([README table](../README.md#results-llmc-devcuda-muse-spark-13)).
- Open issue, reported: `softmax_forward` fails the hidden check.

**The first runs** (same tasks and checks, before the loop fixes) got 17 correct, 8 loop
successes and 14 on NVIDIA for $5.51. They spent their turns clearing SYCLomatic's output file
by file. The fixes: `delete_file`, ranged `read_file`, a 32k output budget per turn, and no
token limit below the USD cap ([how we got here](../README.md#how-we-got-here-the-first-runs)).

**Without SYCLomatic:** the same 10 Tier 1 files with the migrate tool taken away. Muse
wrote every port from the CUDA source: **10 of 10** loop successes and 10 correct, in 5.3
turns per item, for $0.48. On these simple kernels the migration tool did not help. The one
file with warp-level CUDA got correct but plain kernels
([evidence](../evidence/port-llmc-tier1-nomigrate/README.md)).

**GPUs:** 15 ports pass on **NVIDIA** GPUs (RTX 2000 Ada, RTX A4500, L4, RTX 4090) against
both the reference and llm.c's own CUDA ([Tier 1](../evidence/port-llmc-tier1-run2/gpu/NVIDIA.md),
[Tiers 2-3](../evidence/port-llmc-tier23-run2/gpu/NVIDIA.md)). The 10 Tier 1 ports also pass
on an **AMD Instinct MI350X** against the reference
([AMD](../evidence/port-llmc-tier1-run2/gpu/AMD.md)). Ports that call oneMath are verified on
the CPU device.

**GPT-2, llm.c's real training code with the GPT-2 124M weights:**
- **Stage 1, the forward pass:** Muse wrote a 463-line SYCL port that matches PyTorch's logits
  and loss within llm.c's tolerance (worst error 0.0012). It took 28 turns and cost $0.74
  ([stage 1](../evidence/gpt2-stage1/README.md)).
- **Stage 2, the backward pass:** Muse wrote a 764-line port, with 16 SYCL kernels and 17
  oneMath calls. All 124,475,904 gradients match PyTorch's within llm.c's CPU-test tolerance of
  2e-2 (worst error 0.0124), with 0 helpers. It took 27 turns and cost $1.21
  ([stage 2](../evidence/gpt2-stage2/README.md)).
- **Stage 3, training:** 10 steps of forward pass, backward pass and AdamW update, llm.c's
  own test. Every step's loss matches PyTorch's within 1e-2 (worst error 0.00015). Muse
  wrote the 907-line port in one turn; the first verify failed, and Muse found and fixed
  its own matmul transpose mistake. It took 29 turns and cost $1.15
  ([stage 3](../evidence/gpt2-stage3/README.md)).

## Technical depth

- **Hidden checks:** they come from llm.c's own CPU code and PyTorch's debug state, at llm.c's
  own tolerances.
  - **Proven both ways:** each check passes a known-good implementation and fails a planted
    bug.
  - **Strict enough to matter:** the GPT-2 check fails SYCLomatic's fast-math output on 14 of
    256 rows.
- **Variant check:** it catches a port that copies one implementation into several kernel
  variants.
- **Grounded planning:** every claim in Muse's plan is checked against a deterministic scan.
- **Toolkit changes made this week:** five pull requests in the pre-existing toolkit, each
  logged in `docs/CHANGES-TO-TOOLKIT.md`:
  - Meta API alignment;
  - `native_flags`;
  - `delete_file` and `read_file` line ranges;
  - AMD header hipify;
  - reading included headers.

## Limits

- fp32 only, at reduced sizes for most kernels; one run per item.
- AMD is checked against the reference only: llm.c's CUDA doesn't convert to HIP
  automatically (errors quoted in the evidence).
- The GPT-2 ports ran on the CPU device. NVIDIA needs oneMath built with its cuBLAS backend
  (plan item D2).
- Muse's GPT-2 ports are lean rewrites with simple kernels, not llm.c's tuned kernels.
  Performance was not compared.

## Links

- **Repository:** https://github.com/checocadena/museport.
- **Report (live product):** https://claude.ai/artifact/5BnXqn4gNiefb81LWdADNC. *The author
  must share it publicly first.*
- **Demo assets:** [`demo/`](../demo/DEMO.md).
