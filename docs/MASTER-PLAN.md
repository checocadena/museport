# Master plan (Oct 27 to Nov 4)

Submission: **Wed Nov 4, 9:00 pm ET**; target midday. It needs:
- the public repository (this one: hackathon work only);
- a 3-minute video that opens with one line on what was pre-existing and what was built;
- the live product.

Judging: Impact 30%, Use of Meta AI tooling 25% ("beyond a basic integration"), Technical
depth 20%, Demo 25%. Bonus prizes in reach:
- Best Agentic Use-Case;
- Best Build on Meta Open Weight Models;
- Most Viral on X;
- Community Vote.

Rules for every item:
- **Paid steps:** a cost estimate and the author's OK before each paid step (API or GPU).
- **Evidence:** no result is claimed until it is measured and its evidence is committed.
- **Pull requests:** small PRs; the author approves merges.
- **Toolkit changes:** logged in `CHANGES-TO-TOOLKIT.md`.

Budget so far:
- **Development:** about $0.55 of $5, plus the first failed plan run (at most $1.15).
- **Comparison runs:** about $22.19 of $30, at most $22.37 (after the C1-C3 reruns, Tier 1 run 2,
  the run without SYCLomatic, GPT-2 stages 1-3, Tiers 2-3 run 2, two softmax_forward reruns, E1,
  E3 and J2's Muse specs, about $1.64-1.82 with the fp32 rerun). The author raised
  the per-run cap from $1 to $3 on Oct 27, and this cap from $20 to $30 on Oct 27.
  The author set a $7 cap for GPT-2 (G3); $4.45 used, including E1's GPT-2 run.
- **GPUs:** about $4.10 (Runpod NVIDIA about $0.31, DigitalOcean AMD about $3.79), of a $10 cap
  the author set on Oct 27. Oct 28 added the J1 recordings ($0.034), Tiers 2-3 and softmax on
  AMD ($0.72), GPT-2 stage 3 on AMD (five runs, $0.65) and J2's GPU proofs ($0.041).
**The author sets their caps before they start.**

## Status at the start (Oct 26)

Done:
- grounded Muse plan of llm.c `dev/cuda`;
- 19 kernel files ported by `muse-spark-1.3`: 17 correct with 0 helpers; 14 pass on NVIDIA
  against llm.c's CUDA;
- the hidden-check harness, the variant check, and the streaming client.

See the README's results table.

---

## A. Repository and submission basics (do first)

- [x] **A1. Push this extract to the fresh public repo** as one initial commit; check that
  its CI is green; fill in the repo URL in the README clone command and `docs/SUBMISSION.md`.
  *Cost: $0. Needs: the empty repo, attached to the session.*
- [x] **A2. Old `museport` repository: make it private or archive it** (author). Done: it was
  renamed `museport-dev` and made private, and this repository took the name `museport`. Its history
  holds two removed items derived from the toolkit's example.
- [ ] **A3. Rebuild the environment** in any new session from `docs/NEXT-SESSION.md`:
  toolchain, oneMath, llm.c clone, toolkit at the commit pinned in `pyproject.toml`.

## B. Live product (the submission requires one)

- [x] **B1. `museport report`:** generate one static site from `evidence/`:
  - **Overview:** the results table, costs, limits.
  - **One page per item:**
    - Muse's port with a diff view against the CUDA harness;
    - the hidden check, the hidden grade and the variant check;
    - the GPU results;
    - a turn-by-turn replay from `trace.jsonl`: model text, tool calls, results.
  - **Plan page:** Muse's plan and its grounding.

  *Cost: $0. Done when the pages build from the committed evidence with no hand-written
  numbers, and tests cover the generator.*
- [ ] **B2. Publish it** (an artifact page or GitHub Pages) and link it from the README and
  the submission. *Cost: $0.* *(Oct 28: published as a private claude.ai artifact, version 13,
  built from main. Still to do: the author shares it, and the README and submission link it.)*

## C. Finish and harden the current results (cheap; removes caveats)

- [x] **C1. Rerun `t2-softmax-forward`.** It is correct but left 45 helpers. Give a goal that
  says to clear c2s output as soon as the port compiles. Report it beside the first run, not
  replacing it. *About $0.60.*
  *Done Oct 27: rerun fails verify (0 helpers); see `evidence/port-llmc-reruns`.*
- [x] **C2. Rerun `t2-layernorm-backward`.** It copied one implementation into 8 of its 9
  variants. The goal should require each variant to keep llm.c's algorithm (atomics, shared
  memory, warp reductions); the variant check must pass. *About $0.60.*
  *Done Oct 27: round 2 is correct, 0 helpers, a distinct kernel per variant; see `evidence/port-llmc-reruns-2`.*
- [x] **C3. Rerun `t1-fused-residual-forward`.** It is correct but left helpers. *About
  $0.40.*
  *Done Oct 27: round 2 is correct, 45 helpers left; see `evidence/port-llmc-reruns-2`.*
- [ ] **C4. Attention kernels:**
  - write specs for `attention_forward` and `attention_backward` (fp32 variants; batched
    gemms to oneMath; the cuDNN variant noted as out of scope);
  - prove the checks first, then port.

  *About $1-2.*
- [ ] **C5. bf16:**
  - add a bf16 build to the harness: `sycl::ext::oneapi::bfloat16` in the port, and llm.c's
    own bf16 tolerances (1e-2 to 5e-2);
  - port the Tier 1 files in bf16.

  bf16 is llm.c's default precision. *About $2. Done when the checks are proven first (c2s or
  native CUDA on a GPU).*
- [ ] **C6. More runs for robustness:** 2 more runs of each Tier 1 item, so each result has a
  success rate, not one sample. *About $3.*
  *Oct 27: one more run done, with the loop fixes (`evidence/port-llmc-tier1-run2`, $1.01);
  repeats with the same setup are still needed for a success rate.*

## D. Vendors and performance

- [x] **D1. AMD GPUs.** *Done Oct 27 for Tier 1: 10/10 pass on an MI350X against the
  reference, $1.42 (`evidence/port-llmc-tier1-run2/gpu/AMD.md`). There is no native CUDA
  comparison on AMD: llm.c's CUDA doesn't hipify into something that builds.*
  - The Runpod stock check is scheduled (`--provider runpod --vendor amd`).
  - Fallback: DigitalOcean MI300X/MI350X from the author's laptop, because this cloud
    session cannot reach a bare-IP machine. Run `oneapi-agent remote-test <task>
    --provider digitalocean --vendor amd --max-usd 1` on each passing port.

  *About $0.12-0.20 per port, about $2 for all. Done when the README and video can say
  NVIDIA and AMD.*
- [ ] **D2. oneMath on the GPU** *(Oct 27-28: done for GPT-2 stage 3 through the toolkit's task `gpu_setup`, which builds oneMath with cuBLAS on NVIDIA and with rocBLAS on AMD; the 3 Tier 3 ports that call oneMath still to run)* for the 3 Tier 3 ports that call it: build oneMath with its
  cuBLAS backend on the NVIDIA host. This means a GPU-bundle option in the toolkit or a
  prebuilt tarball, logged as a toolkit change. *GPU minutes, about $0.50, plus build time.*
- [ ] **D3. Performance pass (optional).**
  - Benchmark each passing port against llm.c's CUDA on the same GPU (`remote-test` already
    records both).
  - Then let Muse optimise one slow kernel under the hidden check (correctness must still
    pass).

  *About $1-2.*

- [x] **D4. GPU runs from the cloud session, not the author's laptop.** *Done Oct 28:
  `remote-test --public-tls` (toolkit #25). softmax_forward on an MI350X passed for $0.143, the
  first DigitalOcean run made from the cloud.* DigitalOcean (and
  Vultr, Vast) droplets are reached on a bare IP with a self-signed, pinned certificate, which
  the session's egress proxy cannot pass (it re-checks TLS and does not support pinning).
  - Fix: give each droplet a hostname (an `sslip.io` name for its IP) and a Let's Encrypt
    certificate at boot; the runner then connects to an ordinary HTTPS host, no pinning.
  - *Earlier on Oct 28 this was blocked by the session's safety checks; the author changed the
    session's permission mode, and it was built and tested the same day.*
- [ ] **D5. Intel GPUs: the third vendor.** oneAPI's native target. Intel Tiber AI Cloud has
  Data Center GPU Max and Arc machines (often free for developers). The runner already knows
  Level Zero (`level_zero:gpu`). Run the Tier 1-3 ports and GPT-2 stage 3. *Needs the author
  to sign up; cost depends on the tier.*
- [x] **D6. AMD for the rest.** *Done Oct 28, from the cloud: the 5 Tier 2-3 ports and the
  softmax_forward fix pass on an MI350X ($0.72); GPT-2 stage 3 trains with oneMath on rocBLAS,
  10 losses within 0.0000246 of PyTorch's (five runs, $0.65). AMD now matches NVIDIA: 15 of 15
  (`evidence/gpt2-stage3/gpu/AMD.md`).* The 6 Tier 2-3 ports without oneMath calls (4 Tier 2, the
  `softmax_forward` fix, `matmul_backward_bias`), and GPT-2 stage 3 with a `gpu_setup` that
  builds oneMath's rocBLAS backend. *About $2; from the laptop unless D4 lands.*

## E. Measure what the method is worth (Use of Meta AI tooling, Technical depth)

- [ ] **E1. Comparison: Muse with the hidden checks against Muse without them.** *(Oct 28: one run each, Tier 1 and GPT-2 stage 3, `evidence/port-llmc-tier1-baseline`, `evidence/gpt2-stage3-baseline`; 3 runs per item per arm still to do.)* Use the
  toolkit's `--baseline` arm on about 8 kernels, 3 runs per item per arm. Grade every final
  port with the same hidden evaluate.

  *About $8-15 (needs a new cap). Done when there is a committed table of correct ports per
  arm, with run-to-run spread.*
- [ ] **E2. Plan quality.**
  - Compare Muse's plan with the mechanical baseline plan (`--dry-run`) on grounding,
    ordering and risks found.
  - Report the plan's predicted tiers against what actually happened in the ports.

  *$0 (data already committed).*
- [x] **E3. Muse as reviewer (multi-agent).** *(Oct 28: 20 ports, $0.59, `evidence/review-muse`.)* A second Muse call reviews each finished port
  for variant fidelity and idiomatic SYCL. Compare its verdicts with the variant check and
  the hidden grade. *About $1.*

## F. Open weights (bonus: Best Build on Meta Open Weight Models)

- [ ] **F1. Get access to Muse Spark Open:** a Baseten key or credits (the participant guide
  lists $100 by request), or Modal. *Author.*
- [ ] **F2. Same pipeline, open weights.**
  - `museport plan --provider baseten`.
  - `museport port --provider baseten` on the 19 items (the toolkit already supports
    Baseten).
  - Publish a head-to-head table against `muse-spark-1.3`: correct ports, turns, cost.

  *Cost on Baseten credits. Done when both models' evidence is committed side by side.*

## G. Headline: port a real program, not just kernel tests (Impact, Technical depth)

- [x] **G1. Survey `train_gpt2_fp32.cu` and `test_gpt2_fp32.cu`.** Map which of the 19 ported
  kernels it uses, its cuBLAS calls, and its data files: the GPT-2 124M weights and debug
  state from llm.c's download script, about 500 MB. *$0.* *Done Oct 27:
  [`GPT2-PLAN.md`](GPT2-PLAN.md).*
- [ ] **G2. Hidden check: llm.c's own `test_gpt2`.** It compares logits, loss and gradients
  with a PyTorch reference over 10 steps. Prove it passes for llm.c's CUDA on a GPU first.
  *About $0.10.*
- [ ] **G3. Muse ports the fp32 training program to SYCL** *(Oct 27: stage 1, the forward pass,
  passes on the CPU device for $0.74; stage 2, the backward pass, for $1.21; stage 3, 10
  AdamW training steps against llm.c's expected losses, for $1.15; see `evidence/gpt2-stage1`
  to `evidence/gpt2-stage3`. Stage 3 also passes on an NVIDIA RTX 4000 Ada against PyTorch and
  llm.c's CUDA, with oneMath built on the machine with cuBLAS: `evidence/gpt2-stage3/gpu`.)*, reusing its kernel ports, in
  stages (forward, backward, optimizer). It passes `test_gpt2` on the CPU device, then on
  NVIDIA, and on AMD if D1 works. *About $3-8 plus GPU minutes; needs a cap. Done when the SYCL
  build matches PyTorch within llm.c's tolerances, with the evidence committed.*
- [ ] **G4. If time allows: train GPT-2 124M for a few hundred steps** on a GPU with the SYCL
  port, and compare its loss curve with the CUDA build. *About $2-5 of GPU time.*

## H. Generalisation (Impact)

- [ ] **H1. A second CUDA project.** *(Replaced by J4.)* Run `scan` and `plan` on a different repository (backup
  in `PLAN.md`: thundersvm; or a smaller one with CPU references). Port 3-5 kernels with new
  specs. *About $1-3. Done when the plan is grounded and the ports pass hidden checks.*
- [ ] **H2. Make specs easier to write:** *(Replaced by J2.)* a small spec format in TOML or JSON, so a new
  project needs no Python. *$0.*

## J. Point at a CUDA repo, get a oneAPI repo (Impact; added Oct 28)

Today `scan` and `plan` take any CUDA repo, but each ported file needs a hand-written spec
(variants, sizes, tolerance, a CPU reference), and the result is one standalone program per
file. This section removes both limits.

- [x] **J1. Automatic hidden checks by differential testing.** Build the original CUDA on an
  NVIDIA GPU, run it and record its outputs; those recordings become the hidden reference.
  Use the repo's own tests where it has them. No hand-written CPU reference. *Runpod time.*
  **Done Oct 28** (`evidence/diff-llmc-tier1`): all 10 Tier 1 kernels recorded for $0.034, with
  bit-identical repeat runs. The recordings match llm.c's CPU references 10/10, Muse's ports
  pass against them 10/10, and a planted dropped-digit bug fails. Not yet done: using the
  repo's own tests.
- [x] **J2. Automatic harness.** Generate each file's harness (inputs, sizes, the contract
  `main`) from the CUDA source and its own `main`/tests, by tool or by Muse, proven like
  today: the original CUDA must pass it. Replaces H2. **Done Oct 28** (`evidence/spec-llmc`):
  Muse wrote the spec for 17 llm.c files, and 16 are proven (CPU proof, or J1's GPU proof when
  SYCLomatic fails). After the fp32 rerun (`evidence/spec-llmc-fp32`), all 16 are as strict as
  llm.c's own check. The first run's 3 loose specs had taken the bf16 tolerance from
  `#if ENABLE_BF16`. `trimat_forward` is not proven.
- [ ] **J3. A reassembled repo: `llm.c-sycl`.** Every ported file, the build converted
  (`Makefile` to `icpx -fsycl`), one project that builds, passes llm.c's own `test_gpt2`
  and trains, on CPU, NVIDIA and AMD. *About $5-10 of Muse.*
- [ ] **J4. A second, unseen CUDA repo, end to end** with J1-J3: scan, plan, port, check,
  reassemble. Replaces H1.

## K. Beyond the GPT-2 port (added Oct 28)

- [ ] **K1. Real training, compared across vendors.** GPT-2 124M for a few hundred steps on
  real text: llm.c's CUDA on NVIDIA, the SYCL port on NVIDIA and AMD (and Intel with D5);
  overlay the loss curves. Extends G4. *A few dollars of GPU.*
- [ ] **K2. llm.c's production trainer `train_gpt2.cu`:** bf16 mixed precision, cuBLASLt,
  fused kernels, multi-GPU. Much harder; about two days.
- [ ] **K3. Llama 3 (`train_llama3.cu`).** The biggest headline and the highest risk.
- [ ] **K4. Speed.** Muse optimises the simple-kernel ports under the hidden check, measured
  against llm.c's CUDA on the same GPU. Extends D3.

## I. Demo, submission text, outreach (Demo 25%, Most Viral, Community Vote)

- [ ] **I1. Video script, shot by shot (3 minutes).**
  - Opening line on pre-existing vs new (draft in `docs/SUBMISSION.md`).
  - Plan.
  - A hidden check failing on a planted bug, then passing.
  - A Muse port with its replay (B1).
  - The variant check catching a copied port.
  - GPU results.
  - The headline result (G or E).
  - Limits.
- [ ] **I2. Demo commands that run in seconds at $0**, plus one optional real run on camera
  (about $0.20).
- [ ] **I3. Record the video** (author).
- [ ] **I4. Submission text for Cerebral Valley:**
  - summary;
  - how Meta's models are used (planning, porting, the reviewer if E3 is done, open weights
    if F is done);
  - technical depth;
  - limits.

  Every number links to evidence.
- [ ] **I5. Posts on X:** a short thread with the results table, a replay clip and the repo
  link. *Author posts.*
- [ ] **I6. Final check before submitting:**
  - no claim without evidence;
  - the README matches the evidence (re-run the number check);
  - the repo contains only hackathon work (re-run the provenance scan);
  - links resolve and CI is green.

---

## Suggested schedule (revised Oct 28, night)

Done by Oct 28: A1-A2, B1, C1-C3, D1, D2 (GPT-2, NVIDIA and AMD), D4, D6, E3, G1, G3 on CPU,
NVIDIA and AMD, J1, J2 (with the fp32 rerun), the post-fix reruns, E1 (one run each), the
demo clips and the submission draft. J1 and J2 were planned for Oct 29-31, so the schedule is a
day ahead.

| Day | Items |
|---|---|
| Oct 29-30 | J3 (`llm.c-sycl`), after its scope and cost are agreed (see `docs/J3-PLAN.md`) |
| Oct 31 | K1 (loss curves on CPU, NVIDIA and AMD); D5 if Intel access arrives |
| Nov 1 | J4 (a second CUDA repo); C6 or E1 runs if the budget allows |
| Nov 2 | F2 if Baseten access arrives; refresh SUBMISSION and DEMO with every new result |
| Nov 3 | I1-I5: script, record, submission text, posts |
| Nov 4 | I6, submit by midday |

**Decisions needed from the author:**
- caps for J3, J4 and K1: comparison runs have about $7.6-7.8 left of $30, and J3 alone is
  estimated at several dollars; GPUs have about $5.90 left of $10; the GPT-2 cap for K1 is $7,
  with $4.45 used;
- Intel Tiber AI Cloud sign-up (D5);
- Baseten or Modal access (F1);
- sharing the report (B2).
