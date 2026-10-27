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
- **Comparison runs:** $5.51 of $20.
- **GPUs:** about $1.13.

Items 1, 2 and 3 below would go past the remaining $20 comparison budget together.
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
  toolchain, oneMath, llm.c clone, toolkit pinned at `7138b09`.

## B. Live product (the submission requires one)

- [ ] **B1. `museport report`:** generate one static site from `evidence/`:
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
  the submission. *Cost: $0.*

## C. Finish and harden the current results (cheap; removes caveats)

- [ ] **C1. Rerun `t2-softmax-forward`.** It is correct but left 45 helpers. Give a goal that
  says to clear c2s output as soon as the port compiles. Report it beside the first run, not
  replacing it. *About $0.60.*
- [ ] **C2. Rerun `t2-layernorm-backward`.** It copied one implementation into 8 of its 9
  variants. The goal should require each variant to keep llm.c's algorithm (atomics, shared
  memory, warp reductions); the variant check must pass. *About $0.60.*
- [ ] **C3. Rerun `t1-fused-residual-forward`.** It is correct but left helpers. *About
  $0.40.*
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

## D. Vendors and performance

- [ ] **D1. AMD GPUs.**
  - The Runpod stock check is scheduled (`--provider runpod --vendor amd`).
  - Fallback: DigitalOcean MI300X/MI350X from the author's laptop, because this cloud
    session cannot reach a bare-IP machine. Run `oneapi-agent remote-test <task>
    --provider digitalocean --vendor amd --max-usd 1` on each passing port.

  *About $0.12-0.20 per port, about $2 for all. Done when the README and video can say
  NVIDIA and AMD.*
- [ ] **D2. oneMath on the GPU** for the 3 Tier 3 ports that call it: build oneMath with its
  cuBLAS backend on the NVIDIA host. This means a GPU-bundle option in the toolkit or a
  prebuilt tarball, logged as a toolkit change. *GPU minutes, about $0.50, plus build time.*
- [ ] **D3. Performance pass (optional).**
  - Benchmark each passing port against llm.c's CUDA on the same GPU (`remote-test` already
    records both).
  - Then let Muse optimise one slow kernel under the hidden check (correctness must still
    pass).

  *About $1-2.*

## E. Measure what the method is worth (Use of Meta AI tooling, Technical depth)

- [ ] **E1. Comparison: Muse with the hidden checks against Muse without them.** Use the
  toolkit's `--baseline` arm on about 8 kernels, 3 runs per item per arm. Grade every final
  port with the same hidden evaluate.

  *About $8-15 (needs a new cap). Done when there is a committed table of correct ports per
  arm, with run-to-run spread.*
- [ ] **E2. Plan quality.**
  - Compare Muse's plan with the mechanical baseline plan (`--dry-run`) on grounding,
    ordering and risks found.
  - Report the plan's predicted tiers against what actually happened in the ports.

  *$0 (data already committed).*
- [ ] **E3. Muse as reviewer (multi-agent).** A second Muse call reviews each finished port
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

- [ ] **G1. Survey `train_gpt2_fp32.cu` and `test_gpt2_fp32.cu`.** Map which of the 19 ported
  kernels it uses, its cuBLAS calls, and its data files: the GPT-2 124M weights and debug
  state from llm.c's download script, about 500 MB. *$0.*
- [ ] **G2. Hidden check: llm.c's own `test_gpt2`.** It compares logits, loss and gradients
  with a PyTorch reference over 10 steps. Prove it passes for llm.c's CUDA on a GPU first.
  *About $0.10.*
- [ ] **G3. Muse ports the fp32 training program to SYCL**, reusing its kernel ports, in
  stages (forward, backward, optimizer). It passes `test_gpt2` on the CPU device, then on
  NVIDIA, and on AMD if D1 works. *About $3-8 plus GPU minutes; needs a cap. Done when the SYCL
  build matches PyTorch within llm.c's tolerances, with the evidence committed.*
- [ ] **G4. If time allows: train GPT-2 124M for a few hundred steps** on a GPU with the SYCL
  port, and compare its loss curve with the CUDA build. *About $2-5 of GPU time.*

## H. Generalisation (Impact)

- [ ] **H1. A second CUDA project.** Run `scan` and `plan` on a different repository (backup
  in `PLAN.md`: thundersvm; or a smaller one with CPU references). Port 3-5 kernels with new
  specs. *About $1-3. Done when the plan is grounded and the ports pass hidden checks.*
- [ ] **H2. Make specs easier to write:** a small spec format in TOML or JSON, so a new
  project needs no Python. *$0.*

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

## Suggested schedule

| Day | Items |
|---|---|
| Oct 27 | A1-A3, B1, C1-C3 |
| Oct 28 | B2, E2, D1 (AMD), I1 draft |
| Oct 29 | E1 (comparison), F1 (request access) |
| Oct 30 | G1-G2, C4 (attention) |
| Oct 31 - Nov 1 | G3 (GPT-2 port), F2 if access arrives |
| Nov 2 | G4 or C5 (bf16), D2/D3 if time; refresh the report pages |
| Nov 3 | I1-I5: script, record, submission text, posts |
| Nov 4 | I6, submit by midday |

**Decisions needed from the author:**
- caps for E1, G3 and G4;
- Baseten or Modal access (F1);
- A2 (old repository);
- which of G or E is the headline if time runs short.
