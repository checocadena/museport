# Muse as a reviewer (plan item E3, Oct 28)

A second agent reviews finished ports. Command: `museport review evidence --out evidence/review-muse --total-cap 1.60`.

For each port, a separate `muse-spark-1.3` call sees only two things: the CUDA program and Muse's SYCL port. It sees no test result, no hidden grade and no variant check. It answers in a fixed JSON schema:

- per kernel variant, whether the SYCL kernel keeps the CUDA variant's algorithm (its parallel structure, not only its math), with a note;
- whether several variants share one implementation;
- whether the SYCL is idiomatic;
- whether the port likely computes the same results;
- concerns, and a verdict: accept or reject.

**Reviewed:** the 19 ports of the post-fix runs ([Tier 1 run 2](../port-llmc-tier1-run2/README.md), [Tiers 2-3 run 2](../port-llmc-tier23-run2/README.md)) and the passing [`softmax_forward` rerun](../port-llmc-softmax-rerun-2/README.md), 20 in all. Every answer is in [`summary.json`](summary.json); the log is [`run.log`](run.log).

**Cost:** $0.5858 for 20 reviews (reported usage).

## Verdicts against what the checks found

| | Count |
|---|---|
| Accepted | 13 |
| Rejected | 7 |

**Copied variants.** The variant check flags two ports: `softmax_forward` (Tiers 2-3 run 2) and `layernorm_backward`. The reviewer said "copies one implementation" for exactly those two, and for none of the other 18.

**Correctness.** The reviewer was wrong both times it disagreed with the hidden grade:

- **It missed the one failing port.** `softmax_forward` (Tiers 2-3 run 2) fails the hidden check on 36,435 values. The reviewer judged it likely correct. It rejected the port only for dropping warp operations and for copied variants.
- **It flagged one passing port as likely incorrect, and found a real latent bug.** The fixed `softmax_forward` passes the hidden check. The reviewer flagged its variants 6 and 8:
  - In each work-group, rows past the end of the input run one `group_barrier` and `return`, while the valid rows go on to several more barriers.
  - SYCL requires every work-item of a group to reach the same barriers. A non-uniform barrier is undefined behaviour and can hang or corrupt results on a GPU.
  - Confirmed by reading [`softmax_forward.cpp`](../port-llmc-softmax-rerun-2/t2-softmax-forward/port/softmax_forward.cpp), lines 290-293 and 449-452.
  - The hidden check never triggers it: it runs N = B·T = 64 rows, a multiple of 8 rows per group.

**Fidelity to the CUDA algorithm.** Neither check measures this, and the reviewer judged it. It rejected four ports that are correct, have distinct kernels per variant and pass the variant check, because some variants drop the CUDA variant's parallel structure:

| Port | Variants not kept | Reviewer's reason, shortened |
|---|---|---|
| `fused_residual_forward` (Tier 1) | 3, 4, 5, 6 | warp-per-token kernels replaced by one work-item per token with serial loops |
| `layernorm_forward` | 6 | multi-row blocks, per-row input cache and warp reductions not kept |
| `classifier_fused` | 1 | warp-per-row became block-per-row |
| `matmul_backward_bias` | 7, 8, 9 | shuffles, shared atomics and occupancy-based grids replaced by serial local sums |

One claim was spot-checked: in `fused_residual_forward`, variant 3 loops over every channel in one work-item, and variant 4 launches one work-item per token. The CUDA kernels use a warp per token, so the claim holds.

## What this shows

- **A code review is not a correctness check.** The reviewer passed the port that fails, and the hidden check is what caught it.
- **A code review does catch what the checks cannot see.** It found a barrier bug outside the tested sizes, and simplified variants that compute the right numbers.
- **On copied variants, the reviewer agrees with the mechanical variant check: 2 of 2 found, no false flags.**

The checks and the reviewer are complementary. Neither replaces the other.

## Limits

- One review per port, by the same model that wrote the ports.
- Only one fidelity claim and the barrier claim were checked by hand; the other notes are the reviewer's.
- The barrier bug is shown from the code, not by running the port at a size that triggers it.
