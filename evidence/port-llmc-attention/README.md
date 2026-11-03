# Muse ports llm.c's attention kernels: `attention_backward.cu` ported, `attention_forward.cu` cut off

`museport port --tier 3` on the two attention items in Muse's
[plan](../plan-llmc-devcuda-stream/README.md), with `muse-spark-1.3`: 20 turns max and a $1
cap per item, $2 for the batch. Each hidden check was [proven first](harness-checks/README.md)
with llm.c's own CUDA on an NVIDIA GPU.

## Results

| Item | Variants | Loop result (turns) | Muse cost | Hidden grade (final sources, CPU) | Kernels + oneMath calls / variants |
|---|---|---|---|---|---|
| `t3-attention-backward` | 8 | max_iterations (20) | $0.9233 | **PASS**, max abs err 5.96e-07 (atol 1e-3), 0 helpers | 13 + 1 / 8 |
| `t3-attention-forward` | 6 | error at turn 5: `Connection refused` from Meta's API | $0.2366 | not graded (no port written) | |

Muse spend: **$1.1599** ([SUMMARY.md](SUMMARY.md), [summary.json](summary.json),
[run.log](run.log)).

## attention_backward: what Muse did

- **The port** ([port/attention_backward.cpp](t3-attention-backward/port/attention_backward.cpp),
  [report](t3-attention-backward/report.md)) keeps llm.c's 8 variants. llm.c's
  `cublasSgemmStridedBatched` calls become oneMath's
  `oneapi::math::blas::column_major::gemm_batch` (fp32, strided), through one helper.
- **It fixed both problems the hidden-check proof found in llm.c's kernels:**
  - variant 6 writes past the end of a row when T is smaller than the block size; the port
    guards the write (`if (t3 < T)`);
  - variant 8 reuses a shared array across rows with no barrier; the port uses a whole
    work-group reduction (`sycl::reduce_over_group`) instead of a warp reduction into
    shared memory, so there is nothing to race on.
- **Timeline:** a first full port at turn 11, verify PASS at turn 14, then it deleted
  SYCLomatic's leftover files so the helper scan passed (turn 17). At turn 20, its last, it
  edited variant 6's kernel and never verified that version, so the loop's strict rule
  (verify and scan on the same sources) was not met. The hidden grade after the run checks
  the final sources, that edit included, and passes.

## attention_forward: not ported yet

The run stopped at turn 5, while Muse was still reading the migrated code: the connection to
`api.meta.ai` was refused, at the time this session's container restarted. It has not been
retried; a retry needs at most $1 of Muse budget.

## Not run yet

- **GPUs:** the backward port has been checked on the CPU device only.

## Checks

| | |
|---|---|
| Task folders before and after | unchanged ([task-before.sha256](task-before.sha256), [task-after.sha256](task-after.sha256)) |
| Secrets | none (`museport.redact.secret_hits`) |
| Muse | $1.1599 |
| GPU (hidden-check proofs) | about $0.02 |
