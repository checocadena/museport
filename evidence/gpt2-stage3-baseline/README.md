# GPT-2 stage 3 (training) without feedback (plan item E1, Oct 28)

The same task as [stage 3](../gpt2-stage3/README.md): 10 AdamW training steps of llm.c's GPT-2,
each step's loss checked against PyTorch's within 1e-2. The task directory is the same one,
`../gpt2-stage3/task`.

The one change is `museport loop --baseline`, the toolkit's arm without feedback:

- Muse has no `verify`, `scan_helpers` or `benchmark`, so it never runs its port.
- The run ends when Muse calls `finish`, and the hidden grade checks the final sources after.

`museport loop ../gpt2-stage3/task/task.json --baseline --max-iterations 40 --max-cost 3`
([`run1/`](run1/)).

## Result: the port passes; $1.3387

| | With feedback ([stage 3](../gpt2-stage3/README.md)) | Without feedback |
|---|---|---|
| End of the run | success at turn 29 | Muse called `finish` at turn 29 |
| Hidden grade | **PASS**, worst error 0.000154 | **PASS**, worst error 0.00149 (atol 0.01); 0 helpers |
| First version | compiled, but all 10 losses wrong: a matmul transpose bug | compiled after one fix and was already correct |
| Cost | $1.1489 | $1.3387 |

## How the run went

1. **Turns 1-10:** `migrate`, then Muse read all of llm.c's `train_gpt2_fp32.cu` (1,755 lines)
   in ranges, plus `llmc/utils.h` and the start of SYCLomatic's output.
2. **Turn 11:** it wrote the whole port in one `write_file` (41,164 characters).
3. **Turns 12-17:** 10 compile errors, from a namespace alias (`tm`) clashing with the C
   library. Muse deleted `migrated/`, renamed the alias and rewrote the file.
4. **Turns 18-21:** it compiled, then Muse tightened numerics on its own: the exact AdamW lerp,
   `powf` bias corrections, `rsqrt` and explicit causal zeroing. It compiled again.
5. **Turns 22-29:** Muse reread parts of llm.c's CUDA against its port, recompiled and called
   `finish`.

## The grade, and how it was rerun

The run's own after-run grade ([`run1/outcome.json`](run1/outcome.json)) reports
`verify_status: error` and "input(s) not found".

- **Cause:** the checkout it ran in had no links to the 1 GB data files. They are git-ignored,
  and they had been made only in the worktree that wrote stage 3's task.
- **Muse was not affected:** without `verify` it never ran the program.

The grade was rerun at $0 with the same toolkit steps:

1. Restore the data links.
2. Copy the run's final work directory.
3. Build the hidden reference.
4. Compile with Muse's last compile arguments.
5. Run the toolkit's `verify` and the helper scan.

The script is [`regrade.py`](regrade.py) and its output is [`regrade.json`](regrade.json). The
sources it graded have the run's final source hash, `ce475dc1cee2b619`. Rerunning it gave the
same result.

## What this shows

- **This time Muse did not need the check.** Its first version was right.
- **With feedback, its first version was wrong, and only the check caught it.** The final port
  there is closer to PyTorch: 0.000154 against 0.00149.
- **One run each way** says nothing about how often a version written without feedback is
  right; repeated runs would.
