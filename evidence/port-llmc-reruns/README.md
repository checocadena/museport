# Reruns with a stricter goal (plan C1-C3), Oct 27

Three ports from the first runs had open issues:
- `t1-fused-residual-forward` and `t2-softmax-forward` were correct but left SYCLomatic helpers;
- `t2-layernorm-backward` copied one implementation into 8 of its 9 variants.

Each was run again with `museport port --strict-goal`. This adds to the goal: clear SYCLomatic's
output as soon as the port compiles, run `scan_helpers` on `.`, and keep each variant's own
algorithm. The first runs stay on record in their tier directories. These reruns do not replace
them.

**Setup:**
- Model `muse-spark-1.3`, up to 14 turns.
- Caps: $0.70 per item, $2.00 for the batch.
- The command and its output are in [`run.log`](run.log).

## Result: none of the three improved ($1.3971)

| Item | First run | This rerun |
|---|---|---|
| `t1-fused-residual-forward` | correct, 53 helpers left | **error** at turn 6, $0.1874: empty answer (see below) |
| `t2-softmax-forward` | correct, 45 helpers left | 0 helpers, but **verify fails** (41,906 of 25.7M values outside tolerance); turn limit reached before Muse ran verify, $0.5534 |
| `t2-layernorm-backward` | correct, 0 helpers, variants copied | **error** at turn 10, $0.6563: empty answer (see below) |

## Two failure modes, measured

### 1. Empty answer, then HTTP 400

Muse spent its whole output budget (16,000 tokens) on reasoning in one turn. It returned no
text and no tool call. The loop sent that turn back as part of the history, and the API refused
it with HTTP 400 ("assistant message must have `content` or `tool_calls`"), which ended the run.

- This happened in 2 of the 3 runs: fused residual at turn 6, layernorm backward at turn 10.
- The API also reported no usage for these turns. They are charged at the worst case (the
  whole request as input plus the full output budget), so the costs above are upper bounds.
- The loop now marks such a turn instead (`stream_transport(mark_empty=True)`), so a run
  continues.

### 2. Large files cut off at the output limit

In both Tier 2 reruns, Muse's first `write_file` of the full port was cut off by the output
limit, and the toolkit reported the missing argument.

- In `t2-softmax-forward`, Muse then wrote the port again and it compiled. It ran out of turns
  after clearing `migrated/`, before it ran verify.
- In `t2-layernorm-backward`, Muse never wrote its port. It wrote placeholder files and emptied
  `migrated/` early, as the stricter goal says. Its SYCLomatic input was 113,224 characters.

The stricter goal did change what Muse did first: it cleared SYCLomatic's output early. It did
not make the ports better within 14 turns and a 16,000-token output budget.
