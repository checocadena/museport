# First real `museport plan` on llm.c `dev/cuda`: failed (HTTP 502)

- **When:** 2026-10-26, 19:18 UTC
- **Target:** [karpathy/llm.c](https://github.com/karpathy/llm.c) at `f1e2ace`, `dev/cuda` (26 entries; scan: 22 CUDA files)
- **Command:** `ONEAPI_AGENT_AUTH_FROM_NETWORK=1 museport plan <llm.c>/dev/cuda --out evidence/plan-llmc-devcuda`
- **Model:** `muse-spark-1.3`, provider `meta`, `max_completion_tokens` 32000 (default), JSON-schema output
- **Request size:** 485,176 characters (about 121k tokens by the 4 chars/token estimate)
- **Worst-case cost before sending:** $0.288 (all input uncached at $1.25/M, full 32k output at $4.25/M)

## Result

No plan was produced. After 2 min 11 s the API returned
`HTTP 502: upstream request failed` (full traceback in [`run.log`](run.log)). Grounding did
not run because there was no plan to check.

**Correction (found later the same day):** the toolkit's `http_transport`, which this run
used, retries HTTP 502 up to 4 times. So the 2 min 11 s was most likely 4 attempts of about
30 s each plus backoff, not one 2-minute timeout. Whether any attempt was billed is not
known: the error responses carry no usage. If every attempt was billed in full, the upper
bound is 4 × $0.288 = $1.15. Since every attempt failed upstream, it was probably much less,
but this is not confirmed.

## Checks made afterwards (same session, same proxy)

- `GET /v1/models` returned 200 and lists `muse-spark-1.3`.
- A one-line chat completion (`max_completion_tokens` 200) returned 200 in about 1 s:
  15 prompt tokens and 200 completion tokens, of which 198 were reasoning. `content` was
  empty because reasoning used up the budget. Cost about $0.0009.

So the key, the proxy and the model id work. The failure is specific to the long request.

**Cause, measured afterwards:** a non-streamed request whose answer takes more than about 30 s
gets `HTTP 502 upstream request failed` at 30.2 s; the same request streamed completed in
51 s. `museport plan` now streams and does not retry paid requests; the streamed rerun is
in [`../plan-llmc-devcuda-stream`](../plan-llmc-devcuda-stream/README.md); the streaming
client is `src/museport/stream.py`.
