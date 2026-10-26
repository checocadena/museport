# Changes made to oneapi-agent-toolkit during the hackathon

The toolkit's state before the hackathon is the tag `pre_hackathon`. Every change made to it from
Oct 27 for Muse Port is listed here, with its pull request.

| Date | Change | PR |
|---|---|---|
| 2026-10-26 | Meta API details aligned with Meta's docs (`MODEL_API_KEY`, `max_completion_tokens`, `reasoning_effort`, built-in Muse prices, `Retry-After`); `ONEAPI_AGENT_AUTH_FROM_NETWORK=1` for a key injected by a proxy; `--provider baseten` for Muse Spark Open | [#19](https://github.com/checocadena/oneapi-agent-toolkit/pull/19) |
| 2026-10-26 | `gpu-test`: optional `native_flags` task field, appended to the native `nvcc` build, so CUDA that calls cuBLAS/cuBLASLt links (`-lcublas -lcublasLt`); found on llm.c's Tier 3 kernels. Merged as `7138b09`; museport pins that commit | [#20](https://github.com/checocadena/oneapi-agent-toolkit/pull/20) |
