"""museport loop: streamed tool calls reach the toolkit's loop; the USD cap is the caller's."""

from __future__ import annotations

import json
from pathlib import Path

from oneapi_agent.models import from_openai_response

from museport.loop import build
from museport.stream import assemble_stream, stream_transport

TASK = Path(__file__).parent / "fixtures" / "loop_task" / "task.json"


def _sse(*chunks):
    return [f"data: {c if isinstance(c, str) else json.dumps(c)}\n".encode() for c in chunks]


def test_tool_call_fragments_are_reassembled():
    # The shape Meta streams (recorded 2026-10-26): id and name first, then argument pieces.
    resp = assemble_stream(_sse(
        {"model": "muse-spark-1.3",
         "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Writing."}}]},
        {"choices": [{"index": 0, "delta": {"tool_calls": [{
            "index": 0, "id": "call_1", "type": "function",
            "function": {"name": "write_file", "arguments": ""}}]}}]},
        {"choices": [{"index": 0, "delta": {"tool_calls": [{
            "index": 0, "function": {"arguments": '{"content":"hello world"'}}]}}]},
        {"choices": [{"index": 0, "delta": {"tool_calls": [{
            "index": 0, "function": {"arguments": ',"path":"a.txt"}'}}]}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
         "usage": {"prompt_tokens": 560, "completion_tokens": 162,
                   "prompt_tokens_details": {"cached_tokens": 113}}},
        "[DONE]"))
    r = from_openai_response(resp)  # the toolkit's own parser
    assert r.stop_reason == "tool_use"
    assert r.content == [
        {"type": "text", "text": "Writing."},
        {"type": "tool_use", "id": "call_1", "name": "write_file",
         "input": {"content": "hello world", "path": "a.txt"}}]
    assert r.usage["input_tokens"] == 447 and r.usage["cache_read_input_tokens"] == 113


def test_missing_usage_is_charged_as_worst_case(monkeypatch):
    from museport import stream as S

    class R(list):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(S.urllib.request, "urlopen", lambda req, timeout: R(_sse(
        {"choices": [{"delta": {"content": "x"}, "finish_reason": "stop"}]}, "[DONE]")))
    body = {"model": "m", "max_completion_tokens": 16000,
            "messages": [{"role": "user", "content": "a" * 4000}]}
    resp = stream_transport(fill_missing_usage=True)("https://x/v1/chat/completions", {}, body)
    assert resp["usage_estimated"] and resp["usage"]["completion_tokens"] == 16000
    assert resp["usage"]["prompt_tokens"] >= 1000


def test_an_empty_answer_is_marked_for_the_loop(monkeypatch):
    # Seen 2026-10-27: reasoning used all 16000 output tokens, the stream ended with no text and
    # no tool call, and sending that turn back gave HTTP 400 ("must have content or tool_calls").
    from museport import stream as S

    class R(list):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(S.urllib.request, "urlopen", lambda req, timeout: R(_sse(
        {"choices": [{"delta": {}, "finish_reason": "stop"}]}, "[DONE]")))
    body = {"model": "m", "messages": [{"role": "user", "content": "a"}]}
    plain = stream_transport()("https://x/v1/chat/completions", {}, body)
    assert plain["choices"][0]["message"]["content"] == ""  # the plan reports it as empty
    marked = stream_transport(mark_empty=True)("https://x/v1/chat/completions", {}, body)
    assert marked["choices"][0]["message"]["content"] == S.EMPTY_TURN
    r = from_openai_response(marked)
    assert r.content == [{"type": "text", "text": S.EMPTY_TURN}]


def test_build_sets_the_cap_and_the_streaming_client():
    sentinel = object()
    cfg, client = build(TASK, model="muse-spark-1.3", max_iterations=3, max_cost_usd=0.75,
                        transport=sentinel)  # type: ignore[arg-type]
    assert (cfg.max_iterations, cfg.max_cost_usd, cfg.provider) == (3, 0.75, "meta")
    assert client.transport is sentinel and client.max_tokens == 16_000
    _, big = build(TASK, model="muse-spark-1.3", max_output_tokens=32_000,
                   transport=sentinel)  # type: ignore[arg-type]
    assert big.max_tokens == 32_000
    assert cfg.max_total_tokens == 600_000  # the toolkit's default stays unless raised
    raised, _ = build(TASK, model="muse-spark-1.3", max_total_tokens=2_000_000,
                      transport=sentinel)  # type: ignore[arg-type]
    assert raised.max_total_tokens == 2_000_000


def test_safe_toolbox_turns_a_cut_off_call_into_an_error_result(tmp_path):
    from museport.loop import SafeToolbox

    cfg, _ = build(TASK, model="muse-spark-1.3", transport=object())  # type: ignore[arg-type]
    r = SafeToolbox(cfg, tmp_path)("write_file", {"_unparsed_arguments": '{"content": "x'})
    assert r.status == "error" and "missing argument" in r.summary
