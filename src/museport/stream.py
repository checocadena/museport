"""Streaming chat completions for the Meta Model API.

A non-streamed request whose answer takes more than about 30 s is cut off with HTTP 502
"upstream request failed" on the way to the Meta API (measured: 502 at 30.2 s, while the same
request streamed ran 51 s; see evidence/plan-llmc-devcuda). Everything museport sends to Muse,
the plan request and the agent loop's turns, therefore goes through ``stream_transport``.
"""

from __future__ import annotations

import http.client
import json
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from typing import Any

from oneapi_agent.models import Transport

CHARS_PER_TOKEN = 4
EMPTY_TURN = "(no answer: this turn's output budget was used up before any text or tool call)"
LOST_TURN = ("(no answer: the connection was lost while this answer was streaming; it was "
             "discarded)")


def assemble_stream(lines: Iterable[bytes]) -> dict[str, Any]:
    """Turn server-sent chat completion chunks into one non-streamed response: text joined,
    tool calls rebuilt from their fragments (first one has the id and name, later ones append
    to ``arguments``).

    ``usage`` is present only if the server sent it (Meta sends it on the last chunk, but not
    always: a stream cut off by the output budget ended with ``finish_reason`` "stop" and no
    usage in our tests), so callers must not read a missing ``usage`` as zero cost."""
    parts: list[str] = []
    calls: dict[int, dict[str, Any]] = {}
    out: dict[str, Any] = {}
    finish: str | None = None
    for raw in lines:
        line = raw.decode(errors="replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        chunk = json.loads(data)
        if chunk.get("error"):
            raise RuntimeError(f"stream error: {json.dumps(chunk['error'])[:500]}")
        out.setdefault("model", chunk.get("model"))
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            parts.append(delta.get("content") or "")
            for tc in delta.get("tool_calls") or []:
                call = calls.setdefault(tc.get("index", len(calls)), {
                    "id": None, "type": "function", "function": {"name": "", "arguments": ""}})
                call["id"] = tc.get("id") or call["id"]
                fn = tc.get("function") or {}
                call["function"]["name"] += fn.get("name") or ""
                call["function"]["arguments"] += fn.get("arguments") or ""
            finish = choice.get("finish_reason") or finish
        if chunk.get("usage"):
            out["usage"] = chunk["usage"]
    message: dict[str, Any] = {"role": "assistant", "content": "".join(parts)}
    if calls:
        message["tool_calls"] = [calls[i] for i in sorted(calls)]
    out["choices"] = [{"index": 0, "finish_reason": finish, "message": message}]
    return out


def worst_case_usage(body: dict[str, Any]) -> dict[str, int]:
    """Usage to charge when the server reports none: the whole request as uncached input (4
    characters per token) and the full output budget. Over-counts on purpose."""
    chars = len(json.dumps(body.get("messages", []))) + len(json.dumps(body.get("tools", [])))
    pin = chars // CHARS_PER_TOKEN
    pout = int(body.get("max_completion_tokens") or 0)
    return {"prompt_tokens": pin, "completion_tokens": pout, "total_tokens": pin + pout}


def stream_transport(timeout_s: float = 900.0, attempts: int = 3, *,
                     fill_missing_usage: bool = False, mark_empty: bool = False) -> Transport:
    """POST with ``stream: true`` and return the assembled response.

    Only HTTP 429 is retried: any other failure may come after the request was accepted and
    billed, so it is reported, never silently repeated. With ``fill_missing_usage`` a response
    without usage gets ``worst_case_usage`` (and ``usage_estimated``), so a caller that sums
    usage against a USD cap, like the toolkit's agent loop, never counts a turn as free.
    With ``mark_empty`` an answer with neither text nor tool calls (reasoning used the whole
    output budget) gets ``EMPTY_TURN`` as its text: the API refuses such an assistant message
    with HTTP 400 when a loop sends the history back, which would end the run. A connection
    lost mid-stream likewise becomes a ``LOST_TURN`` answer, charged at the worst case, instead
    of ending the run (seen Oct 27: ConnectionResetError at turn 7); it is not resent.
    Headers (the key) are never logged."""

    def post(url: str, headers: dict[str, str], body: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps({**body, "stream": True,
                           "stream_options": {"include_usage": True}}).encode()
        for attempt in range(attempts):
            req = urllib.request.Request(url, data=data, method="POST",
                                         headers={**headers, "Content-Type": "application/json",
                                                  "Accept": "text/event-stream"})
            try:
                with urllib.request.urlopen(req, timeout=timeout_s) as r:
                    resp = assemble_stream(r)
            except (ConnectionError, TimeoutError, http.client.IncompleteRead) as e:
                if not mark_empty:
                    raise RuntimeError(f"{url}: connection lost: {type(e).__name__}: "
                                       f"{e}") from None
                print(f"museport: connection lost mid-answer ({type(e).__name__}); the turn "
                      "is marked and charged at the worst case", file=sys.stderr)
                return {"model": body.get("model"), "usage": worst_case_usage(body),
                        "usage_estimated": True, "connection_lost": True,
                        "choices": [{"index": 0, "finish_reason": "stop",
                                     "message": {"role": "assistant", "content": LOST_TURN}}]}
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:500]
                if e.code == 429 and attempt + 1 < attempts:
                    time.sleep(min(float(e.headers.get("Retry-After") or 2 ** attempt), 60.0))
                    continue
                raise RuntimeError(f"{url}: HTTP {e.code}: {detail}") from None
            except urllib.error.URLError as e:
                raise RuntimeError(f"{url}: {e.reason}") from None
            if fill_missing_usage and not resp.get("usage"):
                resp["usage"] = worst_case_usage(body)
                resp["usage_estimated"] = True
                print("museport: the API reported no usage for this turn; charging the worst "
                      f"case ({resp['usage']['total_tokens']} tokens)", file=sys.stderr)
            msg = resp["choices"][0]["message"]
            if mark_empty and not msg.get("tool_calls") and not msg["content"].strip():
                msg["content"] = EMPTY_TURN
                print("museport: empty answer (output budget used up); marked and sent back",
                      file=sys.stderr)
            return resp
        raise AssertionError("unreachable")

    return post
