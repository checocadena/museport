"""Context compaction for long loop runs: old tool results and written files are shortened,
the last turns are kept whole, and compacting twice changes nothing more."""

import json

from museport.loop import ELIDED, CompactingModel


class _Inner:
    name = "fake"

    def __init__(self) -> None:
        self.seen: list[int] = []

    def create(self, *, system, messages, tools):
        self.seen.append(len(json.dumps(messages)))
        return "resp"


def _turns(n: int) -> list[dict]:
    msgs: list[dict] = [{"role": "user", "content": "goal"}]
    for i in range(n):
        msgs.append({"role": "assistant", "content": [
            {"type": "tool_use", "id": f"u{i}", "name": "write_file",
             "input": {"path": f"f{i}.cpp", "content": "x" * 5000}}]})
        msgs.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": f"u{i}", "content": "y" * 5000},
            {"type": "text", "text": "state note"}]})
    return msgs


def test_old_turns_are_shortened_and_recent_ones_kept() -> None:
    inner = _Inner()
    m = CompactingModel(inner, trigger_chars=100_000, keep_turns=5, head=100)
    msgs = _turns(30)
    assert m.create(system="s", messages=msgs, tools=[]) == "resp"
    assert m.compactions == 1 and inner.seen[0] < 100_000
    old_use, old_res = msgs[1]["content"][0], msgs[2]["content"][0]
    assert old_use["input"]["content"].endswith(ELIDED) and old_use["input"]["path"] == "f0.cpp"
    assert old_res["content"].endswith(ELIDED) and old_res["content"].startswith("y" * 100)
    assert msgs[-1]["content"][0]["content"] == "y" * 5000  # the last turns stay whole
    assert msgs[0]["content"] == "goal"


def test_below_the_trigger_nothing_changes() -> None:
    m = CompactingModel(_Inner(), trigger_chars=10**9, keep_turns=5, head=100)
    msgs = _turns(10)
    before = json.dumps(msgs)
    m.create(system="s", messages=msgs, tools=[])
    assert json.dumps(msgs) == before and m.compactions == 0


def test_compacting_again_is_stable() -> None:
    m = CompactingModel(_Inner(), trigger_chars=1, keep_turns=5, head=100)
    msgs = _turns(30)
    m.compact(msgs)
    once = json.dumps(msgs)
    m.compact(msgs)
    assert json.dumps(msgs) == once


def test_shell_failures_do_not_count_as_stuck() -> None:
    import oneapi_agent.loop as tk
    from oneapi_agent.result import ToolResult

    from museport.loop import _shell_not_stuck

    run_fail = ToolResult("run", "fail", "exit 127: /tmp/dbg6: No such file or directory", {})
    verify_fail = ToolResult("verify", "fail", "FAIL: 0/1 case(s) match", {})
    assert tk.failure_signature(run_fail) is not None
    with _shell_not_stuck():
        assert tk.failure_signature(run_fail) is None
        assert tk.failure_signature(verify_fail) is not None
    assert tk.failure_signature(run_fail) is not None
