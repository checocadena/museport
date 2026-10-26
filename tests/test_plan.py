"""museport plan: request shape, cost cap, grounding, and the offline dry run."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from museport.cli import main
from museport.plan import (
    PLAN_SCHEMA,
    ScriptedPlanTransport,
    baseline_plan,
    ground,
    make_plan,
)
from museport.scan import scan

MINI = Path(__file__).parent / "fixtures" / "mini"
PRICED = {"ONEAPI_AGENT_AUTH_FROM_NETWORK": "1"}  # muse-spark-1.3 has a built-in price


@pytest.fixture(scope="module")
def report():
    return scan(MINI)


def test_baseline_plan_is_grounded_and_covers_everything(report):
    g = ground(baseline_plan(report), report)
    assert g["grounded"] and g["issues"] == 0
    assert g["coverage"] == {"files": 3, "covered": 3, "planned": 2}  # gemm.cu: no kernels


def test_grounding_catches_invented_files_and_kernels(report):
    plan = baseline_plan(report)
    plan["items"][0]["files"].append("kernels/does_not_exist.cu")
    plan["items"][0]["kernels"].append("imaginary_kernel")
    g = ground(plan, report)
    assert not g["grounded"]
    assert g["unknown_files"] == ["kernels/does_not_exist.cu"]
    assert g["unknown_kernels"] == [{"item": plan["items"][0]["id"],
                                     "kernel": "imaginary_kernel"}]


def test_grounding_catches_gaps(report):
    plan = baseline_plan(report)
    reduce_item = next(it for it in plan["items"] if it["files"] == ["kernels/reduce.cu"])
    reduce_item["risks"] = []          # warp shuffle + cooperative groups not mentioned
    reduce_item["tier"] = 1            # scan says 2
    reduce_item["depends_on"] = ["nope"]
    plan["out_of_scope"] = []          # gemm.cu no longer placed
    plan["items"].append({**copy.deepcopy(reduce_item), "id": "gemm", "files": ["gemm.cu"],
                          "kernels": [], "tier": 3, "library_mappings": [],
                          "depends_on": ["gemm"]})
    g = ground(plan, report)
    assert g["grounded"]  # nothing invented; the rest are gaps
    assert {x["construct"] for x in g["unmentioned_constructs"]} == {
        "warp_shuffle", "cooperative_groups"}
    assert g["tier_mismatches"] == [{"item": reduce_item["id"], "plan": 1, "scan": 2}]
    assert {"item": reduce_item["id"], "depends_on": "nope"} in g["bad_dependencies"]
    assert {"item": "gemm", "depends_on": "gemm"} in g["bad_dependencies"]
    assert {x["library"] for x in g["missing_library_mappings"]} == {"cuBLAS", "Thrust"}


def test_grounding_detects_dependency_cycles(report):
    plan = baseline_plan(report)
    a, b = plan["items"][0], plan["items"][1]
    a["depends_on"], b["depends_on"] = [b["id"]], [a["id"]]
    assert {"cycle": True} in ground(plan, report)["bad_dependencies"]


def test_request_uses_structured_output_and_meta_limits(report):
    t = ScriptedPlanTransport(report)
    r = make_plan(report, model="muse-spark-1.3", transport=t, environ=PRICED)
    body = t.requests[0]
    assert body["response_format"]["json_schema"]["schema"] == PLAN_SCHEMA
    assert body["max_completion_tokens"] == 32_000 and "max_tokens" not in body
    assert "tool_choice" not in body
    user = body["messages"][1]["content"]
    assert "### kernels/common.h" in user and "### gemm.cu" in user  # headers are included
    assert r.grounding["grounded"] and r.usage["input_tokens"] > 0
    assert r.cost_usd is not None and r.cost_usd < 0.01


def test_cost_cap_refuses_before_sending(report):
    t = ScriptedPlanTransport(report)
    with pytest.raises(SystemExit, match="exceeds the cap"):
        make_plan(report, model="muse-spark-1.3", transport=t, environ=PRICED,
                  max_cost_usd=0.01)
    assert t.requests == []


def test_unpriced_model_is_refused(report):
    with pytest.raises(ValueError, match="no price known"):
        make_plan(report, model="some-model", transport=ScriptedPlanTransport(report),
                  environ={})


def test_cli_dry_run_writes_labelled_outputs(tmp_path, capsys):
    assert main(["plan", str(MINI), "--dry-run", "--out", str(tmp_path)]) == 0
    saved = json.loads((tmp_path / "plan.json").read_text())
    assert saved["model"] == "scripted-dry-run (muse-spark-1.3)" and saved["cost_usd"] == 0.0
    assert saved["grounding"]["grounded"]
    assert (tmp_path / "scan.json").is_file()
    assert "# Port plan (scripted-dry-run" in (tmp_path / "plan.md").read_text()
    assert "grounded=True" in capsys.readouterr().out


# ----------------------------------------------------------------------------- streaming

def _sse(*chunks: dict | str) -> list[bytes]:
    out = []
    for c in chunks:
        out += [f"data: {c if isinstance(c, str) else json.dumps(c)}\n".encode(), b"\n"]
    return out


def test_assemble_stream_joins_content_and_keeps_usage():
    from museport.stream import assemble_stream
    usage = {"prompt_tokens": 10, "completion_tokens": 5}
    resp = assemble_stream(_sse(
        {"model": "m", "choices": [{"index": 0, "delta": {"content": '{"a"'}}]},
        {"choices": [{"index": 0, "delta": {"content": ": 1}"}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": usage},
        "[DONE]"))
    assert resp["choices"][0]["message"]["content"] == '{"a": 1}'
    assert resp["choices"][0]["finish_reason"] == "stop"
    assert resp["usage"] == usage and resp["model"] == "m"


def test_assemble_stream_without_usage_and_with_error():
    from museport.stream import assemble_stream
    resp = assemble_stream(_sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}, "[DONE]"))
    assert "usage" not in resp and resp["choices"][0]["message"]["content"] == ""
    with pytest.raises(RuntimeError, match="stream error"):
        assemble_stream(_sse({"error": {"message": "boom"}}))


class _Canned:
    """Transport returning a fixed response (as stream_transport would assemble it)."""

    def __init__(self, resp):
        self.resp, self.requests = resp, []

    def __call__(self, url, headers, body):
        self.requests.append(body)
        return self.resp


def test_empty_answer_is_an_error_naming_the_budget(report):
    t = _Canned({"choices": [{"finish_reason": "stop", "message": {"content": ""}}]})
    with pytest.raises(RuntimeError, match="spent on reasoning.*at most \\$"):
        make_plan(report, model="muse-spark-1.3", transport=t, environ=PRICED)


def test_missing_usage_means_unknown_cost_not_zero(report):
    plan = json.dumps(baseline_plan(report))
    t = _Canned({"choices": [{"finish_reason": "stop", "message": {"content": plan}}]})
    r = make_plan(report, model="muse-spark-1.3", transport=t, environ=PRICED)
    assert r.cost_usd is None and not r.usage_reported
    from museport.plan import render_plan_markdown
    assert "not reported by the API" in render_plan_markdown(r)


def test_stream_transport_does_not_retry_server_errors(monkeypatch):
    import io
    import urllib.error

    from museport import stream as P
    calls = []

    def fake_urlopen(req, timeout):
        calls.append(json.loads(req.data))
        raise urllib.error.HTTPError(req.full_url, 502, "bad", {}, io.BytesIO(b"upstream"))

    monkeypatch.setattr(P.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="HTTP 502"):
        P.stream_transport()("https://x/v1/chat/completions", {}, {"model": "m"})
    assert len(calls) == 1
    assert calls[0]["stream"] is True and calls[0]["stream_options"] == {"include_usage": True}
