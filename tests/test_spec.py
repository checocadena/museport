"""museport spec: Muse's harness specs, their static checks and the proof loop (no compiler)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import museport.spec as sp
from museport.llmc import SPECS

MINI = Path(__file__).parent / "fixtures" / "llmc_mini"

SCALE = {"cpu_functions": ["scale_cpu"], "variants": [1], "dropped_variants": [],
         "sizes": [{"name": "N", "value": 1000}],
         "inputs": [{"name": "inp", "count": "N", "dtype": "f32", "init": "uniform",
                     "device_count": ""},
                    {"name": "shift", "count": "1", "dtype": "f32", "init": "uniform",
                     "device_count": ""}],
         "outputs": [{"name": "out", "count": "N", "dtype": "f32", "init": "zeros",
                      "device_count": ""}],
         "cpu_call": "scale_cpu(out, inp, shift, N)",
         "gpu_call": "scale(k, d_out, d_inp, d_shift, N, 256)", "tol": 1e-5, "note": "",
         "host_prep": "", "gpu_prelude": "", "keep_in_harness": [], "extra_includes": [],
         "keep_cublas": False, "keep_cublaslt": False, "output_post": ""}


def test_hand_written_specs_round_trip_through_the_answer_schema():
    src = (MINI / "scale.cu").read_text()
    for f, spec in SPECS.items():
        ans = sp.spec_to_answer(spec)
        assert set(ans) == set(sp.SPEC_SCHEMA["required"]), f
        assert sp.to_kernel_spec(f, ans, src).cpu_call == spec.cpu_call


def test_dispatcher_variants_and_static_checks():
    src = (MINI / "scale.cu").read_text()
    assert sp.dispatcher_variants(src) == [1]
    assert sp.static_errors(SCALE, src) == []
    bad = {**SCALE, "cpu_functions": ["nope_cpu"], "variants": [1, 2],
           "sizes": [{"name": "N", "value": 1 << 30}]}
    errs = sp.static_errors(bad, src)
    assert any("'nope_cpu' is not defined" in e for e in errs)
    assert any("variants [2] are not cases" in e for e in errs)
    assert any("reduce the sizes" in e for e in errs)
    assert any("unknown size 'M'" in e for e in sp.static_errors(
        {**SCALE, "inputs": [{**SCALE["inputs"][0], "count": "M*2"}]}, src))
    assert any("__import__" in e or "unsupported" in e for e in sp.static_errors(
        {**SCALE, "inputs": [{**SCALE["inputs"][0], "count": "__import__('os')"}]}, src))


def test_dropped_variants_need_a_reason():
    src = "void f() { switch (kernel_num) { case 1: break; case 2: break; } }\n" \
          + (MINI / "scale.cu").read_text()
    assert any("[2] are neither" in e for e in sp.static_errors(SCALE, src))
    ok = {**SCALE, "dropped_variants": [{"variant": 2, "reason": "bf16 only"}]}
    assert not any("neither" in e for e in sp.static_errors(ok, src))


def test_failed_checks_go_back_to_the_model_until_the_spec_is_proven(tmp_path, monkeypatch):
    answers = [{**SCALE, "variants": [1, 7]}, {**SCALE, "tol": 1e-3}, SCALE]
    seen = []

    def fake(url, headers, body):
        seen.append(body["messages"])
        return {"choices": [{"message": {"content": json.dumps(answers.pop(0))},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 100}}

    proofs = [{"ok": False, "failed": "verify", "steps": {"verify": {"summary": "FAIL: 3 off"}}},
              {"ok": True, "steps": {}}]
    def check(task, work):
        assert task.is_absolute() and work.is_absolute()  # a relative work dir breaks migrate
        return proofs.pop(0)

    monkeypatch.setattr(sp, "check_harness", check)
    monkeypatch.chdir(tmp_path)
    row = sp.write_spec("scale.cu", MINI, Path("out"), model="muse-spark-1.3", transport=fake,
                        log=lambda m: None)
    tmp_path = tmp_path / "out"
    assert row["proven"] and row["attempts"] == 3 and row["compare"] is None
    assert "variants [7] are not cases" in seen[1][-1]["content"]
    assert "'verify': FAIL: 3 off" in seen[2][-1]["content"]
    attempts = json.loads((tmp_path / "scale" / "attempts.json").read_text())
    assert [bool(a["static_errors"]) for a in attempts] == [True, False, False]
    assert json.loads((tmp_path / "scale" / "spec.json").read_text())["tol"] == 1e-5
    assert (tmp_path / "scale" / "task" / "scale_harness.cu").exists()
    assert not (tmp_path / "scale" / "task" / "input.bin").exists()


def test_an_unproven_spec_leaves_no_task_and_the_cap_stops_attempts(tmp_path, monkeypatch):
    def fake(url, headers, body):
        return {"choices": [{"message": {"content": json.dumps(SCALE)},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 100}}

    monkeypatch.setattr(sp, "check_harness",
                        lambda task, work: {"ok": False, "failed": "compile",
                                            "steps": {"compile": {"summary": "error"}}})
    row = sp.write_spec("scale.cu", MINI, tmp_path, model="muse-spark-1.3", transport=fake,
                        log=lambda m: None)
    assert not row["proven"] and row["attempts"] == sp.MAX_ATTEMPTS
    assert not (tmp_path / "scale" / "task").exists()
    # an attempt starts only if its worst case (16k output tokens, about $0.07) fits the cap
    row = sp.write_spec("scale.cu", MINI, tmp_path, model="muse-spark-1.3", transport=fake,
                        cap_usd=0.05, log=lambda m: None)
    assert not row["proven"] and row["attempts"] == 0 and row["cost_usd"] == 0


def test_an_answer_wrapped_in_prose_still_parses():
    from museport.plan import json_object

    text = 'Dropping the kernel alone didn\'t fix it {"a": {"b": 1}} trailing'
    assert json_object(text) == {"a": {"b": 1}}
    assert json_object('{"x": 2}') == {"x": 2}


def _cpu_fails(step):
    return lambda task, work: {"ok": False, "failed": step, "steps": {step: {"summary": "c2s"}}}


def test_the_gpu_proof_runs_before_any_feedback(tmp_path, monkeypatch):
    seen = []

    def fake(url, headers, body):
        seen.append(body["messages"])
        return {"choices": [{"message": {"content": json.dumps(SCALE)}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 100}}

    monkeypatch.setattr(sp, "check_harness", _cpu_fails("compile"))
    gpu_calls = []

    def gpu(task_dir, work):
        gpu_calls.append(task_dir)
        assert (task_dir / "input.bin").exists()  # the recording needs the task's inputs
        return {"ok": True, "step": "compare", "summary": "PASS", "cost_usd": 0.003}

    row = sp.write_spec("scale.cu", MINI, tmp_path, model="muse-spark-1.3", transport=fake,
                        prove_on_gpu=gpu, log=lambda m: None)
    assert row["proven"] and row["proven_by"] == "gpu" and row["attempts"] == 1
    assert row["gpu_usd"] == 0.003 and len(seen) == 1 and len(gpu_calls) == 1


def test_feedback_names_both_proofs_when_both_fail(tmp_path, monkeypatch):
    seen = []

    def fake(url, headers, body):
        seen.append(body["messages"])
        return {"choices": [{"message": {"content": json.dumps(SCALE)}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 100}}

    monkeypatch.setattr(sp, "check_harness", _cpu_fails("verify"))
    row = sp.write_spec("scale.cu", MINI, tmp_path, model="muse-spark-1.3", transport=fake,
                        prove_on_gpu=lambda t, w: {"ok": False, "step": "compare",
                                                   "summary": "FAIL: 9 off", "cost_usd": 0.003},
                        log=lambda m: None)
    assert not row["proven"] and row["gpu_usd"] == 0.009
    assert "also failed, at compare: FAIL: 9 off" in seen[1][-1]["content"]
    # a failure SYCLomatic cannot cause (the reference does not build) skips the GPU proof
    monkeypatch.setattr(sp, "check_harness", _cpu_fails("reference_build"))
    row = sp.write_spec("scale.cu", MINI, tmp_path, model="muse-spark-1.3", transport=fake,
                        prove_on_gpu=lambda t, w: pytest.fail("no GPU proof expected"),
                        log=lambda m: None)
    assert row["gpu_usd"] == 0


def test_gpu_proof_reads_the_recording_and_its_cost(tmp_path, monkeypatch):
    import hashlib

    import museport.record as rec

    task = tmp_path / "task"
    task.mkdir()
    (task / "input.bin").write_bytes(b"\0" * 8)
    (task / "task.json").write_text(json.dumps({"name": "llmc_scale", "inputs": ["input.bin"],
                                                "reference_build": ["g++"],
                                                "reference_cmd": "build/reference {input}"}))

    def runner(cmd, cwd, **kw):
        assert cmd[:2] == ["oneapi-agent", "remote-test"] and "--into" in cmd
        res = Path(cmd[cmd.index("--into") + 1]) / "nvidia-l4" / "llmc_scale"
        (res / "results" / "recorded").mkdir(parents=True)
        (res / "results" / "recorded" / "case0.out").write_bytes(b"out")
        (res / "summary.json").write_text(json.dumps({
            "status": "ok", "summary": "recorded", "evidence": {
                "device": {"name": "NVIDIA L4"}, "deterministic": True,
            "recorded": [{"case": 0, "input": "input.bin", "file": "recorded/case0.out",
                          "sha256": hashlib.sha256(b"out").hexdigest()}]}}))
        return type("P", (), {"stdout": 'log\n{"est_cost_usd": 0.004}', "stderr": ""})()

    monkeypatch.setattr(rec, "check_recording",
                        lambda t, d, w: {"status": "ok", "summary": "PASS (worst 0)"})
    out = sp.gpu_proof(task, tmp_path / "gpu", runner=runner)
    assert out["ok"] and out["cost_usd"] == 0.004 and out["device"] == "NVIDIA L4"
    assert out["summary"] == "PASS (worst 0)" and out["deterministic"] is True
