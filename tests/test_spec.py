"""museport spec: Muse's harness specs, their static checks and the proof loop (no compiler)."""

from __future__ import annotations

import json
from pathlib import Path

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
    monkeypatch.setattr(sp, "check_harness", lambda task, work: proofs.pop(0))
    row = sp.write_spec("scale.cu", MINI, tmp_path, model="muse-spark-1.3", transport=fake,
                        log=lambda m: None)
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
