"""The film's spoken numbers match the committed evidence (demo/remotion/public/lines.json)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LINES = {ln["shot"]: ln["text"]
         for ln in json.loads((ROOT / "demo/remotion/public/lines.json").read_text())}


def _film() -> dict:
    spec = importlib.util.spec_from_file_location(
        "make_film_data", ROOT / "demo/remotion/make_film_data.py")
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod.main()["film"]


def test_spoken_numbers_come_from_the_evidence() -> None:
    f = _film()
    k = f["devcuda"]
    assert (k["files"], k["attempted"], k["correct"], k["nvidia"]) == (22, 21, 20, 19)
    assert (f"ported {k['attempted']} of llm.c's {k['files']} kernel files" in LINES["kernels"]
            and f"{k['correct']} pass their hidden check" in LINES["kernels"]
            and f"{k['nvidia']} also pass on NVIDIA" in LINES["kernels"])
    ratios = {s["device"]: round(s["ratio"], 2) for s in f["portable_speeds"]}
    assert ratios["NVIDIA H100 80GB HBM3"] == 2.90 and "2.9 times" in LINES["speed"]
    mid = [r for d, r in ratios.items() if "H100" not in d]
    assert all(1.3 <= r <= 1.6 for r in mid) and "about one and a half" in LINES["speed"]
    assert round(f["amd"]["before_ms"]) == 1274 and int(f["amd"]["after_ms"]) == 63
    assert "1,274 milliseconds a step, down to 63" in LINES["speed"]
    assert "2.9 times slower" in LINES["limits"]
    assert f["onemath"] == {"pr": "uxlfoundation/oneMath#793", "passed_before": 323,
                            "passed_after": 325}
    assert f["j3_test"][-1] == "overall okay: 1"
    assert f["j4_session"][-2].startswith("Once upon a time")


def test_the_required_opening_line_is_the_submissions() -> None:
    sub = (ROOT / "docs/SUBMISSION.md").read_text()
    start = sub.index('> "Pre-existing:')
    quote = " ".join(ln.lstrip("> ").strip() for ln in sub[start:].split("\n\n")[0].splitlines())
    assert LINES["disclosure"] == quote.strip('"')
