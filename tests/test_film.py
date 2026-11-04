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
    assert round(f["run5"]["ratio"]) == 36 and "36 times slower than CUDA" in LINES["speed"]
    ratios = {s["device"]: round(s["ratio"], 2) for s in f["portable_speeds"]}
    mid = [r for d, r in ratios.items() if "H100" not in d]
    assert mid and all(1.3 <= r <= 1.6 for r in mid)
    assert "about one and a half times on mid-range NVIDIA" in LINES["speed"]
    assert ratios["NVIDIA H100 80GB HBM3"] > 2  # shown on the speed card
    assert round(f["amd"]["before_ms"]) == 1274 and int(f["amd"]["after_ms"]) == 63
    assert f["gpu_calls"]["calls"] == 23
    assert f["onemath"] == {"pr": "uxlfoundation/oneMath#793", "passed_before": 323,
                            "passed_after": 325}
    # facts the voiceover leaves out stay on screen
    film = (ROOT / "demo/remotion/src/Film.tsx").read_text()
    cards = (ROOT / "demo/remotion/src/shots/Cards.tsx").read_text()
    assert "worked in 1 of 2 tries" in film  # README: the one-run port succeeded in 1 of 2
    assert "Intel" in LINES["intel"] and "not tested: no cloud access" in cards
    assert f["j4_session"][-2].startswith("Once upon a time")


def test_the_required_opening_line_is_the_submissions() -> None:
    sub = (ROOT / "docs/SUBMISSION.md").read_text()
    start = sub.index("## Opening line of the 3-minute video (required)")
    start = sub.index('> "', start)
    quote = " ".join(ln.lstrip("> ").strip() for ln in sub[start:].split("\n\n")[0].splitlines())
    assert LINES["disclosure"] == quote.strip('"')
    # the disclosure the rules ask for: the pre-existing project, named, and what is new
    assert "Before the hackathon" in quote and "I built this week" in quote
    assert "oneapi-agent-toolkit" in sub[sub.index("## Opening line"):][:1200]  # named on screen
