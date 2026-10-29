"""E1/C6: the three-runs-per-arm table, rebuilt from the committed batch summaries."""

import json
from pathlib import Path

from museport.repeats import collect

ROOT = Path(__file__).resolve().parents[1]


def test_repeats_match_the_committed_table():
    c = collect(ROOT / "evidence")
    committed = json.loads((ROOT / "evidence" / "e1-c6-repeats" / "repeats.json").read_text())
    assert c == committed
    assert c["totals"]["feedback"]["per_run"] == [10, 10, 9]
    assert c["totals"]["baseline"]["per_run"] == [10, 9, 9]
    # every miss is fused_residual_forward
    assert {it for it, n in c["per_item"].items() if n["feedback"] < 3 or n["baseline"] < 3} == {
        "t1-fused-residual-forward"}
    # the item lost to an API error was rerun, and the rerun is the graded row
    b2 = c["arms"]["baseline"][1]
    assert b2["retried_after_api_error"] == ["t1-gelu-backward"]
    assert b2["by_item"]["t1-gelu-backward"]["correct"]
