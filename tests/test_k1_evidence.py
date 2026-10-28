"""K1's committed numbers come from its committed results."""

from __future__ import annotations

import json
from pathlib import Path

from museport.curves import chart_svg, collect, worst

ROOT = Path(__file__).resolve().parents[1]


def test_k1_numbers_and_chart_match_the_results():
    c = collect(ROOT / "evidence")
    assert len(c["cuda"]["train"]) == 298 and len(c["cuda"]["val"]) == 13
    assert round(worst(c["cuda_repeat"], c["cuda"]), 4) == 0.0030
    assert round(worst(c["sycl_nvidia"], c["cuda"]), 4) == 0.0042
    assert round(worst(c["sycl_amd"], c["cuda"]), 4) == 0.0037
    d = ROOT / "evidence" / "k1-curves"
    assert (d / "chart.svg").read_text() == chart_svg(c)
    assert json.loads((d / "curves.json").read_text())["curves"] == json.loads(json.dumps(c))
