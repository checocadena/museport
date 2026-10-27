import json
import re
from pathlib import Path

from museport.report import build, load, tier_table

ROOT = Path(__file__).resolve().parents[1]


def _batch(root: Path, name: str, items: list[dict]) -> None:
    b = root / name
    for it in items:
        d = b / it["item"]
        (d / "task").mkdir(parents=True)
        (d / "task" / "task.json").write_text(json.dumps({
            "name": f"llmc_{it['item']}", "cuda_path": "k_harness.cu", "goal": "Port <k>.",
            "tolerance": 1e-5, "museport": {"source": "llm.c k.cu", "variants": [1]}}))
        (d / "task" / "k_harness.cu").write_text("__global__ void k() {}\n")
        (d / "port").mkdir()
        (d / "port" / "k.cpp").write_text("q.parallel_for(r, [=](auto i) {});\n")
        (d / "trace.jsonl").write_text("\n".join(json.dumps(e) for e in [
            {"type": "model_response", "iteration": 1,
             "content": [{"type": "text", "text": "Writing <the> port."}]},
            {"type": "tool_result", "iteration": 1, "name": "write_file",
             "input": {"path": "k.cpp", "content": "int x;\n"},
             "result": {"status": "ok", "summary": "wrote k.cpp"}},
            {"type": "stop", "stop_reason": "success", "stop_detail": "done"}]))
    (b / "summary.json").write_text(json.dumps({"model": "muse", "spent_usd": 0.3,
                                                "items": items}))


def _row(item: str, helpers: int, status: str = "success", cost: float = 0.1) -> dict:
    return {"item": item, "file": "k.cu", "status": status, "iterations": 3, "cost_usd": cost,
            "evaluation": {"verify_status": "ok", "helpers": helpers, "verify_summary": "PASS"},
            "variant_check": {"variants": 1, "kernels": 1, "distinct_kernel_bodies": 1,
                              "duplicated_variants": False}}


def test_site_is_built_from_the_evidence(tmp_path):
    ev = tmp_path / "evidence"
    _batch(ev, "port-x-tier1", [_row("t1-a", 0), _row("t1-b", 4, "max_iterations")])
    _batch(ev, "port-x-reruns", [_row("t1-b", 0, cost=0.2)])
    g = ev / "port-x-tier1" / "gpu" / "nvidia" / "rtx" / "llmc_t1-a"
    g.mkdir(parents=True)
    (g / "summary.json").write_text(json.dumps({"status": "ok", "summary": "PASS on GPU",
        "evidence": {"vendor": "nvidia", "device": {"name": "NVIDIA RTX"}, "steps": []}}))

    data = load(ev)
    t1, _, _, total = tier_table(data["items"])
    assert (t1["items"], t1["correct"], t1["success"], t1["gpu"]) == (2, 1, 1, 1)
    assert total["cost"] == 0.2  # the rerun is reported beside the first runs, not counted

    out = build(ev, tmp_path / "site")
    assert "items/port-x-reruns--t1-b.html" in out["pages"]
    index = (tmp_path / "site" / "index.html").read_text()
    assert "Reruns with a stricter goal" in index and "correct, 4 helpers" in index
    item = (tmp_path / "site" / "items" / "port-x-tier1--t1-a.html").read_text()
    assert "Writing &lt;the&gt; port." in item and "<the>" not in item
    assert "TURN 1" in item and "pass: RTX" in item and "Port &lt;k&gt;." in item


def test_readme_table_matches_the_evidence():
    """The README's results table is checked against the committed evidence."""
    rows = {r["label"]: r for r in tier_table(load(ROOT / "evidence")["items"])}
    readme = (ROOT / "README.md").read_text()

    def cells(name: str) -> list[str]:
        line = next(ln for ln in readme.splitlines() if ln.startswith(f"| {name}"))
        return [re.match(r"\**\$?([\d.]+)", c.strip()).group(1)
                for c in line.strip("|").split("|")[1:]]

    labels = ["Tier 1", "Tier 2", "Tier 3", "All"]
    assert cells("Items ported") == [str(rows[k]["items"]) for k in labels]
    assert cells("Correct with 0 helpers") == [str(rows[k]["correct"]) for k in labels]
    assert cells("Also met the loop") == [str(rows[k]["success"]) for k in labels]
    assert cells("Pass on an NVIDIA GPU") == [str(rows[k]["gpu"]) for k in labels]
    assert cells("Muse cost") == [f"{rows[k]['cost']:.2f}" for k in labels]
