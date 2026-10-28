import json
import re
from pathlib import Path
from typing import Any

from museport.report import build, current_table, load, tier_table

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
    (ev / "port-x-reruns" / "README.md").write_text("# Tier 1 again, with <fixes>\n\nText.\n")
    g = ev / "port-x-tier1" / "gpu" / "nvidia" / "rtx" / "llmc_t1-a"
    g.mkdir(parents=True)
    (g / "summary.json").write_text(json.dumps({"status": "ok", "summary": "PASS on GPU",
        "evidence": {"vendor": "nvidia", "device": {"name": "NVIDIA RTX"}, "steps": [
            {"step": "sycl_verify_vs_native", "status": "ok"}]}}))
    # the AMD layout (gpu/<device>/<task>/): checked against the reference only
    a = ev / "port-x-reruns" / "gpu" / "amd-mi350x" / "llmc_t1-b"
    a.mkdir(parents=True)
    (a / "summary.json").write_text(json.dumps({"status": "ok", "summary": "PASS on AMD",
        "evidence": {"vendor": "amd", "device": {"name": "AMD Instinct MI350X VF gfx950"},
                     "steps": [{"step": "native_build", "status": "fail"}]}}))

    data = load(ev)
    t1, _, _, total = tier_table(data["items"])
    assert (t1["items"], t1["correct"], t1["success"], t1["gpu"]) == (2, 1, 1, 1)
    assert total["cost"] == 0.2  # the rerun is reported beside the first runs, not counted

    out = build(ev, tmp_path / "site")
    assert "items/port-x-reruns--t1-b.html" in out["pages"]
    index = (tmp_path / "site" / "index.html").read_text()
    assert "Tier 1 again, with &lt;fixes&gt;" in index and "correct, 4 helpers" in index
    item = (tmp_path / "site" / "items" / "port-x-tier1--t1-a.html").read_text()
    assert "Writing &lt;the&gt; port." in item and "<the>" not in item
    assert "TURN 1" in item and "pass: RTX" in item and "Port &lt;k&gt;." in item
    amd = (tmp_path / "site" / "items" / "port-x-reruns--t1-b.html").read_text()
    assert "pass: MI350X VF (vs reference)" in amd


def test_readme_tables_match_the_evidence():
    """Both README results tables are checked against the committed evidence: the runs with
    the fixed loop (headline), and the first runs (under "How we got here")."""
    items = load(ROOT / "evidence")["items"]
    readme = (ROOT / "README.md").read_text()
    head, history = readme.split("## How we got here", 1)
    labels = ["Tier 1", "Tier 2", "Tier 3", "All"]

    def cells(text: str, name: str) -> list[str]:
        line = next(ln for ln in text.splitlines() if ln.startswith(f"| {name}"))
        return [re.match(r"\**\$?([\d.]+)", c.strip()).group(1)
                for c in line.strip("|").split("|")[1:]]

    def check(text: str, table: list[dict[str, Any]], amd: bool) -> None:
        rows = {r["label"]: r for r in table}
        assert cells(text, "Items ported") == [str(rows[k]["items"]) for k in labels]
        assert cells(text, "Correct with 0 helpers") == [str(rows[k]["correct"]) for k in labels]
        assert cells(text, "Also met the loop") == [str(rows[k]["success"]) for k in labels]
        assert cells(text, "Pass on an NVIDIA GPU") == [str(rows[k]["gpu"]) for k in labels]
        if amd:
            assert cells(text, "Pass on an AMD GPU") == [str(rows[k]["amd"]) for k in labels]
        assert cells(text, "Muse cost") == [f"{rows[k]['cost']:.2f}" for k in labels]

    check(head, current_table(items), amd=True)
    check(history, tier_table(items), amd=False)


def test_gpt2_stages_get_pages_and_an_overview_row(tmp_path):
    out = build(ROOT / "evidence", tmp_path / "site")
    stages = sorted(p for p in out["pages"] if p.startswith("gpt2/"))
    assert stages == ["gpt2/gpt2-stage1.html", "gpt2/gpt2-stage2.html", "gpt2/gpt2-stage3.html"]
    index = (tmp_path / "site" / "index.html").read_text()
    assert 'href="gpt2/gpt2-stage3.html"' in index and "RTX 4000 Ada" in index
    stage3 = (tmp_path / "site" / "gpt2" / "gpt2-stage3.html").read_text()
    assert "TURN 16" in stage3 and "gpt2_train.cpp" in stage3 and "sycl_verify_vs_native" in stage3


def test_differential_checks_get_a_section(tmp_path):
    build(ROOT / "evidence", tmp_path / "site")
    index = (tmp_path / "site" / "index.html").read_text()
    section = index[index.index('<section id="diff">'):]
    section = section[:section.index("</section>")]
    assert section.count("<tr><td><code>t1-") == 10
    assert section.count("pill ok\">pass") == 2 * 10 + 2  # 10 x (recording, port) + subtle bug
    assert "pill bad\">fail" in section and "worst 1.43e-06" in section
    assert 'href="#diff"' in index and "MI350X VF" in index[index.index('id="gpt2"'):]


def test_muse_written_specs_get_a_section(tmp_path):
    build(ROOT / "evidence", tmp_path / "site")
    index = (tmp_path / "site" / "index.html").read_text()
    section = index[index.index('<section id="specs">'):]
    section = section[:section.index("</section>")]
    assert section.count("<tr><td><code>") == 17
    assert ">16 of 17<" in section and ">16 of 16<" in section
    assert section.count("not proven") == 1 and "trimat_forward" in section
    assert "gpu, after the run" in section and 'href="#specs"' in index
