"""``museport review``: Muse as a second agent that reviews finished ports (plan item E3).

For each port, a separate Muse call sees only the CUDA harness and Muse's SYCL port, never a
test result. It answers in a fixed schema: per variant, whether the SYCL kernel keeps the CUDA
variant's algorithm; whether variants were copied; whether the SYCL is idiomatic; whether the
port likely computes the same results; and a verdict. The verdicts are then compared with the
hidden grade and the variant check, which the reviewer never sees.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from oneapi_agent.loop import model_pricing

from .plan import CHARS_PER_TOKEN, call_model
from .report import CURRENT_BATCHES, Item, load

REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["variants", "copies_one_implementation", "idiomatic_sycl", "likely_correct",
                 "concerns", "verdict"],
    "properties": {
        "variants": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["variant", "keeps_cuda_algorithm", "note"],
            "properties": {"variant": {"type": "integer"},
                           "keeps_cuda_algorithm": {"type": "boolean"},
                           "note": {"type": "string"}}}},
        "copies_one_implementation": {"type": "boolean"},
        "idiomatic_sycl": {"type": "boolean"},
        "likely_correct": {"type": "boolean"},
        "concerns": {"type": "array", "items": {"type": "string"}},
        "verdict": {"type": "string", "enum": ["accept", "reject"]},
    },
}

SYSTEM = (
    "You review a port of CUDA code to SYCL 2020, written by another agent. You see the CUDA "
    "program and the SYCL port, and no test results. Judge from the code alone:\n"
    "- for each kernel variant k, whether its SYCL kernel keeps the CUDA variant's algorithm "
    "(its parallel structure: work per thread, shared memory, warp or sub-group operations, "
    "atomics, launch shape), not merely the same math;\n"
    "- whether several variants share one implementation;\n"
    "- whether the SYCL is idiomatic (no dpct helpers, no CUDA idioms carried over);\n"
    "- whether the port likely computes the same results as the CUDA program within fp32 "
    "tolerance (look for indexing, reduction, synchronization and precision bugs);\n"
    "- a verdict: accept only if the port is likely correct and keeps every variant.\n"
    "Be specific in notes and concerns; cite functions or lines."
)

# Answer budget per review, reasoning included.
MAX_OUTPUT_TOKENS = 16_000


def review_targets(evidence: Path, batches: tuple[str, ...]) -> list[Item]:
    return [it for it in load(evidence)["items"] if it.batch in batches]


def messages_for(it: Item) -> list[dict[str, str]]:
    harness = it.dir / "task" / it.task["cuda_path"]
    ports = sorted(p for p in (it.dir / "port").rglob("*")
                   if p.is_file() and "migrated" not in p.parts)
    variants = it.task.get("museport", {}).get("variants", [])
    parts = [f"Kernel variants to keep, each as its own SYCL kernel: {variants}.",
             f"\n=== CUDA program: {harness.name} ===\n{harness.read_text()}"]
    parts += [f"\n=== SYCL port: {p.relative_to(it.dir / 'port')} ===\n{p.read_text()}"
              for p in ports]
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n".join(parts)}]


def worst_case_usd(model: str, messages: list[dict[str, str]],
                   environ: Mapping[str, str] | None = None) -> float:
    p = model_pricing(model, environ)
    tokens_in = sum(len(m["content"]) for m in messages) / CHARS_PER_TOKEN
    return (tokens_in * p.input + MAX_OUTPUT_TOKENS * p.output) / 1e6


def compare(it: Item, review: dict[str, Any]) -> dict[str, Any]:
    """The reviewer's answers beside what the hidden grade and the variant check found."""
    vc = it.row.get("variant_check") or {}
    return {
        "hidden_grade_correct": it.correct,
        "reviewer_likely_correct": review["likely_correct"],
        "variant_check_duplicated": bool(vc.get("duplicated_variants")),
        "reviewer_copies": review["copies_one_implementation"],
        "reviewer_verdict": review["verdict"],
        "variants_not_kept": [v["variant"] for v in review["variants"]
                              if not v["keeps_cuda_algorithm"]],
    }


def run_reviews(evidence: Path, out: Path, *, model: str, provider: str = "meta",
                batches: tuple[str, ...] = (*CURRENT_BATCHES, "port-llmc-softmax-rerun-2"),
                total_cap_usd: float = 1.5, transport: Any = None,
                log: Any = print) -> dict[str, Any]:
    """Review every port in ``batches``; a review starts only if the spend so far plus its
    worst case stays within ``total_cap_usd``."""
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    spent = 0.0
    p = model_pricing(model)
    for it in review_targets(evidence, batches):
        msgs = messages_for(it)
        worst = worst_case_usd(model, msgs)
        row: dict[str, Any] = {"batch": it.batch, "item": it.id, "worst_case_usd": round(worst, 4)}
        rows.append(row)
        if spent + worst > total_cap_usd:
            row["status"] = "skipped_total_cap"
            log(f"[review] {it.batch}/{it.id}: skipped (${spent:.2f} + ${worst:.2f} > cap)")
            continue
        try:
            review, usage = call_model(msgs, model=model, provider=provider,
                                       max_output_tokens=MAX_OUTPUT_TOKENS, transport=transport,
                                       schema_name="port_review", schema=REVIEW_SCHEMA)
        except RuntimeError as e:
            row.update(status="error", error=str(e)[:500], cost_usd=worst,
                       cost_note="no usage: counted at the worst case")
            spent += worst
            log(f"[review] {it.batch}/{it.id}: error {e}")
            continue
        cost = ((usage["input_tokens"] * p.input + usage["output_tokens"] * p.output
                 + usage["cache_read_input_tokens"] * p.cache_read) / 1e6 if usage else worst)
        spent += cost
        row.update(status="ok", usage=usage, cost_usd=round(cost, 6), review=review,
                   compare=compare(it, review))
        log(f"[review] {it.batch}/{it.id}: {review['verdict']} "
            f"(hidden grade {'pass' if it.correct else 'fail'}), ${cost:.4f}")
        _write(out, model, rows, spent)
    return _write(out, model, rows, spent)


def _write(out: Path, model: str, rows: list[dict[str, Any]], spent: float) -> dict[str, Any]:
    summary = {"model": model, "spent_usd": round(spent, 4), "reviews": rows}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
