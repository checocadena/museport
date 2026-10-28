"""``museport plan``: Muse reads a CUDA repository and writes its porting plan; the plan is
then checked against the scanner's ground truth.

The model sees the repository's CUDA sources and the scan, and must answer with a plan in a
fixed JSON schema (``PLAN_SCHEMA``). ``ground`` then checks every claim in the plan that can
be checked without a compiler: referenced files and kernels exist, every CUDA file is either
planned or explicitly left out, NVIDIA libraries get a oneAPI mapping, CUDA-only constructs
show up as risks, tiers agree with the scan, and dependencies form a valid order. A plan that
invents files or kernels is reported as ungrounded, whatever it claims.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from oneapi_agent.loop import model_pricing
from oneapi_agent.models import (
    PROVIDERS,
    Transport,
    auth_from_network,
    provider_api_key,
)

from .scan import CONSTRUCTS
from .stream import stream_transport

PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["summary", "items", "out_of_scope"],
    "properties": {
        "summary": {"type": "string"},
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "title", "files", "kernels", "tier", "approach",
                             "library_mappings", "risks", "verification", "depends_on"],
                "properties": {
                    "id": {"type": "string"},
                    "title": {"type": "string"},
                    "files": {"type": "array", "items": {"type": "string"}},
                    "kernels": {"type": "array", "items": {"type": "string"}},
                    "tier": {"type": "integer", "enum": [1, 2, 3]},
                    "approach": {"type": "string"},
                    "library_mappings": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["from", "to", "notes"],
                            "properties": {"from": {"type": "string"}, "to": {"type": "string"},
                                           "notes": {"type": "string"}},
                        },
                    },
                    "risks": {"type": "array", "items": {"type": "string"}},
                    "verification": {"type": "string"},
                    "depends_on": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "out_of_scope": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["files", "reason"],
                "properties": {"files": {"type": "array", "items": {"type": "string"}},
                               "reason": {"type": "string"}},
            },
        },
    },
}

SYSTEM_PROMPT = """\
You plan the migration of a CUDA code base to SYCL 2020 (Intel oneAPI DPC++), to run on
NVIDIA, AMD and Intel GPUs. You get every CUDA source file of the repository and a scan of it.

Write a plan that an agent can execute one item at a time, where each item is verified on
its own before the next starts. Rules:
- Use only file paths and kernel names that appear in the sources. Never invent any.
- Every CUDA file in the scan must be in exactly one item or in out_of_scope, with a reason.
- Tier: 1 plain kernels; 2 CUDA-only constructs that need a SYCL design (warp size 32,
  warp shuffles and votes, cooperative groups, bf16, inline PTX, tensor cores); 3 NVIDIA
  libraries (cuBLAS, cuBLASLt, cuDNN, cuFFT, Thrust, NCCL, ...).
- For every NVIDIA library an item touches, give a library_mappings entry with the oneAPI
  replacement (oneMath, oneDNN, oneDPL, oneCCL) and what does not map directly.
- risks: name each CUDA-only construct in the item's files and how SYCL handles it
  (e.g. do not assume a sub-group size of 32).
- verification: how the item's output is checked against the original (reference program,
  tolerance, inputs). Prefer checks that already exist in the repository.
- depends_on: ids of items that must be done first (shared headers before their users).
- Order items from easiest to hardest, so the pipeline is proven on simple kernels first.
Answer with JSON only, matching the given schema."""

# ~4 characters per token for code; used only to estimate cost before sending.
CHARS_PER_TOKEN = 4


@dataclass
class PlanResult:
    plan: dict[str, Any]
    usage: dict[str, int]
    cost_usd: float | None
    model: str
    grounding: dict[str, Any]
    usage_reported: bool = True


def collect_sources(scan_report: dict[str, Any], *, max_chars: int = 2_000_000
                    ) -> dict[str, str]:
    """The text of every scanned CUDA file plus the local headers they include."""
    root = Path(scan_report["root"])
    paths: list[str] = []
    for f in scan_report["files"]:
        for p in [*f.get("includes", []), f["path"]]:
            if p not in paths:
                paths.append(p)
    out: dict[str, str] = {}
    total = 0
    for p in paths:
        text = (root / p).read_text(errors="replace")
        if total + len(text) > max_chars:
            raise ValueError(f"sources exceed {max_chars} characters; narrow the scan "
                             "(museport plan --subdir ...)")
        out[p] = text
        total += len(text)
    return out


def build_messages(scan_report: dict[str, Any], sources: Mapping[str, str]
                   ) -> list[dict[str, str]]:
    scan_view = {k: scan_report[k] for k in ("totals", "libraries", "constructs")}
    scan_view["files"] = [{k: f[k] for k in ("path", "lines", "tier", "kernels", "libraries",
                                             "constructs", "includes", "inherited_libraries",
                                             "inherited_constructs")}
                          for f in scan_report["files"]]
    parts = ["## Scan (deterministic, from museport scan)",
             "```json", json.dumps(scan_view, indent=1), "```", "", "## Sources"]
    for path, text in sources.items():
        parts += [f"### {path}", "```cuda", text, "```"]
    parts += ["", "Write the plan now, as JSON matching the schema."]
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(parts)}]


def estimate_cost_usd(model: str, messages: list[dict[str, str]], max_output_tokens: int,
                      environ: Mapping[str, str] | None = None) -> float:
    """Upper-bound estimate: all input uncached, the full output budget used."""
    p = model_pricing(model, environ)
    tokens_in = sum(len(m["content"]) for m in messages) / CHARS_PER_TOKEN
    return (tokens_in * p.input + max_output_tokens * p.output) / 1e6


def call_model(messages: list[dict[str, str]], *, model: str, provider: str,
               max_output_tokens: int, transport: Transport | None = None,
               environ: Mapping[str, str] | None = None, schema_name: str = "port_plan",
               schema: dict[str, Any] | None = None
               ) -> tuple[dict[str, Any], dict[str, int] | None]:
    """One structured-output chat completion (the plan schema unless ``schema`` is given);
    returns (answer, usage), usage None if the server did not report it."""
    p = PROVIDERS[provider]
    env = os.environ if environ is None else environ
    key = provider_api_key(provider, env)
    if not key and not auth_from_network(env) and transport is None:
        raise SystemExit(f"no API key for {provider}: set {p.key_env[0]} (or "
                         "ONEAPI_AGENT_AUTH_FROM_NETWORK=1 if a proxy adds it)")
    base = ((env.get(p.base_url_env) if p.base_url_env else None) or p.default_base_url or "")
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_output_tokens,
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": schema_name,
                                            "schema": schema or PLAN_SCHEMA}},
    }
    if env.get("ONEAPI_AGENT_EFFORT"):
        body["reasoning_effort"] = env["ONEAPI_AGENT_EFFORT"]
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    resp = (transport or stream_transport())(f"{base.rstrip('/')}/chat/completions", headers,
                                             body)
    choice = (resp.get("choices") or [{}])[0]
    content = (choice.get("message") or {}).get("content") or ""
    u = resp.get("usage")
    if choice.get("finish_reason") == "length" or not content.strip():
        reasoning = ((u or {}).get("completion_tokens_details") or {}).get("reasoning_tokens")
        raise RuntimeError(
            f"no complete plan (finish_reason={choice.get('finish_reason')}, "
            f"{len(content)} characters, reasoning tokens {reasoning if u else 'not reported'}"
            "): the output budget was probably spent on reasoning; raise --max-output "
            "or lower ONEAPI_AGENT_EFFORT")
    try:
        plan = json.loads(content)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"the model's answer is not JSON: {e}; first 200 chars: "
                           f"{content[:200]!r}") from None
    if not u:
        return plan, None
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
    usage = {"input_tokens": max((u.get("prompt_tokens") or 0) - cached, 0),
             "output_tokens": u.get("completion_tokens") or 0,
             "cache_creation_input_tokens": 0, "cache_read_input_tokens": cached}
    return plan, usage


# ----------------------------------------------------------------------------- grounding


def ground(plan: dict[str, Any], scan_report: dict[str, Any]) -> dict[str, Any]:
    """Check the plan against the scan. ``grounded`` is False if it names anything that does
    not exist; the other findings are gaps in coverage or consistency."""
    files = {f["path"]: f for f in scan_report["files"]}
    known_paths = set(files) | {i for f in scan_report["files"] for i in f.get("includes", [])}
    items = plan.get("items", [])
    ids = [it.get("id") for it in items]
    findings: dict[str, list[Any]] = {k: [] for k in (
        "unknown_files", "unknown_kernels", "uncovered_files", "duplicate_files",
        "missing_library_mappings", "unmentioned_constructs", "tier_mismatches",
        "bad_dependencies", "duplicate_ids")}

    findings["duplicate_ids"] = sorted({i for i in ids if ids.count(i) > 1})
    placed: dict[str, list[str]] = {}
    for it in items:
        for path in it.get("files", []):
            placed.setdefault(path, []).append(it.get("id", "?"))
    for oos in plan.get("out_of_scope", []):
        for path in oos.get("files", []):
            placed.setdefault(path, []).append("out_of_scope")
    for path, where in placed.items():
        if path not in known_paths:
            findings["unknown_files"].append(path)
        elif len(where) > 1 and path in files:
            findings["duplicate_files"].append({"file": path, "in": where})
    findings["uncovered_files"] = sorted(p for p in files if p not in placed)

    design = {c.name for c in CONSTRUCTS if c.severity == "design"}
    for it in items:
        iid = it.get("id", "?")
        its_files = [files[p] for p in it.get("files", []) if p in files]
        kernels_there = {k for f in its_files for k in f["kernels"]}
        for k in it.get("kernels", []):
            if k not in kernels_there:
                findings["unknown_kernels"].append({"item": iid, "kernel": k})
        mapped = " ".join(f"{m.get('from', '')} {m.get('to', '')}"
                          for m in it.get("library_mappings", [])).lower()
        for lib in sorted({lib for f in its_files for lib in f["libraries"]}):
            if lib.lower() not in mapped:
                findings["missing_library_mappings"].append({"item": iid, "library": lib})
        risk_text = " ".join(it.get("risks", []) + [it.get("approach", "")]).lower()
        for con in sorted({c for f in its_files for c in f["constructs"] if c in design}):
            if not any(word in risk_text for word in _construct_words(con)):
                findings["unmentioned_constructs"].append({"item": iid, "construct": con})
        if its_files:
            want = max(f["tier"] for f in its_files)
            if it.get("tier") != want:
                findings["tier_mismatches"].append({"item": iid, "plan": it.get("tier"),
                                                    "scan": want})
        for dep in it.get("depends_on", []):
            if dep not in ids or dep == iid:
                findings["bad_dependencies"].append({"item": iid, "depends_on": dep})
    if _has_cycle(items):
        findings["bad_dependencies"].append({"cycle": True})

    covered = len([p for p in files if p in placed])
    return {
        "grounded": not findings["unknown_files"] and not findings["unknown_kernels"],
        "coverage": {"files": len(files), "covered": covered,
                     "planned": len({p for it in items for p in it.get("files", [])
                                     if p in files})},
        "items": len(items),
        **findings,
        "issues": sum(len(v) for v in findings.values()),
    }


def _construct_words(name: str) -> tuple[str, ...]:
    return {
        "warp_size_32": ("warp size", "sub-group size", "subgroup size", "warp_size", "32"),
        "warp_shuffle": ("shuffle", "shfl", "shift_group", "permute_group", "select_from_group"),
        "warp_vote": ("ballot", "vote", "any_of_group", "all_of_group"),
        "cooperative_groups": ("cooperative", "cg::", "group algorithm", "sycl::group"),
        "bf16": ("bf16", "bfloat16"),
        "inline_ptx": ("ptx", "inline asm", "asm"),
        "tensor_cores": ("tensor core", "wmma", "joint_matrix", "mma"),
        "async_copy": ("async", "cp.async", "memcpy_async"),
        "textures": ("texture", "image", "sampler"),
    }.get(name, (name.replace("_", " "),))


def _has_cycle(items: list[dict[str, Any]]) -> bool:
    deps = {it.get("id"): [d for d in it.get("depends_on", [])] for it in items}
    state: dict[Any, int] = {}

    def visit(n: Any) -> bool:
        if state.get(n) == 1:
            return True
        if state.get(n) == 2 or n not in deps:
            return False
        state[n] = 1
        if any(visit(d) for d in deps[n]):
            return True
        state[n] = 2
        return False

    return any(visit(n) for n in deps)


# ----------------------------------------------------------------------------- dry run


def baseline_plan(scan_report: dict[str, Any]) -> dict[str, Any]:
    """A mechanical plan built from the scan alone: one item per file, in tier order.

    Used for ``--dry-run`` (offline, no model) and as the comparison point for Muse's plan:
    everything Muse adds beyond this (grouping, order, risks, verification) is its work."""
    design = {c.name for c in CONSTRUCTS if c.severity == "design"}
    lib_to = {lib["name"]: lib["oneapi"] for lib in scan_report["libraries"]}
    items, oos = [], []
    for f in sorted(scan_report["files"], key=lambda f: (f["tier"], f["lines"], f["path"])):
        if not f["kernels"]:
            oos.append({"files": [f["path"]], "reason": "no kernels (host code or tests)"})
            continue
        items.append({
            "id": Path(f["path"]).stem.replace(".", "_"),
            "title": f"Port {f['path']}",
            "files": [f["path"]],
            "kernels": list(f["kernels"]),  # a copy: the plan must never alias the scan
            "tier": f["tier"],
            "approach": "migrate with SYCLomatic, then replace dpct helpers by hand",
            "library_mappings": [{"from": n, "to": lib_to.get(n, "?"), "notes": ""}
                                 for n in f["libraries"]],
            "risks": [c.replace("_", " ") for c in f["constructs"] if c in design],
            "verification": "compare against the original on the same inputs",
            "depends_on": [],
        })
    return {"summary": "Mechanical baseline plan built from the scan (no model).",
            "items": items, "out_of_scope": oos}


class ScriptedPlanTransport:
    """Fake chat-completions server that answers with ``baseline_plan`` as structured output."""

    def __init__(self, scan_report: dict[str, Any]) -> None:
        self.plan = baseline_plan(scan_report)
        self.requests: list[dict[str, Any]] = []

    def __call__(self, url: str, headers: dict[str, str], body: dict[str, Any]) -> dict[str, Any]:
        self.requests.append(body)
        chars = sum(len(m["content"]) for m in body["messages"])
        text = json.dumps(self.plan)
        return {"model": body["model"],
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": text}}],
                "usage": {"prompt_tokens": chars // CHARS_PER_TOKEN,
                          "completion_tokens": len(text) // CHARS_PER_TOKEN}}


# ----------------------------------------------------------------------------- run


def make_plan(scan_report: dict[str, Any], *, model: str, provider: str = "meta",
              max_output_tokens: int = 32_000, max_cost_usd: float = 1.0,
              transport: Transport | None = None, environ: Mapping[str, str] | None = None,
              confirm: Callable[[float], bool] | None = None) -> PlanResult:
    """Plan the port. Refuses before sending if the worst-case cost exceeds ``max_cost_usd``."""
    sources = collect_sources(scan_report)
    messages = build_messages(scan_report, sources)
    worst = estimate_cost_usd(model, messages, max_output_tokens, environ)
    if worst > max_cost_usd:
        raise SystemExit(f"worst-case cost ${worst:.2f} exceeds the cap ${max_cost_usd:.2f}; "
                         "narrow the scan or raise --max-cost")
    if confirm is not None and not confirm(worst):
        raise SystemExit("cancelled")
    try:
        plan, reported = call_model(messages, model=model, provider=provider,
                                    max_output_tokens=max_output_tokens, transport=transport,
                                    environ=environ)
    except RuntimeError as e:
        raise RuntimeError(f"{e} (the request was sent; its cost is unknown, at most "
                           f"${worst:.2f})") from None
    if reported is None:  # unknown, not free: report it as such
        usage = {k: 0 for k in ("input_tokens", "output_tokens",
                                "cache_creation_input_tokens", "cache_read_input_tokens")}
        return PlanResult(plan, usage, None, model, ground(plan, scan_report),
                          usage_reported=False)
    usage = reported
    try:
        p = model_pricing(model, environ)
        cost: float | None = (usage["input_tokens"] * p.input + usage["output_tokens"] * p.output
                              + usage["cache_read_input_tokens"] * p.cache_read) / 1e6
    except ValueError:
        cost = None
    return PlanResult(plan, usage, cost, model, ground(plan, scan_report))


def render_plan_markdown(result: PlanResult) -> str:
    g, plan = result.grounding, result.plan
    cost = f"${result.cost_usd:.4f}" if result.cost_usd is not None else "unknown"
    out = [f"# Port plan ({result.model})", "", plan.get("summary", ""), "",
           f"- **Grounded:** {'yes' if g['grounded'] else 'NO'} "
           f"(files and kernels it names exist in the repository)",
           f"- **Coverage:** {g['coverage']['covered']}/{g['coverage']['files']} CUDA files "
           f"placed ({g['coverage']['planned']} planned, the rest out of scope)",
           f"- **Issues found by the grounding check:** {g['issues']}",
           (f"- **Tokens:** {result.usage['input_tokens']} in "
            f"({result.usage['cache_read_input_tokens']} cached), "
            f"{result.usage['output_tokens']} out; **cost** {cost}"
            if result.usage_reported else
            "- **Tokens:** not reported by the API; **cost** unknown"), "",
           "## Items", "", "| # | Item | Tier | Files | Kernels | Depends on |",
           "|---|---|---|---|---|---|"]
    for n, it in enumerate(plan.get("items", []), 1):
        out.append(f"| {n} | `{it.get('id')}` {it.get('title', '')} | {it.get('tier')} | "
                   f"{', '.join(it.get('files', []))} | {len(it.get('kernels', []))} | "
                   f"{', '.join(it.get('depends_on', [])) or '-'} |")
    if plan.get("out_of_scope"):
        out += ["", "## Out of scope", ""]
        out += [f"- {', '.join(o.get('files', []))}: {o.get('reason', '')}"
                for o in plan["out_of_scope"]]
    problems = {k: v for k, v in g.items()
                if isinstance(v, list) and v}
    if problems:
        out += ["", "## Grounding findings", ""]
        out += [f"- **{k}**: {json.dumps(v)}" for k, v in problems.items()]
    return "\n".join(out) + "\n"
