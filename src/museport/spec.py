"""``museport spec``: Muse writes a file's harness spec, and the spec is proven (plan item J2).

Every hidden check so far comes from a ``KernelSpec`` written by hand in ``llmc.py``: which CPU
functions are the reference, which kernel variants the port must keep, the inputs and sizes,
and the two calls. ``harness_source`` and ``reference_source`` then build the CUDA contract
harness and the hidden reference from llm.c's file mechanically. Here Muse writes the spec
from the CUDA file alone, and nothing it writes is trusted until it is checked:

1. Static checks, no compiler: the CPU functions exist, the variants are the dispatcher's
   ``case`` labels (any left out need a reason), the sizes and counts evaluate, the buffers
   fit in memory.
2. The proof, as for the hand-written specs (``check_harness``): SYCLomatic migrates the
   generated CUDA harness, icpx compiles the raw output, and it must match the reference
   built from llm.c's own CPU functions. llm.c's GPU code and its CPU code must agree on the
   generated inputs.
3. When SYCLomatic's output does not migrate, compile or match (it fails on 5 of llm.c's 19
   files even with the hand-written specs), the GPU proof (``gpu_proof``, opt-in): llm.c's
   harness built with ``nvcc`` on a rented NVIDIA GPU, its outputs recorded (plan item J1)
   and compared here with the same CPU reference. Only when both fail is the spec wrong.

A failed proof goes back to Muse as feedback, for a few attempts. The result is compared with
the hand-written spec where one exists.
"""

from __future__ import annotations

import ast
import json
import re
import shutil
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from oneapi_agent.loop import model_pricing

from .llmc import ONEMATH_FLAGS, ONEMKL_CHECK_FLAGS, SPECS, Array, KernelSpec
from .plan import CHARS_PER_TOKEN, call_model
from .port import check_harness, find_function, make_task

_ARRAY = {"type": "object", "additionalProperties": False,
          "required": ["name", "count", "dtype", "init", "device_count"],
          "properties": {"name": {"type": "string"}, "count": {"type": "string"},
                         "dtype": {"type": "string", "enum": ["f32", "i32"]},
                         "init": {"type": "string"},
                         "device_count": {"type": "string"}}}
_STRS = {"type": "array", "items": {"type": "string"}}

SPEC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["cpu_functions", "variants", "dropped_variants", "sizes", "inputs", "outputs",
                 "cpu_call", "gpu_call", "tol", "note", "host_prep", "gpu_prelude",
                 "keep_in_harness", "extra_includes", "keep_cublas", "keep_cublaslt",
                 "output_post"],
    "properties": {
        "cpu_functions": _STRS,
        "variants": {"type": "array", "items": {"type": "integer"}},
        "dropped_variants": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["variant", "reason"],
            "properties": {"variant": {"type": "integer"}, "reason": {"type": "string"}}}},
        "sizes": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["name", "value"],
            "properties": {"name": {"type": "string"}, "value": {"type": "integer"}}}},
        "inputs": {"type": "array", "items": _ARRAY},
        "outputs": {"type": "array", "items": _ARRAY},
        "cpu_call": {"type": "string"}, "gpu_call": {"type": "string"},
        "tol": {"type": "number"}, "note": {"type": "string"},
        "host_prep": {"type": "string"}, "gpu_prelude": {"type": "string"},
        "keep_in_harness": _STRS, "extra_includes": _STRS,
        "keep_cublas": {"type": "boolean"}, "keep_cublaslt": {"type": "boolean"},
        "output_post": {"type": "string"},
    },
}

SYSTEM = """\
You write the harness spec for one CUDA file from llm.c's dev/cuda. The spec builds two \
programs mechanically, and a model that ports the file to SYCL is graded by comparing them:

- The hidden reference: the file's CPU functions you name (cpu_functions), copied verbatim, \
plus a main that reads the inputs, runs cpu_call once, applies output_post, and writes the \
outputs once per variant.
- The CUDA contract harness: the file with those CPU functions and its own main removed \
(functions in keep_in_harness stay), plus a main that reads the same inputs, copies each \
input to a device buffer d_<name>, allocates each output and its device buffer d_<name>, runs \
gpu_prelude, then for each variant k: resets every output on the device (to zeros, or to the \
input it names), runs gpu_call, copies outputs back, applies output_post, writes them.

Both mains take `<input> <output> sizes...`: sizes are ints parsed in order into consts with \
your names. Inputs are generated once, in order, into one file; host_prep is C run in both mains \
right after the inputs are read. Counts are C expressions over the size names. setup_main() \
(common.h) runs first in the harness.

Fields:
- cpu_functions: the CPU reference functions the check needs (they leave the harness).
- variants: every kernel_num the file's dispatcher runs in the fp32 build. Any dispatcher case \
you leave out goes in dropped_variants with the reason (e.g. bf16-only).
- sizes: reduce llm.c's main sizes so the CPU reference and a CPU SYCL device finish in seconds \
and all buffers stay under about 200 MB; keep the dimensions the kernels assume (C, V, \
alignments, asserts like C % x == 0). Say what changed in note.
- inputs: init is uniform ([-1, 1), llm.c's make_random_float), unit ([0, 1), \
make_random_float_01), one (all 1.0), or int:<C expr> (ints in [0, expr), make_random_int). \
Follow how llm.c's main makes each input. dtype f32 or i32. device_count "" (inputs).
- outputs: f32 only; init zeros or input:<input name> (an in-place update starts from that \
input). device_count: "" unless the kernel writes past count (then the device size).
- cpu_call: C using host names, as llm.c's main calls the CPU reference. Several statements \
may be joined with "; ".
- gpu_call: C using device pointers d_<name> and the variant k, as llm.c's main calls the \
dispatcher (block size 256 unless main uses another).
- tol: the tolerance llm.c's main passes to validate_result (its default is 1e-4 when omitted).
- host_prep, gpu_prelude, output_post: C, "" when not needed. Use them for what llm.c's main \
does that the template does not: scaling inputs, outlier injection, device properties, \
scratch buffers, cuBLAS math mode, masking elements validate_result skips.
- keep_cublas / keep_cublaslt: true when the file calls cuBLAS / cuBLASLt.
- extra_includes: headers (e.g. "<cuda_bf16.h>") the fp32 build needs before common.h; \
usually empty.

Match llm.c's own check: the same CPU function, the same inputs and tolerance, all fp32 variants. \
The spec is proven by migrating the harness with SYCLomatic, compiling it, and comparing with \
the reference on a CPU device; if a step fails you will be told and asked for a corrected spec.\
"""

# Two hand-written specs shown to Muse as worked examples; not graded.
EXAMPLES = ("gelu_forward.cu", "global_norm.cu")
MAX_OUTPUT_TOKENS = 16_000
MAX_ATTEMPTS = 3


def spec_to_answer(spec: KernelSpec) -> dict[str, Any]:
    """A hand-written spec in the answer schema (for the examples)."""
    d = asdict(spec)

    def arr(a: dict[str, Any]) -> dict[str, Any]:
        return {**a, "device_count": a["device_count"] or ""}

    return {"cpu_functions": list(spec.cpu_functions), "variants": list(spec.variants),
            "dropped_variants": [], "sizes": [{"name": n, "value": v} for n, v in spec.sizes],
            "inputs": [arr(a) for a in d["inputs"]], "outputs": [arr(a) for a in d["outputs"]],
            **{k: d[k] for k in ("cpu_call", "gpu_call", "tol", "note", "host_prep",
                                 "gpu_prelude", "output_post", "keep_cublas", "keep_cublaslt")},
            "keep_in_harness": list(spec.keep_in_harness),
            "extra_includes": list(spec.extra_includes)}


def to_kernel_spec(file: str, ans: dict[str, Any], src: str) -> KernelSpec:
    def arr(a: dict[str, Any]) -> Array:
        return Array(a["name"], a["count"], a["dtype"], a["init"], a["device_count"] or None)

    cublas = re.search(r"^\s*#include\s*<cublas", src, re.M) is not None
    return KernelSpec(
        file=file, cpu_functions=tuple(ans["cpu_functions"]), variants=tuple(ans["variants"]),
        sizes=tuple((s["name"], s["value"]) for s in ans["sizes"]),
        inputs=tuple(arr(a) for a in ans["inputs"]), outputs=tuple(arr(a) for a in ans["outputs"]),
        cpu_call=ans["cpu_call"], gpu_call=ans["gpu_call"], tol=ans["tol"], note=ans["note"],
        host_prep=ans["host_prep"], gpu_prelude=ans["gpu_prelude"],
        keep_in_harness=tuple(ans["keep_in_harness"]),
        extra_includes=tuple(ans["extra_includes"]), keep_cublas=ans["keep_cublas"],
        keep_cublaslt=ans["keep_cublaslt"], output_post=ans["output_post"],
        # the build flags follow from the answer: oneMath for the port when the file calls
        # cuBLAS, oneMKL for SYCLomatic's raw output whenever the file includes cuBLAS
        base_flags=ONEMATH_FLAGS if ans["keep_cublas"] else (),
        check_flags=ONEMKL_CHECK_FLAGS if cublas else ())


def dispatcher_variants(src: str) -> list[int]:
    """The ``case`` labels of the file's ``switch (kernel_num)`` statements."""
    out: set[int] = set()
    for m in re.finditer(r"switch\s*\(\s*kernel_num\s*\)\s*\{", src):
        depth, i = 1, m.end()
        while depth and i < len(src):
            depth += {"{": 1, "}": -1}.get(src[i], 0)
            i += 1
        out |= {int(c) for c in re.findall(r"\bcase\s+(\d+)\s*:", src[m.end():i])}
    return sorted(out)


def _eval(expr: str, sizes: dict[str, int]) -> int:
    """A count expression over the size names: integers, names, + - * / % and parentheses."""
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Expression, ast.BinOp, ast.Constant, ast.Name, ast.Load,
                                 ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
                                 ast.UnaryOp, ast.USub)):
            raise ValueError(f"unsupported expression {expr!r}")
        if isinstance(node, ast.Name) and node.id not in sizes:
            raise ValueError(f"{expr!r}: unknown size {node.id!r}")
    return int(eval(compile(tree, "<count>", "eval"), {}, dict(sizes)))  # noqa: S307 - checked


MAX_BYTES = 512 * 1024 * 1024
IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def static_errors(ans: dict[str, Any], src: str) -> list[str]:
    """Everything that can be checked without a compiler."""
    errs = []
    for f in ans["cpu_functions"] + ans["keep_in_harness"]:
        try:
            find_function(src, f)
        except ValueError:
            errs.append(f"cpu function {f!r} is not defined in the file")
    if not ans["cpu_functions"]:
        errs.append("no cpu_functions: the hidden reference needs at least one")
    disp = dispatcher_variants(src)
    dropped = {d["variant"] for d in ans["dropped_variants"]}
    if disp:
        extra = sorted(set(ans["variants"]) - set(disp))
        missing = sorted(set(disp) - set(ans["variants"]) - dropped)
        if extra:
            errs.append(f"variants {extra} are not cases of the dispatcher (cases: {disp})")
        if missing:
            errs.append(f"dispatcher cases {missing} are neither in variants nor in "
                        "dropped_variants with a reason")
    if not ans["variants"]:
        errs.append("no variants")
    sizes = {s["name"]: s["value"] for s in ans["sizes"]}
    names = [*sizes, *(a["name"] for a in ans["inputs"] + ans["outputs"])]
    errs += [f"{n!r} is not a C identifier" for n in names if not IDENT.match(n)]
    errs += [f"name {n!r} is used twice" for n in sorted({n for n in names if names.count(n) > 1})]
    errs += [f"size {n} = {v} must be positive" for n, v in sizes.items() if v <= 0]
    total = 0
    for a in ans["inputs"] + ans["outputs"]:
        try:
            n = _eval(a["count"], sizes)
            if a["init"].startswith("int:"):
                _eval(a["init"][4:], sizes)
        except (ValueError, SyntaxError, ZeroDivisionError) as e:
            errs.append(f"{a['name']}: {e}")
            continue
        if n <= 0:
            errs.append(f"{a['name']}: count {a['count']} = {n}")
        total += 4 * max(n, 0)
    inputs = {a["name"] for a in ans["inputs"]}
    for a in ans["inputs"]:
        if not (a["init"] in ("uniform", "unit", "one") or a["init"].startswith("int:")):
            errs.append(f"input {a['name']}: init {a['init']!r} is not uniform, unit, one or int:")
    for a in ans["outputs"]:
        if a["dtype"] != "f32":
            errs.append(f"output {a['name']}: outputs are f32")
        if not (a["init"] == "zeros" or a["init"].split(":", 1)[-1] in inputs
                and a["init"].startswith("input:")):
            errs.append(f"output {a['name']}: init {a['init']!r} is not zeros or input:<input>")
    if total > MAX_BYTES:
        errs.append(f"buffers total {total / 2**20:.0f} MB, over {MAX_BYTES // 2**20} MB: "
                    "reduce the sizes")
    return errs


def messages_for(file: str, src: str, common: str) -> list[dict[str, str]]:
    ex = "\n\n".join(f"Example: the spec for {f}:\n{json.dumps(spec_to_answer(SPECS[f]))}"
                     for f in EXAMPLES)
    return [{"role": "system", "content": SYSTEM + "\n\n" + ex},
            {"role": "user", "content": f"common.h:\n```cuda\n{common}\n```\n\n"
             f"Write the spec for {file}:\n```cuda\n{src}\n```"}]


def worst_case_usd(model: str, messages: list[dict[str, str]],
                   environ: Mapping[str, str] | None = None) -> float:
    p = model_pricing(model, environ)
    tokens_in = sum(len(m["content"]) for m in messages) / CHARS_PER_TOKEN
    return (tokens_in * p.input + MAX_OUTPUT_TOKENS * p.output) / 1e6


def compare_with_hand(spec: KernelSpec, hand: KernelSpec | None) -> dict[str, Any] | None:
    if hand is None:
        return None
    return {"variants": list(spec.variants) == list(hand.variants),
            "cpu_functions": set(spec.cpu_functions) == set(hand.cpu_functions),
            "sizes": spec.sizes == hand.sizes, "tol": spec.tol == hand.tol,
            "hand_variants": list(hand.variants), "hand_tol": hand.tol}


def _keep_gpu_evidence(work: Path, dest: Path) -> None:
    """The GPU proof's logs and results, without the recorded outputs (large; sha256 kept)."""
    dest.mkdir(parents=True, exist_ok=True)
    if (work / "remote-test.log").exists():
        shutil.copyfile(work / "remote-test.log", dest / "remote-test.log")
    for f in sorted((work / "res").glob("*/*/*")):
        if f.is_file():
            shutil.copyfile(f, dest / f.name)


def _feedback(static: list[str], check: dict[str, Any] | None,
              gpu: dict[str, Any] | None = None) -> str:
    if static:
        return "The spec fails these checks:\n- " + "\n- ".join(static) + \
               "\nReturn a corrected spec."
    assert check is not None
    step = check.get("failed")
    detail = check["steps"].get(step, {})
    text = detail.get("summary") or detail.get("stderr") or ""
    ev = check.get("evidence", "")
    return (f"The proof failed at step {step!r}: {str(text)[:1500]}\n"
            + (f"Details: {ev[:2500]}\n" if ev else "")
            + (f"llm.c's CUDA built with nvcc on an NVIDIA GPU also failed, at {gpu['step']}: "
               f"{gpu['summary'][:1500]}\n" if gpu else "")
            + "Return a corrected spec.")


# CPU-proof failures that SYCLomatic itself can cause, so the GPU proof is tried before
# telling the model its spec is wrong
GPU_PROOF_STEPS = ("migrate", "compile", "verify")
GpuProof = Callable[[Path, Path], dict[str, Any]]


def gpu_proof(task_dir: Path, work: Path, *, max_usd: float = 0.10,
              runner: Any = subprocess.run) -> dict[str, Any]:
    """llm.c's CUDA harness built with ``nvcc`` on a rented NVIDIA GPU (Runpod), its outputs
    recorded twice, and compared here with the CPU reference built from llm.c's own CPU code.
    Needs the task's ``input.bin`` and the GPU provider's key."""
    from .record import check_recording, differential_task, recording_task

    if work.exists():
        shutil.rmtree(work)
    rec = recording_task(task_dir, work / "rec")
    name = json.loads(rec.read_text())["name"]
    proc = runner(["oneapi-agent", "remote-test", "task.json", "--provider", "runpod",
                   "--vendor", "nvidia", "--max-usd", str(max_usd), "--max-minutes", "20",
                   "--into", str(work / "res")], cwd=str(rec.parent), capture_output=True,
                  text=True, timeout=1800)
    text = (proc.stdout or "") + (proc.stderr or "")
    (work / "remote-test.log").write_text(text)
    costs = re.findall(r'"est_cost_usd":\s*([0-9.]+)', text)
    out: dict[str, Any] = {"ok": False, "cost_usd": float(costs[-1]) if costs else max_usd}
    found = sorted((work / "res").glob(f"*/{name}/summary.json"))
    if not found:
        return {**out, "step": "record", "summary": text[-1500:]}
    summary = json.loads(found[-1].read_text())
    ev = summary.get("evidence") or {}
    out.update(device=(ev.get("device") or {}).get("name"), deterministic=ev.get("deterministic"),
               results=str(found[-1].parent))
    try:
        differential_task(task_dir, found[-1].parent, work / "diff")
    except ValueError:
        steps = "; ".join(f"{s['step']}: {s['summary']}" for s in ev.get("steps", []))
        return {**out, "step": "record", "summary": f"{summary.get('summary')} ({steps})"[:2000]}
    r = check_recording(task_dir, work / "diff", work / "cpu")
    return {**out, "ok": r["status"] == "ok", "step": "compare", "summary": r["summary"]}


def write_spec(file: str, cuda_dir: Path, out: Path, *, model: str, provider: str = "meta",
               transport: Any = None, cap_usd: float = 0.5, prove_on_gpu: GpuProof | None = None,
               log: Any = print) -> dict[str, Any]:
    """Muse writes ``file``'s spec, which is checked and proven; up to MAX_ATTEMPTS answers."""
    # absolute paths: the toolkit refuses tool paths that resolve outside a relative work dir
    cuda_dir, out = cuda_dir.resolve(), out.resolve()
    src = (cuda_dir / file).read_text()
    stem = file.removesuffix(".cu")
    item = out / stem
    if item.exists():
        shutil.rmtree(item)
    item.mkdir(parents=True)
    msgs = messages_for(file, src, (cuda_dir / "common.h").read_text())
    p = model_pricing(model)
    attempts: list[dict[str, Any]] = []
    spent, gpu_spent, final, proven_by = 0.0, 0.0, None, None
    for n in range(1, MAX_ATTEMPTS + 1):
        worst = worst_case_usd(model, msgs)
        if spent + worst > cap_usd:
            log(f"[spec] {file}: attempt {n} skipped (${spent:.2f} + ${worst:.2f} > ${cap_usd})")
            break
        try:
            ans, usage = call_model(msgs, model=model, provider=provider,
                                    max_output_tokens=MAX_OUTPUT_TOKENS, transport=transport,
                                    schema_name="harness_spec", schema=SPEC_SCHEMA)
        except RuntimeError as e:
            spent += worst
            attempts.append({"attempt": n, "error": str(e)[:500], "cost_usd": worst})
            log(f"[spec] {file}: attempt {n} error {e}")
            break
        cost = ((usage["input_tokens"] * p.input + usage["output_tokens"] * p.output
                 + usage["cache_read_input_tokens"] * p.cache_read) / 1e6 if usage else worst)
        spent += cost
        static = static_errors(ans, src)
        check = gpu = None
        if not static:
            spec = to_kernel_spec(file, ans, src)
            task = make_task(spec, cuda_dir, item / "task")
            check = check_harness(task, item / ".work")
            if not check["ok"] and check.get("failed") in GPU_PROOF_STEPS and prove_on_gpu:
                gpu = prove_on_gpu(item / "task", item / ".gpu")
                gpu_spent += gpu["cost_usd"]
                _keep_gpu_evidence(item / ".gpu", item / "gpu-proof" / f"attempt{n}")
        attempts.append({"attempt": n, "answer": ans, "usage": usage, "cost_usd": round(cost, 6),
                         "static_errors": static, "check": check, "gpu_proof": gpu})
        by = ("cpu" if check and check["ok"] else "gpu" if gpu and gpu["ok"] else None)
        log(f"[spec] {file}: attempt {n}: "
            + (f"proven ({by.upper()})" if by else f"static: {static}" if static else
               f"proof failed at {check and check.get('failed')}"
               + (f"; GPU proof failed at {gpu['step']}" if gpu else ""))
            + f", ${cost:.4f}" + (f" + GPU ${gpu['cost_usd']:.3f}" if gpu else ""))
        if by:
            final, proven_by = ans, by
            break
        msgs = [*msgs, {"role": "assistant", "content": json.dumps(ans)},
                {"role": "user", "content": _feedback(static, check, gpu)}]
    shutil.rmtree(item / ".work", ignore_errors=True)
    shutil.rmtree(item / ".gpu", ignore_errors=True)
    (item / "task" / "input.bin").unlink(missing_ok=True)  # regenerated from the spec
    row: dict[str, Any] = {"file": file, "proven": final is not None, "proven_by": proven_by,
                           "attempts": len(attempts), "cost_usd": round(spent, 6),
                           "gpu_usd": round(gpu_spent, 4)}
    if final is not None:
        spec = to_kernel_spec(file, final, src)
        (item / "spec.json").write_text(json.dumps(final, indent=2) + "\n")
        row["compare"] = compare_with_hand(spec, SPECS.get(file))
        row["variants"] = list(spec.variants)
        row["dropped_variants"] = final["dropped_variants"]
    elif (item / "task").exists():
        shutil.rmtree(item / "task")  # an unproven task is not evidence of anything
    (item / "attempts.json").write_text(json.dumps(attempts, indent=2) + "\n")
    return row


def run_specs(files: list[str], cuda_dir: Path, out: Path, *, model: str,
              provider: str = "meta", total_cap_usd: float = 2.0, file_cap_usd: float = 0.4,
              prove_on_gpu: GpuProof | None = None, transport: Any = None,
              log: Any = print) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    prev = out / "summary.json"
    rows: list[dict[str, Any]] = ([r for r in json.loads(prev.read_text())["files"]
                                   if r["file"] not in files] if prev.exists() else [])
    spent = 0.0
    for f in files:
        cap = min(file_cap_usd, total_cap_usd - spent)
        if cap <= 0:
            rows.append({"file": f, "proven": False, "skipped": "total cap"})
            continue
        row = write_spec(f, cuda_dir, out, model=model, provider=provider, transport=transport,
                         cap_usd=cap, prove_on_gpu=prove_on_gpu, log=log)
        spent += row["cost_usd"]
        rows.append(row)
        _write(out, model, rows)
    return _write(out, model, rows)


def _write(out: Path, model: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    order = {f: i for i, f in enumerate(SPECS)}
    rows = sorted(rows, key=lambda r: order.get(r["file"], len(order)))
    summary = {"model": model, "spent_usd": round(sum(r.get("cost_usd", 0) for r in rows), 4),
               "gpu_usd": round(sum(r.get("gpu_usd", 0) for r in rows), 4),
               "examples": list(EXAMPLES), "files": rows}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "SUMMARY.md").write_text(render_summary(summary))
    return summary


def render_summary(summary: dict[str, Any]) -> str:
    md = ["# Harness specs written by Muse and proven", "",
          f"Model `{summary['model']}`, spent ${summary['spent_usd']:.4f} (GPU proofs "
          f"${summary.get('gpu_usd', 0):.3f}). Worked examples shown "
          f"to Muse (not graded): {', '.join(summary['examples'])}.", "",
          "| file | proven | attempts | variants | same as hand-written: variants / CPU functions "
          "/ sizes / tol | cost |", "|---|---|---|---|---|---|"]
    for r in summary["files"]:
        c = r.get("compare")
        same = (" / ".join("yes" if c[k] else "no" for k in ("variants", "cpu_functions",
                                                              "sizes", "tol"))
                if c else "no hand-written spec")
        proven = f"yes ({(r.get('proven_by') or 'cpu').upper()})" if r["proven"] else "no"
        md.append(f"| {r['file']} | {proven} | {r.get('attempts', 0)} "
                  f"| {r.get('variants', '')} | {same if r['proven'] else ''} "
                  f"| ${r.get('cost_usd', 0):.4f} |")
    return "\n".join(md) + "\n"
