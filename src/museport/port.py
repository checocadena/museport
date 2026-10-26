"""``museport port``: the toolkit's agent loop, driven by Muse, on each item of a plan.

For every plan item whose file has a ``KernelSpec`` (``llmc.py``), ``make_task`` writes a
toolkit task directory:

- ``<stem>_harness.cu``: the CUDA the agent ports. It is the original file with three
  mechanical changes: the fp32 build (``#define ENABLE_BF16`` removed), the CPU reference
  functions removed (the agent never sees the answer's code), and llm.c's ``main`` replaced
  by a contract ``main``: ``prog <input> <output> <sizes...>`` reads the inputs, runs every
  kernel variant through the file's own dispatcher and writes each variant's outputs, in
  order, as raw float32. ``nvcc`` builds the same file as the native comparison on a GPU.
- ``<stem>_ref.cpp``: the hidden reference, which the agent never sees: the CPU reference
  functions extracted from the original file plus a generated ``main`` with the same
  contract (its outputs written once per variant).
- ``input.bin``: deterministic inputs, generated like llm.c's ``make_random_*``.
- ``task.json``: the toolkit task (verify with llm.c's own fp32 tolerance).

``check_harness`` proves a task before any model runs: SYCLomatic's raw migration of the
harness, compiled with the dpct helpers, must pass ``verify`` against the reference.
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import struct
from pathlib import Path
from typing import Any

from .llmc import FLT_EPSILON, SPECS, Array, KernelSpec

SEED = 1337


# ----------------------------------------------------------------------------- source surgery


def find_function(src: str, name: str) -> tuple[int, int]:
    """Span of the file-scope definition of ``name`` (from its first line to its closing
    brace and newline)."""
    m = re.search(rf"^(?:static\s+|inline\s+)*(?:void|float|int|double|long)\s*\*?\s*"
                  rf"{re.escape(name)}\s*\(", src, re.M)
    if not m:
        raise ValueError(f"no definition of {name}() found")
    depth, i = 0, m.end() - 1
    while True:  # matching ')' of the parameter list
        depth += {"(": 1, ")": -1}.get(src[i], 0)
        i += 1
        if depth == 0:
            break
    j = src.index("{", i)
    if ";" in src[i:j]:
        raise ValueError(f"{name}() is declared, not defined, at its first match")
    depth, k = 0, j
    while True:
        depth += {"{": 1, "}": -1}.get(src[k], 0)
        k += 1
        if depth == 0:
            break
    if k < len(src) and src[k] == "\n":
        k += 1
    return m.start(), k


def defines(src: str) -> list[str]:
    """``#define`` lines the CPU reference may use (precision switches excluded)."""
    return [ln for ln in src.splitlines()
            if re.match(r"#define\s+\w+", ln) and not re.match(r"#define\s+ENABLE_(BF|FP)16", ln)]


CUBLAS_DECLS = ("static size_t cublaslt_workspace_size", "static void* cublaslt_workspace",
                "static cublasComputeType_t cublas_compute_type", "cublasHandle_t cublas_handle",
                "cublasLtHandle_t cublaslt_handle")


def trim_common(text: str) -> str:
    """llm.c's common.h without its cuBLAS setup. No Tier 1 kernel uses cuBLAS, but its
    headers make SYCLomatic's output depend on oneMKL. Every edit must match, so a changed
    common.h fails loudly instead of being half-trimmed."""
    def sub(pattern: str, repl: str, s: str, flags: int = re.M) -> str:
        out, n = re.subn(pattern, repl, s, count=1, flags=flags)
        if n != 1:
            raise ValueError(f"common.h: expected pattern not found: {pattern!r}")
        return out

    note = "// museport: cuBLAS setup removed (no Tier 1 kernel uses cuBLAS)"
    text = sub(r"^#include <cublas_v2\.h>\n", "", text)
    text = sub(r"^#include <cublasLt\.h>\n", "", text)
    start, end = find_function(text, "cublasCheck")
    text = text[:start] + note + "\n" + text[end:]
    text = sub(r"^#define cublasCheck\(status\).*\n", "", text)
    for decl in CUBLAS_DECLS:
        text = sub(rf"^{re.escape(decl)}\b.*\n", "", text)
    start, end = find_function(text, "setup_main")
    body = text[start:end]
    cut = body.find("    // setup cuBLAS and cuBLASLt")
    if cut < 0:
        raise ValueError("common.h: setup_main has no cuBLAS section")
    return text[:start] + body[:cut].rstrip() + "\n    " + note + "\n}\n" + text[end:]


LT_LINES = (r"#include <cublasLt\.h>", r"static size_t cublaslt_workspace_size\b.*",
            r"static void\* cublaslt_workspace\b.*", r"cublasLtHandle_t cublaslt_handle;.*",
            r"\s*cublasCheck\(cublasLtCreate\(&cublaslt_handle\)\);",
            r"\s*cudaCheck\(cudaMalloc\(&cublaslt_workspace, cublaslt_workspace_size\)\);")


def trim_cublaslt(text: str) -> str:
    """llm.c's common.h without its cuBLASLt setup (cuBLAS kept). SYCLomatic maps cuBLASLt to
    a oneDNN-based helper, so files that never call cuBLASLt should not carry it. Every edit
    must match."""
    for pat in LT_LINES:
        text, n = re.subn(rf"^{pat}\n", "", text, count=1, flags=re.M)
        if n != 1:
            raise ValueError(f"common.h: expected cuBLASLt line not found: {pat!r}")
    return text


def _c_type(a: Array) -> str:
    return "int" if a.dtype == "i32" else "float"


def _args(spec: KernelSpec) -> list[str]:
    n = len(spec.sizes)
    lines = [f"    if (argc != {3 + n}) {{",
             f'        fprintf(stderr, "usage: %s <input> <output> '
             f'{" ".join(s for s, _ in spec.sizes)}\\n", argv[0]);',
             "        return 2;", "    }"]
    lines += [f"    const int {s} = atoi(argv[{3 + i}]);" for i, (s, _) in enumerate(spec.sizes)]
    lines += ['    FILE* fin = fopen(argv[1], "rb");',
              '    if (!fin) { perror(argv[1]); return 2; }']
    for a in spec.inputs:
        t = _c_type(a)
        lines += [f"    {t}* {a.name} = ({t}*)malloc(sizeof({t}) * (size_t)({a.count}));",
                  f"    if (fread({a.name}, sizeof({t}), (size_t)({a.count}), fin) != "
                  f"(size_t)({a.count})) {{",
                  f'        fprintf(stderr, "input too short at {a.name}\\n"); return 2;', "    }"]
    lines += ["    fclose(fin);", '    FILE* fout = fopen(argv[2], "wb");',
              '    if (!fout) { perror(argv[2]); return 2; }']
    return lines


def reference_source(spec: KernelSpec, original: str) -> str:
    funcs = []
    for name in spec.cpu_functions:
        a, b = find_function(original, name)
        funcs.append(original[a:b])
    out = ["// Hidden reference for museport (generated; not shown to the agent).",
           f"// CPU functions extracted verbatim from llm.c dev/cuda/{spec.file}.",
           "#define _USE_MATH_DEFINES", "#include <cfloat>", "#include <cmath>",
           "#include <cstdio>", "#include <cstdlib>", "#include <cstring>", "",
           "typedef float floatX;  // the fp32 build, as in the harness", *defines(original), "",
           *funcs, "int main(int argc, char** argv) {", *_args(spec)]
    if spec.host_prep:
        out.append(f"    {spec.host_prep}")
    for o in spec.outputs:
        out.append(f"    float* {o.name} = (float*)calloc((size_t)({o.count}), sizeof(float));")
        if o.init.startswith("input:"):
            src = o.init.split(":", 1)[1]
            out.append(f"    memcpy({o.name}, {src}, sizeof(float) * (size_t)({o.count}));")
    out.append(f"    {spec.cpu_call};")
    if spec.output_post:
        out.append(f"    {spec.output_post}")
    out.append(f"    for (int r = 0; r < {len(spec.variants)}; r++) {{  // one copy per variant")
    out += [f"        fwrite({o.name}, sizeof(float), (size_t)({o.count}), fout);"
            for o in spec.outputs]
    out += ["    }", "    fclose(fout);", "    return 0;", "}", ""]
    return "\n".join(out)


def harness_source(spec: KernelSpec, original: str) -> str:
    src = re.sub(r"^#define\s+ENABLE_BF16\s*$",
                 "// museport: fp32 build (#define ENABLE_BF16 removed)", original, flags=re.M)
    if spec.extra_includes:
        inc = "".join(f"#include {h}  // museport: needed by the fp32 build\n"
                      for h in spec.extra_includes)
        src, n = re.subn(r'^#include "common\.h"', lambda m: inc + m.group(0), src, count=1,
                         flags=re.M)
        if n != 1:
            raise ValueError(f'{spec.file}: no #include "common.h" to put extra includes before')
    src = re.sub(r"^#include <omp\.h>\s*$",
                 "// museport: #include <omp.h> removed (only the hidden CPU reference used it)",
                 src, flags=re.M)
    hidden = [f for f in spec.cpu_functions if f not in spec.keep_in_harness]
    for name in [*hidden, "main"]:
        start, end = find_function(src, name)
        note = ("" if name == "main" else
                f"// museport: CPU reference {name}() removed (it is the hidden check)\n")
        src = src[:start] + note + src[end:]
    lines = ["", "// " + "-" * 76,
             "// museport contract main: <input> <output> sizes...; runs every kernel variant and",
             "// writes each variant's outputs, in order, as raw float32.",
             "int main(int argc, char** argv) {", "    setup_main();", *_args(spec)]
    if spec.host_prep:
        lines.append(f"    {spec.host_prep}")
    for a in spec.inputs:
        t = _c_type(a)
        lines += [f"    {t}* d_{a.name};",
                  f"    cudaCheck(cudaMalloc(&d_{a.name}, sizeof({t}) * (size_t)({a.count})));",
                  f"    cudaCheck(cudaMemcpy(d_{a.name}, {a.name}, sizeof({t}) * "
                  f"(size_t)({a.count}), cudaMemcpyHostToDevice));"]
    for o in spec.outputs:
        lines += [f"    float* {o.name} = (float*)malloc(sizeof(float) * (size_t)({o.count}));",
                  f"    float* d_{o.name};",
                  f"    cudaCheck(cudaMalloc(&d_{o.name}, sizeof(float) * "
                  f"(size_t)({o.device_count or o.count})));"]
    if spec.gpu_prelude:
        lines.append(f"    {spec.gpu_prelude}")
    lines.append(f"    const int variants[] = {{{', '.join(map(str, spec.variants))}}};")
    lines.append(f"    for (int vi = 0; vi < {len(spec.variants)}; vi++) {{")
    lines.append("        const int k = variants[vi];")
    for o in spec.outputs:
        if o.init.startswith("input:"):
            src_name = o.init.split(":", 1)[1]
            lines.append(f"        cudaCheck(cudaMemcpy(d_{o.name}, {src_name}, sizeof(float) * "
                         f"(size_t)({o.count}), cudaMemcpyHostToDevice));")
        else:
            lines.append(f"        cudaCheck(cudaMemset(d_{o.name}, 0, sizeof(float) * "
                         f"(size_t)({o.count})));")
    lines += [f"        {spec.gpu_call};", "        cudaCheck(cudaGetLastError());",
              "        cudaCheck(cudaDeviceSynchronize());"]
    copy = [f"        cudaCheck(cudaMemcpy({o.name}, d_{o.name}, sizeof(float) * "
            f"(size_t)({o.count}), cudaMemcpyDeviceToHost));" for o in spec.outputs]
    write = [f"        fwrite({o.name}, sizeof(float), (size_t)({o.count}), fout);"
             for o in spec.outputs]
    if spec.output_post:  # all outputs back on the host before the post step
        lines += [*copy, f"        {spec.output_post}", *write]
    else:
        lines += [line for pair in zip(copy, write, strict=True) for line in pair]
    lines += ["    }", "    fclose(fout);", "    return 0;", "}", ""]
    return src.rstrip() + "\n" + "\n".join(lines)


def write_inputs(spec: KernelSpec, path: Path) -> int:
    """Deterministic inputs in spec order, like llm.c's make_random_float (uniform in
    [-1, 1)), make_random_float_01 ([0, 1)) and make_random_int ([0, V))."""
    rng = random.Random(f"{SEED}:{spec.file}")
    sizes = dict(spec.sizes)
    total = 0
    with path.open("wb") as f:
        for a in spec.inputs:
            n = int(eval(a.count, {}, sizes))  # noqa: S307 - expressions come from our specs
            if a.init.startswith("int:"):
                hi = int(eval(a.init[4:], {}, sizes))  # noqa: S307
                f.write(struct.pack(f"<{n}i", *(rng.randrange(hi) for _ in range(n))))
            elif a.init == "one":
                f.write(struct.pack(f"<{n}f", *([1.0] * n)))
            elif a.init == "unit":
                f.write(struct.pack(f"<{n}f", *(rng.random() for _ in range(n))))
            else:
                f.write(struct.pack(f"<{n}f", *(rng.random() * 2 - 1 for _ in range(n))))
            total += 4 * n
    return total


def goal(spec: KernelSpec, harness: str) -> str:
    sizes = " ".join(s for s, _ in spec.sizes)
    return (
        f"Port {harness} (llm.c dev/cuda/{spec.file}, fp32 build) to idiomatic SYCL 2020 with "
        f"zero dpct helpers. Keep every kernel variant k in {list(spec.variants)} as its own "
        "SYCL kernel, computed on the SYCL device, and keep the program's contract: "
        f"`build/candidate <input> <output> {sizes}` reads the inputs and writes each variant's "
        "outputs in order as raw float32, exactly as the CUDA main does. A hidden reference "
        f"checks the output (atol {spec.tol:g} + rtol FLT_EPSILON*|ref|, llm.c's own fp32 "
        "rule). Success requires compile, verify and a zero-helper scan_helpers of the whole "
        "work directory (path '.') on the same sources: migrate's output under migrated/ "
        "counts, so replace or empty those files once your own port is done."
        + (" cuBLAS calls go to oneMath (oneapi::math::blas, #include <oneapi/math.hpp>; the "
           "compile flags already link -lonemath), not to oneapi::mkl product APIs."
           if spec.keep_cublas else "")
    )


def make_task(spec: KernelSpec, cuda_dir: Path, out: Path) -> Path:
    """Write the task directory for ``spec``; returns the task.json path."""
    original = (cuda_dir / spec.file).read_text()
    out.mkdir(parents=True, exist_ok=True)
    common = (cuda_dir / "common.h").read_text()
    if not spec.keep_cublas:
        common = trim_common(common)
    elif not spec.keep_cublaslt:
        common = trim_cublaslt(common)
    (out / "common.h").write_text(common)
    harness = f"{spec.stem}_harness.cu"
    (out / harness).write_text(harness_source(spec, original))
    (out / f"{spec.stem}_ref.cpp").write_text(reference_source(spec, original))
    write_inputs(spec, out / "input.bin")
    args = " ".join(str(v) for _, v in spec.sizes)
    task = {
        "name": f"llmc_{spec.stem}",
        "goal": goal(spec, harness),
        "cuda_path": harness,
        "include_dirs": ["."],
        "reference_build": ["g++", "-O2", "-std=c++17", f"{{task_dir}}/{spec.stem}_ref.cpp",
                            "-o", "build/reference"],
        "reference_cmd": f"build/reference {{input}} {{output}} {args}",
        "candidate_cmd": f"build/candidate {{input}} {{output}} {args}",
        "inputs": ["input.bin"],
        "mode": "array",
        "output_format": "f32",
        **({"base_flags": list(spec.base_flags)} if spec.base_flags else {}),
        # the native nvcc build on a GPU must link the NVIDIA libraries the file calls
        **({"native_flags": ["-lcublas", *(["-lcublasLt"] if spec.keep_cublaslt else [])]}
           if spec.keep_cublas else {}),
        "tolerance": spec.tol,
        "rtol": FLT_EPSILON,
        "museport": {"source": f"karpathy/llm.c dev/cuda/{spec.file}",
                     "variants": list(spec.variants), "sizes": dict(spec.sizes),
                     "note": spec.note,
                     **({"check_flags": list(spec.check_flags)} if spec.check_flags else {})},
    }
    path = out / "task.json"
    path.write_text(json.dumps(task, indent=2) + "\n")
    return path


# ----------------------------------------------------------------------------- self-check


def check_harness(task_file: Path, work: Path) -> dict[str, Any]:
    """No model: c2s migrates the harness, icpx compiles the raw output with the dpct
    helpers, and verify compares it against the hidden reference. A pass shows the spec,
    harness and reference agree; a fail means the task is not usable yet."""
    from oneapi_agent.loop import LoopConfig, Toolbox
    from oneapi_agent.toolchain import run_command

    cfg = LoopConfig.from_task_file(task_file)
    task_meta = json.loads(Path(task_file).read_text()).get("museport", {})
    if work.exists():  # c2s merges with an earlier migration in the same directory
        shutil.rmtree(work)
    work.mkdir(parents=True)
    (work / "build").mkdir(exist_ok=True)
    box = Toolbox(cfg, work)
    steps: dict[str, Any] = {}
    ref = run_command(cfg.reference_build or [], cwd=str(work), timeout=600)
    steps["reference_build"] = {"returncode": ref.returncode, "stderr": ref.stderr[-2000:]}
    if ref.returncode != 0:
        return {"ok": False, "failed": "reference_build", "steps": steps}
    for name in ("migrate", "compile", "verify"):
        args: dict[str, Any] = {}
        if name == "compile":
            migrated = sorted(str(p.relative_to(work)) for p in (work / "migrated").rglob("*")
                              if p.suffix in (".cpp",) and p.name.endswith(".dp.cpp"))
            args = {"sources": migrated, "add_dpct_include": True,
                    "extra_flags": [os.path.expandvars(f) for f in
                                    task_meta.get("check_flags", [])]}
        r = box(name, args)
        steps[name] = {"status": r.status, "summary": r.summary}
        if not r.ok:
            return {"ok": False, "failed": name, "steps": steps,
                    "evidence": json.dumps(r.evidence)[:3000]}
    return {"ok": True, "steps": steps}


def specs_for_plan(plan: dict[str, Any], tier: int | None = 1
                   ) -> list[tuple[dict[str, Any], KernelSpec | None]]:
    """Plan items (of ``tier``) with their harness spec, None when museport has none."""
    out = []
    for it in plan.get("items", []):
        if tier is not None and it.get("tier") != tier:
            continue
        files = [f for f in it.get("files", []) if f.endswith(".cu")]
        if not files:
            continue
        out.append((it, SPECS.get(Path(files[0]).name)))
    return out


# ----------------------------------------------------------------------------- runs


def grade_run(task_file: Path, port_dir: Path, trace: Path, work: Path) -> dict[str, Any]:
    """The toolkit's hidden ``evaluate`` for a finished run, from its copied sources: compile
    with the model's last compile arguments, verify, and scan the whole tree. For runs made
    before ``museport loop`` turned ``evaluate`` on."""
    from oneapi_agent.loop import LoopConfig, Toolbox
    from oneapi_agent.toolchain import run_command
    from oneapi_agent.tools.scan_helpers import scan_helpers

    cfg = LoopConfig.from_task_file(task_file)
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(port_dir, work)
    (work / "build").mkdir(exist_ok=True)
    run_command(cfg.reference_build or [], cwd=str(work), timeout=600)
    last = None
    for line in trace.read_text().splitlines():
        e = json.loads(line)
        if e.get("type") == "tool_call" and e.get("name") == "compile":
            last = e["input"]
    ev: dict[str, Any] = {"compiled": False, "verify_status": None, "verify_summary": None,
                          "helpers": None, "graded": "after the run, by museport grade_run"}
    if last is None:
        ev["note"] = "the model never compiled"
        return ev
    box = Toolbox(cfg, work)
    res = box("compile", last)
    if not res.ok:
        ev["note"] = "final sources do not compile: " + res.summary
        return ev
    ev["compiled"] = True
    v = box("verify", {})
    ev["verify_status"], ev["verify_summary"] = v.status, v.summary
    sc = scan_helpers(str(work))
    ev["helpers"] = int(sc.evidence.get("total", 0)) if sc.status in ("ok", "fail") else None
    return ev

SOURCE_SUFFIXES = (".cpp", ".hpp", ".h")


def collect_port(work: Path, dest: Path) -> list[str]:
    """Copy the agent's sources (not binaries, outputs or snapshots) to ``dest``."""
    out = []
    for p in sorted(work.rglob("*")):
        rel = p.relative_to(work)
        if p.is_file() and p.suffix in SOURCE_SUFFIXES and not {"build", "verify_out"} & set(
                rel.parts):
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(p, dest / rel)
            out.append(str(rel))
    return out


def run_items(plan: dict[str, Any], cuda_dir: Path, out: Path, *, model: str,
              items: list[str] | None = None, max_iterations: int = 12,
              max_cost_usd: float = 1.0, total_cap_usd: float = 20.0,
              runs_dir: Path = Path("runs/port"), tasks_only: bool = False,
              check: bool = False, tier: int = 1, log: Any = print) -> dict[str, Any]:
    """Port each selected item of ``tier`` with Muse. A run starts only if the spend so far plus
    its own cap stays within ``total_cap_usd``."""
    from .loop import run as run_loop

    results: list[dict[str, Any]] = []
    spent = 0.0
    for it, spec in specs_for_plan(plan, tier):
        iid = it.get("id", "?")
        if items and iid not in items:
            continue
        row: dict[str, Any] = {"item": iid, "file": (it.get("files") or ["?"])[0]}
        results.append(row)
        if spec is None:
            row["status"] = "no_harness_spec"
            continue
        d = out / iid
        task = make_task(spec, cuda_dir, d / "task")
        row["task"] = str(task.relative_to(out))
        if check:
            row["harness_check"] = check_harness(task, runs_dir / iid / "check")
            log(f"[port] {iid}: harness check {'ok' if row['harness_check']['ok'] else 'FAIL'}")
            if not row["harness_check"]["ok"]:
                row["status"] = "harness_check_failed"
                continue
        if tasks_only:
            row["status"] = "task_written"
            continue
        if spent + max_cost_usd > total_cap_usd:
            row["status"] = "skipped_total_cap"
            log(f"[port] {iid}: skipped, ${spent:.2f} spent + ${max_cost_usd:.2f} cap > "
                f"${total_cap_usd:.2f}")
            continue
        log(f"[port] {iid}: loop with {model} (max {max_iterations} turns, cap "
            f"${max_cost_usd:.2f}); spent so far ${spent:.4f}")
        try:
            outcome = run_loop(task, model=model, max_iterations=max_iterations,
                               max_cost_usd=max_cost_usd, runs_dir=runs_dir / iid)
        except Exception as e:  # noqa: BLE001 - one item must not end the batch
            partial = trace_spend(runs_dir / iid, model)
            known = partial["cost_usd"] is not None
            row.update(status="error", error=f"{type(e).__name__}: {e}"[:500],
                       iterations=partial["turns"],
                       cost_usd=partial["cost_usd"] if known else max_cost_usd,
                       cost_note=("from the trace's reported usage up to the error" if known
                                  else "no trace: unknown, counted at the cap"),
                       run_id=partial["trace"].parent.name if partial["trace"] else None)
            if partial["trace"]:
                shutil.copy(partial["trace"], d / "trace.jsonl")
            spent += partial["cost_usd"] if partial["cost_usd"] is not None else max_cost_usd
            log(f"[port] {iid}: error {type(e).__name__}: {e}")
            write_summary(out, plan, model, results)
            continue
        run_dir = Path(outcome["run_dir"])
        for stale in ("port", "best"):  # a rerun must not leave the previous run's files
            if (d / stale).exists():
                shutil.rmtree(d / stale)
        for name in ("report.md", "trace.jsonl"):
            shutil.copy(run_dir / name, d / name)
        row.update(run_id=run_dir.name, status=outcome["stop_reason"],
                   detail=outcome["stop_detail"],
                   iterations=outcome["iterations"], usage=outcome["usage"],
                   cost_usd=outcome["cost_usd"], best=outcome.get("best"),
                   evaluation=outcome.get("evaluation"),
                   sources=collect_port(run_dir / "work", d / "port"))
        if outcome.get("best"):
            collect_port(Path(outcome["best"]["path"]), d / "best")
        row["variant_check"] = variant_check(d / "port",
                                             last_compiled_sources(d / "trace.jsonl"),
                                             len(spec.variants))
        spent += outcome["cost_usd"] if outcome["cost_usd"] is not None else max_cost_usd
        log(f"[port] {iid}: {row['status']} in {row['iterations']} turns, "
            f"${row['cost_usd'] or 0:.4f}")
        write_summary(out, plan, model, results)
    return write_summary(out, plan, model, results)


def write_summary(out: Path, plan: dict[str, Any], model: str,
                  results: list[dict[str, Any]]) -> dict[str, Any]:
    """Write summary.json and SUMMARY.md (after every item, so a crash loses nothing), keeping
    the rows of earlier batches in the same directory."""
    out.mkdir(parents=True, exist_ok=True)
    prev_path = out / "summary.json"
    extra: dict[str, Any] = {}
    results = [dict(r) for r in results]
    if prev_path.exists():  # earlier batches in the same directory: keep their rows
        prev = json.loads(prev_path.read_text())
        extra = {k: v for k, v in prev.items() if k not in ("model", "spent_usd", "items")}
        old = {r["item"]: r for r in prev.get("items", [])}
        for i, r in enumerate(results):
            o = old.get(r["item"])
            if not o or o.get("cost_usd") is None or o is r:
                continue
            if "cost_usd" not in r:  # tasks only / check only: keep the paid row's results
                results[i] = {**o, **{k: v for k, v in r.items() if k == "harness_check"}}
            elif o.get("run_id") != r.get("run_id"):
                # a rerun: the earlier paid run stays on record and in the spend
                earlier = [*o.get("earlier_runs", []),
                           {k: o.get(k) for k in ("status", "iterations", "cost_usd", "error")
                            if o.get(k) is not None}]
                r["earlier_runs"] = [*r.get("earlier_runs", []), *earlier]
        ran = {r["item"] for r in results}
        kept = [r for r in prev.get("items", []) if r["item"] not in ran]
        order = [it.get("id") for it, _ in specs_for_plan(plan, None)]
        results = sorted([*kept, *results], key=lambda r: order.index(r["item"])
                         if r["item"] in order else len(order))
    total = sum((r.get("cost_usd") or 0.0) + sum((e.get("cost_usd") or 0.0)
                                                 for e in r.get("earlier_runs", []))
                for r in results)
    summary = {"model": model, "spent_usd": round(total, 4), **extra, "items": results}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "SUMMARY.md").write_text(render_summary(summary))
    return summary


def trace_spend(item_runs: Path, model: str) -> dict[str, Any]:
    """Turns and cost of the newest run under ``item_runs``, from its trace's model responses
    (for a run that ended in an error, so no outcome was written)."""
    from oneapi_agent.loop import model_pricing

    traces = sorted(item_runs.glob("*/trace.jsonl"))
    if not traces:
        return {"turns": 0, "cost_usd": None, "trace": None}
    p = model_pricing(model)
    turns, cost = 0, 0.0
    for line in traces[-1].read_text().splitlines():
        e = json.loads(line)
        if e.get("type") == "model_response":
            u = e.get("usage", {})
            turns += 1
            cost += (u.get("input_tokens", 0) * p.input + u.get("output_tokens", 0) * p.output
                     + u.get("cache_read_input_tokens", 0) * p.cache_read) / 1e6
    return {"turns": turns, "cost_usd": round(cost, 6), "trace": traces[-1]}


def _variants(vc: dict[str, Any] | None) -> str:
    if not vc:
        return "-"
    flag = " **duplicated**" if vc["duplicated_variants"] else ""
    lib = f" + {vc['onemath_calls']} oneMath" if vc.get("onemath_calls") else ""
    return f"{vc['distinct_kernel_bodies']}{lib} / {vc['variants']}{flag}"


def _grade(ev: dict[str, Any] | None) -> str:
    if not ev:
        return "-"
    if not ev.get("compiled"):
        return f"does not compile ({ev.get('note', '')})"
    return (f"verify {ev.get('verify_status')}: {ev.get('verify_summary')}; "
            f"{ev.get('helpers')} helpers")


def render_summary(summary: dict[str, Any]) -> str:
    lines = [f"# museport port ({summary['model']})", "",
             f"Spent: ${summary['spent_usd']:.4f} (every run, including earlier runs of an item "
             "and runs that ended in an error, counted from their reported usage)", "",
             "| Item | File | Result | Turns | Cost | Best passing (verify) | Hidden grade of "
             "final sources | Distinct kernels / variants |",
             "|---|---|---|---|---|---|---|---|"]
    for r in summary["items"]:
        best = r.get("best") or {}
        lines.append(
            f"| `{r['item']}` | {r['file']} | {r['status']} | {r.get('iterations', '-')} | "
            + (f"${r['cost_usd']:.4f}" if r.get("cost_usd") is not None else "-") + " | "
            + (f"{best.get('verify_summary')} ({best.get('helpers')} helpers)" if best else "-")
            + " | " + _grade(r.get("evaluation")) + " | " + _variants(r.get("variant_check"))
            + " |")
    earlier = [(r["item"], e) for r in summary["items"] for e in r.get("earlier_runs", [])]
    if earlier:
        lines += ["", "Earlier runs (counted in the spend):", ""]
        lines += [f"- `{item}`: {e.get('status')}"
                  + (f" ({e['error']})" if e.get("error") else "")
                  + (f", {e['iterations']} turns" if e.get("iterations") is not None else "")
                  + (f", ${e['cost_usd']:.4f}" if e.get("cost_usd") is not None else "")
                  for item, e in earlier]
    lines += ["", "`success`: compile, verify against the hidden reference and a zero-helper "
              "scan of the whole work tree, all on the same final sources (the toolkit's rule).",
              "Hidden grade: the toolkit's `evaluate`, run after the loop on the final sources "
              "(never shown to the model).",
              "Distinct kernels / variants: distinct `parallel_for` bodies (plus oneMath calls, "
              "which replace cuBLAS variants) in the final sources against the variants the "
              "port must keep; fewer means variants were copied (outputs can still be "
              "correct)."]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- variant check


def kernel_bodies(src: str) -> list[str]:
    """The body of every ``parallel_for`` kernel lambda in ``src``, comments and whitespace
    dropped and numbered kernel/variant names replaced, so copies compare equal."""
    out = []
    for m in re.finditer(r"parallel_for\s*(?:<[^>]*>)?\s*\(", src):
        lam = src.find("[=]", m.end())
        start = src.find("{", lam) if lam >= 0 else -1
        if start < 0:
            continue
        depth, k = 0, start
        while k < len(src):
            depth += {"{": 1, "}": -1}.get(src[k], 0)
            k += 1
            if depth == 0:
                break
        body = re.sub(r"//.*", "", src[start:k])
        body = re.sub(r"\b\w*(?:[Kk]ernel|[Vv]ariant|_v|V)\d+\w*\b", "X", body)
        out.append(re.sub(r"\s+", " ", body))
    return out


def variant_check(port_dir: Path, sources: list[str], variants: int) -> dict[str, Any]:
    """Did the port keep its variants apart? The hidden check compares outputs only, so a
    port that copies one implementation into every variant still passes it. Fewer distinct
    kernel bodies than variants flags that (it cannot prove a variant keeps llm.c's
    algorithm, only that variants were not duplicated)."""
    src = "".join((port_dir / s).read_text() for s in sources if (port_dir / s).exists())
    bodies = kernel_bodies(src)
    distinct = len(set(bodies))
    # a cuBLAS variant becomes a oneMath call, not a kernel: count those calls too
    library = len(re.findall(r"oneapi::math::blas::\w+::\w+\s*\(", src))
    return {"variants": variants, "kernels": len(bodies), "distinct_kernel_bodies": distinct,
            "onemath_calls": library, "duplicated_variants": distinct + library < variants}


# ----------------------------------------------------------------------------- GPU tasks


def last_compiled_sources(trace: Path) -> list[str]:
    """The sources of the model's last compile call (paths relative to its work dir)."""
    last: list[str] = []
    for line in trace.read_text().splitlines():
        e = json.loads(line)
        if e.get("type") == "tool_call" and e.get("name") == "compile":
            last = list(e["input"].get("sources", []))
    return last


def make_gpu_task(item_dir: Path, dest: Path, model: str) -> Path:
    """A ``remote-test`` task for a finished port: the item's task (harness, hidden reference,
    inputs) plus ``sycl_sources`` pointing at the agent's last compiled sources, unchanged."""
    srcs = last_compiled_sources(item_dir / "trace.jsonl")
    if not srcs:
        raise ValueError(f"{item_dir.name}: the model never compiled")
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(item_dir / "task", dest)
    shutil.copytree(item_dir / "port", dest / "agent_port")
    task = json.loads((dest / "task.json").read_text())
    if not (dest / "input.bin").exists():
        raise ValueError(f"{item_dir.name}: task/input.bin missing; rerun make_task")
    task["sycl_sources"] = [f"agent_port/{s}" for s in srcs]
    task["port_origin"] = (f"agent-made: {model} via museport port, item {item_dir.name}; "
                           "sources as last compiled by the agent, unchanged")
    (dest / "task.json").write_text(json.dumps(task, indent=2) + "\n")
    return dest / "task.json"
