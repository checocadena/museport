"""Deterministic inventory of a CUDA repository: the ground truth a Muse plan is checked against.

``scan(root)`` walks a repository and reports, per file and in total: CUDA kernels, NVIDIA
library use (with the oneAPI equivalent), and CUDA-only constructs that need a deliberate
SYCL design (warp size 32, warp shuffles, cooperative groups, bf16, inline PTX, ...). It reads
text only; nothing is compiled or run, and no model is involved.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

CUDA_SUFFIXES = (".cu", ".cuh")
HEADER_SUFFIXES = (".h", ".hpp", ".hh", ".inl")
SKIP_DIRS = {".git", "build", "dist", "node_modules", "__pycache__", ".venv", "third_party"}


@dataclass(frozen=True)
class Library:
    name: str
    pattern: re.Pattern[str]
    oneapi: str
    note: str = ""


# NVIDIA libraries, matched by include or API prefix, and their oneAPI counterparts.
LIBRARIES: tuple[Library, ...] = (
    Library("cuBLASLt", re.compile(r"cublasLt\w*|<cublasLt\.h>"), "oneMath BLAS",
            "no direct equivalent: gemm plus a hand-written epilogue (bias, GELU)"),
    Library("cuBLAS", re.compile(r"\bcublas(?!Lt)\w*|<cublas(_v2)?\.h>"), "oneMath BLAS"),
    Library("cuDNN", re.compile(r"\bcudnn\w*|<cudnn\w*\.h>"), "oneDNN"),
    Library("cuFFT", re.compile(r"\bcufft\w*|<cufft\w*\.h>"), "oneMath DFT"),
    Library("cuRAND", re.compile(r"\bcurand\w*|<curand\w*\.h>"), "oneMath RNG"),
    Library("cuSPARSE", re.compile(r"\bcusparse\w*|<cusparse\w*\.h>"), "oneMath sparse BLAS"),
    Library("cuSOLVER", re.compile(r"\bcusolver\w*|<cusolver\w*\.h>"), "oneMath LAPACK"),
    Library("Thrust", re.compile(r"\bthrust::|<thrust/"), "oneDPL"),
    Library("CUB", re.compile(r"\bcub::|<cub/"), "oneDPL / SYCL group algorithms"),
    Library("NCCL", re.compile(r"\bnccl[A-Z]\w*|<nccl\.h>"), "oneCCL"),
)


@dataclass(frozen=True)
class Construct:
    name: str
    pattern: re.Pattern[str]
    sycl: str
    severity: str  # "design": needs a deliberate SYCL design; "mechanical": c2s handles it


CONSTRUCTS: tuple[Construct, ...] = (
    Construct("warp_size_32", re.compile(r"#define\s+WARP_SIZE\s+32|\bwarpSize\b"),
              "SYCL sub-group size is not guaranteed to be 32; query it or request one",
              "design"),
    Construct("warp_shuffle", re.compile(r"__shfl_\w*sync"),
              "sycl::shift_group_left / permute_group_by_xor / select_from_group", "design"),
    Construct("warp_vote", re.compile(r"__(ballot|any|all)_sync|__activemask"),
              "sycl::any_of_group / all_of_group / sub-group ballot extension", "design"),
    Construct("cooperative_groups", re.compile(r"cooperative_groups|\bcg::"),
              "sycl::group / sub_group and group algorithms", "design"),
    Construct("bf16", re.compile(r"__nv_bfloat16|cuda_bf16\.h"),
              "sycl::ext::oneapi::bfloat16", "design"),
    Construct("fp16", re.compile(r"\b__half2?\b|cuda_fp16\.h"), "sycl::half", "mechanical"),
    Construct("inline_ptx", re.compile(r"\basm\s*(volatile\s*)?\("),
              "no SYCL equivalent: rewrite in SYCL", "design"),
    Construct("tensor_cores", re.compile(r"\bwmma::|mma\.sync"),
              "sycl::ext::oneapi::experimental::matrix (joint_matrix)", "design"),
    Construct("async_copy", re.compile(r"cp\.async|memcpy_async|__pipeline_"),
              "no direct equivalent: plain loads into local memory", "design"),
    Construct("textures", re.compile(r"\btex[123]D\w*|cudaTextureObject|\btexture<"),
              "sycl::image / sampled images, or plain buffers", "design"),
    Construct("unified_memory", re.compile(r"cudaMallocManaged"), "sycl::malloc_shared",
              "mechanical"),
    Construct("atomics", re.compile(r"\batomic(Add|Sub|Max|Min|Exch|CAS|And|Or|Xor)\b"),
              "sycl::atomic_ref", "mechanical"),
    Construct("shared_memory", re.compile(r"__shared__"), "sycl::local_accessor",
              "mechanical"),
    Construct("streams_events", re.compile(r"\bcuda(Stream|Event)\w*"),
              "sycl::queue (in-order for stream semantics) and sycl::event", "mechanical"),
)

INCLUDE_RE = re.compile(r'^\s*#\s*include\s*"([^"]+)"', re.M)
LAUNCH_BOUNDS_RE = re.compile(r"__launch_bounds__\s*\([^)]*\)")
# __global__ [qualifiers] <return type> name(  -- launch bounds are removed first.
KERNEL_RE = re.compile(r"__global__\s+[\w:<>,\s*&]*?\b(\w+)\s*\(")


@dataclass
class FileReport:
    path: str
    lines: int
    kernels: list[str] = field(default_factory=list)
    libraries: dict[str, int] = field(default_factory=dict)
    constructs: dict[str, int] = field(default_factory=dict)
    includes: list[str] = field(default_factory=list)  # local headers, transitively
    # What this file gets from those headers: name -> headers it comes from.
    inherited_libraries: dict[str, list[str]] = field(default_factory=dict)
    inherited_constructs: dict[str, list[str]] = field(default_factory=dict)

    @property
    def tier(self) -> int:
        """1: plain kernels; 2: CUDA-only constructs that need a SYCL design; 3: libraries.

        Based on what the file itself uses. What it inherits from local headers is reported
        separately: a header that includes cuBLAS does not make every includer call it."""
        if self.libraries:
            return 3
        design = {c.name for c in CONSTRUCTS if c.severity == "design"}
        if any(n in design for n in self.constructs):
            return 2
        return 1


def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def _local_includes(path: Path, code: str, root: Path) -> list[Path]:
    out = []
    for name in INCLUDE_RE.findall(code):
        for base in (path.parent, root):
            cand = (base / name).resolve()
            if cand.is_file() and cand.is_relative_to(root):
                out.append(cand)
                break
    return out


def scan_file(path: Path, root: Path) -> FileReport:
    raw = path.read_text(errors="replace")
    code = _strip_comments(raw)
    rep = FileReport(str(path.relative_to(root)), raw.count("\n") + (not raw.endswith("\n")))
    rep.kernels = sorted(set(KERNEL_RE.findall(LAUNCH_BOUNDS_RE.sub(" ", code))))
    for lib in LIBRARIES:
        n = len(lib.pattern.findall(code))
        if n:
            rep.libraries[lib.name] = n
    for c in CONSTRUCTS:
        n = len(c.pattern.findall(code))
        if n:
            rep.constructs[c.name] = n
    return rep


def _add_inherited(reports: dict[Path, FileReport], root: Path) -> None:
    """Fill ``includes`` and ``inherited_*`` by following local ``#include "..."`` lines."""
    cache: dict[Path, FileReport] = dict(reports)

    def report_for(p: Path) -> FileReport:
        if p not in cache:
            cache[p] = scan_file(p, root)
        return cache[p]

    for path, rep in reports.items():
        seen: list[Path] = []
        todo = _local_includes(path, _strip_comments(path.read_text(errors="replace")), root)
        while todo:
            inc = todo.pop(0)
            if inc in seen or inc == path:
                continue
            seen.append(inc)
            todo += _local_includes(inc, _strip_comments(inc.read_text(errors="replace")), root)
        rep.includes = [str(p.relative_to(root)) for p in seen]
        for inc in seen:
            r, name = report_for(inc), str(inc.relative_to(root))
            for lib in r.libraries:
                if lib not in rep.libraries:
                    rep.inherited_libraries.setdefault(lib, []).append(name)
            for con in r.constructs:
                if con not in rep.constructs:
                    rep.inherited_constructs.setdefault(con, []).append(name)


def _cuda_files(root: Path) -> list[Path]:
    out = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or any(part in SKIP_DIRS for part in p.relative_to(root).parts):
            continue
        if p.suffix in CUDA_SUFFIXES:
            out.append(p)
        elif p.suffix in HEADER_SUFFIXES and "__global__" in p.read_text(errors="replace"):
            out.append(p)  # headers that define kernels or device code
    return out


def scan(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"not a directory: {root}")
    by_path = {p: scan_file(p, root) for p in _cuda_files(root)}
    _add_inherited(by_path, root)
    files = list(by_path.values())
    libs: Counter[str] = Counter()
    constructs: Counter[str] = Counter()
    for f in files:
        libs.update(f.libraries)
        constructs.update(f.constructs)
    lib_info = {lib.name: lib for lib in LIBRARIES}
    con_info = {c.name: c for c in CONSTRUCTS}
    return {
        "root": str(root),
        "totals": {
            "files": len(files),
            "lines": sum(f.lines for f in files),
            "kernels": sum(len(f.kernels) for f in files),
            "by_tier": dict(sorted(Counter(f.tier for f in files).items())),
        },
        "libraries": [{"name": n, "uses": c, "oneapi": lib_info[n].oneapi,
                       "note": lib_info[n].note,
                       "files": [f.path for f in files if n in f.libraries]}
                      for n, c in libs.most_common()],
        "constructs": [{"name": n, "uses": c, "severity": con_info[n].severity,
                        "sycl": con_info[n].sycl,
                        "files": [f.path for f in files if n in f.constructs]}
                       for n, c in constructs.most_common()],
        "files": [{**asdict(f), "tier": f.tier} for f in files],
    }


def render_markdown(report: dict[str, Any]) -> str:
    t = report["totals"]
    tiers = ", ".join(f"tier {k}: {v}" for k, v in t["by_tier"].items()) or "none"
    out = [f"# CUDA inventory: `{report['root']}`", "",
           f"{t['files']} CUDA file(s), {t['lines']} lines, {t['kernels']} kernel(s) ({tiers}).",
           "", "Tier 1: plain kernels. Tier 2: CUDA-only constructs that need a SYCL design. "
           "Tier 3: NVIDIA libraries.", ""]
    if report["libraries"]:
        out += ["## NVIDIA libraries", "", "| Library | Uses | Files | oneAPI | Note |",
                "|---|---|---|---|---|"]
        out += [f"| {x['name']} | {x['uses']} | {len(x['files'])} | {x['oneapi']} | {x['note']} |"
                for x in report["libraries"]]
        out.append("")
    if report["constructs"]:
        out += ["## CUDA-only constructs", "", "| Construct | Uses | Files | Needs | SYCL |",
                "|---|---|---|---|---|"]
        out += [f"| {x['name']} | {x['uses']} | {len(x['files'])} | {x['severity']} | {x['sycl']} |"
                for x in report["constructs"]]
        out.append("")
    out += ["## Files", "", "Tier is set by what a file uses itself. Names in *italics* come "
            "from a local header it includes: the header must be ported too, but the file may "
            "not use them.", "",
            "| File | Lines | Tier | Kernels | Libraries | Design constructs |",
            "|---|---|---|---|---|---|"]
    design = {c.name for c in CONSTRUCTS if c.severity == "design"}

    def names(direct: dict[str, Any], inherited: dict[str, Any], keep: Any = None) -> str:
        own = [n for n in direct if keep is None or n in keep]
        via = [f"*{n}*" for n in inherited if keep is None or n in keep]
        return ", ".join(own + via) or "-"

    for f in sorted(report["files"], key=lambda f: (f["tier"], f["lines"])):
        out.append(f"| {f['path']} | {f['lines']} | {f['tier']} | {len(f['kernels'])} | "
                   f"{names(f['libraries'], f['inherited_libraries'])} | "
                   f"{names(f['constructs'], f['inherited_constructs'], design)} |")
    return "\n".join(out) + "\n"
