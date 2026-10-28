"""museport port: source surgery, generated reference and harness, task files, plan mapping."""

from __future__ import annotations

import array
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from museport.llmc import FLT_EPSILON, TIER1, Array, KernelSpec
from museport.port import (
    STRICT_GOAL,
    find_function,
    harness_source,
    make_task,
    reference_source,
    specs_for_plan,
    trim_common,
    write_inputs,
)

MINI = Path(__file__).parent / "fixtures" / "llmc_mini"
SPEC = KernelSpec(
    file="scale.cu", cpu_functions=("scale_cpu",), variants=(1, 1), sizes=(("N", 1000),),
    inputs=(Array("inp", "N"), Array("shift", "1")), outputs=(Array("out", "N", init="zeros"),),
    cpu_call="scale_cpu(out, inp, shift, N)", gpu_call="scale(k, d_out, d_inp, d_shift, N, 256)",
    tol=1e-5)


def test_find_function_spans_the_whole_definition():
    src = (MINI / "scale.cu").read_text()
    a, b = find_function(src, "scale_cpu")
    assert src[a:].startswith("void scale_cpu(") and src[a:b].rstrip().endswith("}")
    assert "__global__" not in src[a:b]
    with pytest.raises(ValueError):
        find_function(src, "nope")


def test_trim_common_removes_cublas_and_keeps_device_setup():
    t = trim_common((MINI / "common.h").read_text())
    assert "cublas_v2.h" not in t and "cublasCreate" not in t and "cublasHandle_t" not in t
    assert "cuda_num_SMs = deviceProp.multiProcessorCount;" in t
    with pytest.raises(ValueError, match="not found"):
        trim_common("#include <stdio.h>\n")  # a changed common.h fails loudly


def test_harness_hides_the_cpu_reference_and_uses_the_contract():
    h = harness_source(SPEC, (MINI / "scale.cu").read_text())
    assert "void scale_cpu(" not in h
    assert not re.search(r"^#define ENABLE_BF16", h, re.M)
    assert "#define SCALE 2.0f" in h  # kernels still need the file's macros
    assert h.count("int main(") == 1 and "scale(k, d_out, d_inp, d_shift, N, 256);" in h
    assert "const int variants[] = {1, 1};" in h


@pytest.mark.skipif(shutil.which("g++") is None, reason="needs g++")
def test_generated_reference_builds_runs_and_repeats_per_variant(tmp_path):
    ref = reference_source(SPEC, (MINI / "scale.cu").read_text())
    (tmp_path / "ref.cpp").write_text(ref)
    write_inputs(SPEC, tmp_path / "in.bin")
    subprocess.run(["g++", "-O2", "ref.cpp", "-o", "ref"], cwd=tmp_path, check=True)
    subprocess.run(["./ref", "in.bin", "out.bin", "1000"], cwd=tmp_path, check=True)
    inp = array.array("f")
    inp.frombytes((tmp_path / "in.bin").read_bytes())
    out = array.array("f")
    out.frombytes((tmp_path / "out.bin").read_bytes())
    assert len(out) == 2 * 1000  # one copy per variant
    assert out[5] == pytest.approx(2.0 * inp[5] + inp[1000])
    assert out[1005] == out[5]


def test_inputs_are_deterministic_and_typed(tmp_path):
    spec = KernelSpec(file="x.cu", cpu_functions=(), variants=(1,), sizes=(("N", 64),),
                      inputs=(Array("a", "N", init="unit"), Array("t", "N", "i32", "int:7")),
                      outputs=(), cpu_call="", gpu_call="", tol=1e-5)
    write_inputs(spec, tmp_path / "1.bin")
    write_inputs(spec, tmp_path / "2.bin")
    raw = (tmp_path / "1.bin").read_bytes()
    assert raw == (tmp_path / "2.bin").read_bytes() and len(raw) == 64 * 4 * 2
    a = array.array("f")
    a.frombytes(raw[:256])
    t = array.array("i")
    t.frombytes(raw[256:])
    assert all(0 <= x < 1 for x in a) and all(0 <= x < 7 for x in t)


def test_make_task_writes_a_toolkit_task(tmp_path):
    path = make_task(SPEC, MINI, tmp_path / "task")
    task = json.loads(path.read_text())
    assert task["cuda_path"] == "scale_harness.cu" and task["inputs"] == ["input.bin"]
    assert task["candidate_cmd"] == "build/candidate {input} {output} 1000"
    assert (task["tolerance"], task["rtol"]) == (1e-5, FLT_EPSILON)
    assert "void scale_cpu(" not in (tmp_path / "task" / "scale_harness.cu").read_text()
    assert "#include <cublas" not in (tmp_path / "task" / "common.h").read_text()
    assert STRICT_GOAL not in task["goal"]
    strict = json.loads(make_task(SPEC, MINI, tmp_path / "s", strict_goal=True).read_text())
    assert strict["goal"].endswith(STRICT_GOAL)
    bare = json.loads(make_task(SPEC, MINI, tmp_path / "n", no_migrate=True).read_text())
    assert "migrate's output" not in bare["goal"] and "not available in this run" in bare["goal"]


def test_plan_items_map_to_specs():
    plan = {"items": [
        {"id": "common", "tier": 1, "files": ["common.h"]},
        {"id": "g", "tier": 1, "files": ["gelu_forward.cu"]},
        {"id": "x", "tier": 1, "files": ["unknown.cu"]},
        {"id": "m", "tier": 3, "files": ["matmul_forward.cu"]}]}
    got = [(it["id"], spec.file if spec else None) for it, spec in specs_for_plan(plan)]
    assert got == [("g", "gelu_forward.cu"), ("x", None)]
    assert len(TIER1) == 10


def test_tier2_fields_host_prep_prelude_device_count_and_includes():
    spec = KernelSpec(
        file="scale.cu", cpu_functions=("scale_cpu",), variants=(1,), sizes=(("N", 8),),
        inputs=(Array("inp", "N"), Array("shift", "1")),
        outputs=(Array("out", "N", init="zeros", device_count="1024"),),
        cpu_call="scale_cpu(out, inp, shift, N)", gpu_call="scale(k, d_out, d_inp, d_shift, N, 32)",
        tol=1e-5, host_prep="inp[0] *= 2;", gpu_prelude="int extra = 1;",
        extra_includes=("<cuda_bf16.h>",))
    src = (MINI / "scale.cu").read_text()
    h = harness_source(spec, src)
    assert h.index("#include <cuda_bf16.h>") < h.index('#include "common.h"')
    assert "cudaMalloc(&d_out, sizeof(float) * (size_t)(1024))" in h
    assert h.index("inp[0] *= 2;") < h.index("cudaMalloc(&d_inp")  # before the upload
    assert h.index("int extra = 1;") < h.index("const int variants[]")
    ref = reference_source(spec, src)
    assert ref.index("inp[0] *= 2;") < ref.index("scale_cpu(out, inp, shift, N);")
    kept = KernelSpec(**{**spec.__dict__, "keep_in_harness": ("scale_cpu",)})
    assert "void scale_cpu(" in harness_source(kept, src)


def test_all_specs_name_existing_tiers():
    from museport.llmc import SPECS, TIER2
    assert len(TIER2) == 5 and set(TIER2) <= set(SPECS)


def test_trace_spend_sums_reported_usage(tmp_path):
    from museport.port import trace_spend
    run = tmp_path / "20261026T000000Z"
    run.mkdir()
    events = [{"type": "start"},
              {"type": "model_response", "usage": {"input_tokens": 1_000_000,
                                                   "output_tokens": 0}},
              {"type": "model_response", "usage": {"input_tokens": 0,
                                                   "output_tokens": 1_000_000,
                                                   "cache_read_input_tokens": 1_000_000}}]
    (run / "trace.jsonl").write_text("\n".join(json.dumps(e) for e in events))
    got = trace_spend(tmp_path, "muse-spark-1.3")
    assert got["turns"] == 2 and got["cost_usd"] == pytest.approx(1.25 + 4.25 + 0.15)
    assert trace_spend(tmp_path / "none", "muse-spark-1.3")["cost_usd"] is None


def test_variant_check_flags_copied_variants(tmp_path):
    from museport.port import variant_check
    k = "q.parallel_for<class K{n}>(sycl::range<1>(n), [=](sycl::id<1> i) {{ out[i] = {e}; }});\n"
    (tmp_path / "a.cpp").write_text(k.format(n=1, e="inp[i]") + k.format(n=2, e="inp[i]"))
    (tmp_path / "b.cpp").write_text(k.format(n=1, e="inp[i]") + k.format(n=2, e="2 * inp[i]"))
    a = variant_check(tmp_path, ["a.cpp"], 2)
    b = variant_check(tmp_path, ["b.cpp"], 2)
    assert (a["distinct_kernel_bodies"], a["duplicated_variants"]) == (1, True)
    assert (b["distinct_kernel_bodies"], b["duplicated_variants"]) == (2, False)
    (tmp_path / "c.cpp").write_text(k.format(n=1, e="inp[i]") + k.format(n=2, e="inp[i]")
                                    + "oneapi::math::blas::column_major::gemm(q, ta, tb);\n")
    c = variant_check(tmp_path, ["c.cpp"], 2)  # a library variant counts as one
    assert (c["onemath_calls"], c["duplicated_variants"]) == (1, False)


def test_output_post_and_interleaved_writes():
    two = KernelSpec(
        file="scale.cu", cpu_functions=("scale_cpu",), variants=(1,), sizes=(("N", 8),),
        inputs=(Array("inp", "N"), Array("shift", "1")),
        outputs=(Array("out", "N", init="zeros"), Array("out2", "N", init="zeros")),
        cpu_call="scale_cpu(out, inp, shift, N)", gpu_call="scale(k, d_out, d_inp, d_shift, N, 32)",
        tol=1e-5)
    src = (MINI / "scale.cu").read_text()
    h = harness_source(two, src)  # without output_post: copy and write interleaved, as before
    assert h.index("fwrite(out,") < h.index("cudaMemcpy(out2,")
    post = KernelSpec(**{**two.__dict__, "output_post": "out[0] = 0;"})
    h = harness_source(post, src)  # with it: all copies, the post step, then the writes
    assert h.index("cudaMemcpy(out2,") < h.index("out[0] = 0;") < h.index("fwrite(out,")
    assert reference_source(post, src).index("out[0] = 0;") < reference_source(post, src).index(
        "fwrite(out,")


def test_summary_merge_keeps_paid_rows_and_earlier_runs(tmp_path):
    from museport.port import write_summary
    plan = {"items": [{"id": "a", "tier": 1, "files": ["gelu_forward.cu"]},
                      {"id": "b", "tier": 1, "files": ["permute.cu"]}]}
    paid_a = {"item": "a", "file": "gelu_forward.cu", "status": "success", "run_id": "r1",
              "iterations": 9, "cost_usd": 0.10}
    s = write_summary(tmp_path, plan, "m", [paid_a])
    s = write_summary(tmp_path, plan, "m", [paid_a])  # same run written again: no double count
    assert s["spent_usd"] == 0.10 and "earlier_runs" not in s["items"][0]
    # tasks only (no cost) into the same directory: the paid row survives
    s = write_summary(tmp_path, plan, "m", [{"item": "a", "file": "gelu_forward.cu",
                                             "status": "task_written",
                                             "harness_check": {"ok": True}}])
    a = s["items"][0]
    assert a["status"] == "success" and a["cost_usd"] == 0.10 and a["harness_check"]["ok"]
    # a rerun: the first run is kept in earlier_runs and counted
    s = write_summary(tmp_path, plan, "m", [{"item": "a", "file": "gelu_forward.cu",
                                             "status": "max_iterations", "run_id": "r2",
                                             "iterations": 12, "cost_usd": 0.20}])
    a = s["items"][0]
    assert a["run_id"] == "r2" and a["earlier_runs"][0]["cost_usd"] == 0.10
    assert s["spent_usd"] == pytest.approx(0.30)
    # run_items writes the same rerun again at the end: its earlier run must stay
    s = write_summary(tmp_path, plan, "m", [{"item": "a", "file": "gelu_forward.cu",
                                             "status": "max_iterations", "run_id": "r2",
                                             "iterations": 12, "cost_usd": 0.20}])
    assert s["items"][0]["earlier_runs"][0]["cost_usd"] == 0.10
    assert s["spent_usd"] == pytest.approx(0.30)
    s = write_summary(tmp_path, plan, "m", [{"item": "b", "file": "permute.cu",
                                             "status": "success", "run_id": "r3",
                                             "cost_usd": 0.05}])
    assert [r["item"] for r in s["items"]] == ["a", "b"] and s["spent_usd"] == pytest.approx(0.35)
    assert "Earlier runs" in (tmp_path / "SUMMARY.md").read_text()


def test_gpu_task_regenerates_missing_inputs(tmp_path, monkeypatch):
    # input.bin is not committed: a fresh clone must still build GPU tasks, with the same bytes
    from museport import port as P

    monkeypatch.setitem(P.SPECS, "scale.cu", SPEC)
    item = tmp_path / "t1-scale"
    make_task(SPEC, MINI, item / "task")
    original = (item / "task" / "input.bin").read_bytes()
    (item / "task" / "input.bin").unlink()
    (item / "port").mkdir()
    (item / "port" / "scale.cpp").write_text("int main() {}\n")
    (item / "trace.jsonl").write_text(json.dumps(
        {"type": "tool_call", "iteration": 1, "name": "compile",
         "input": {"sources": ["scale.cpp"]}}) + "\n")
    task = json.loads(P.make_gpu_task(item, tmp_path / "gpu", "m").read_text())
    assert (tmp_path / "gpu" / "input.bin").read_bytes() == original
    assert task["sycl_sources"] == ["agent_port/scale.cpp"]


def test_kernel_bodies_reads_functor_kernels():
    from museport.port import kernel_bodies

    src = """
struct AddK1 { float *x; void operator()(sycl::id<1> i) const { x[i] += 1; } };
struct AddK2 { float *x; void operator()(sycl::id<1> i) const { x[i] *= 2; } };
void a(sycl::queue &q, float *x) { q.parallel_for(sycl::nd_range<1>(64, 32), AddK1{x}); }
void b(sycl::queue &q, float *x) { q.parallel_for(sycl::nd_range<1>(64, 32), AddK2{x}); }
void c(sycl::queue &q, float *x) { q.parallel_for(sycl::range<1>(8), [=](auto i) { x[i] = 0; }); }
"""
    bodies = kernel_bodies(src)
    assert len(bodies) == 3 and len(set(bodies)) == 3
    assert "*= 2" in bodies[1] and "x[i] = 0" in bodies[2]


def test_gpu_task_for_a_onemath_port_builds_onemath_for_the_vendor(tmp_path):
    import json

    import pytest

    from museport.port import make_gpu_task

    root = Path(__file__).resolve().parents[1]
    item = root / "evidence" / "port-llmc-tier23-run2" / "t3-matmul-forward"
    with pytest.raises(ValueError, match="name the GPU vendor"):
        make_gpu_task(item, tmp_path / "x", "muse-spark-1.3")
    for vendor, backend in (("nvidia", "cublas"), ("amd", "rocblas")):
        path = make_gpu_task(item, tmp_path / vendor, "muse-spark-1.3", vendor)
        task = json.loads(path.read_text())
        assert task["gpu_setup"] == f"onemath_{backend}_setup.sh"
        setup = (path.parent / task["gpu_setup"]).read_text()
        assert f"-DENABLE_{backend.upper()}_BACKEND=ON" in setup
    # a port without oneMath needs no vendor and gets no setup
    plain = make_gpu_task(root / "evidence" / "port-llmc-tier1-run2" / "t1-gelu-forward",
                          tmp_path / "plain", "muse-spark-1.3")
    assert "gpu_setup" not in json.loads(plain.read_text())
