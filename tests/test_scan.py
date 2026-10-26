"""The repository scanner on a small fixture repository."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from museport.cli import main
from museport.scan import render_markdown, scan

MINI = Path(__file__).parent / "fixtures" / "mini"


@pytest.fixture(scope="module")
def report():
    return scan(MINI)


def files(report):
    return {f["path"]: f for f in report["files"]}


def test_finds_cuda_sources_and_kernel_headers_but_skips_build_dirs(report):
    assert set(files(report)) == {"gemm.cu", "kernels/reduce.cu", "kernels/simple.cu"}
    # common.h defines device code but no __global__ kernel, so it is not a target itself
    assert report["totals"]["files"] == 3


def test_kernels_ignore_comments_and_handle_templates_and_launch_bounds(report):
    assert files(report)["kernels/simple.cu"]["kernels"] == ["add_kernel", "scale_kernel"]
    assert files(report)["kernels/reduce.cu"]["kernels"] == ["sum_kernel"]


def test_libraries_and_their_oneapi_counterparts(report):
    libs = {x["name"]: x for x in report["libraries"]}
    assert set(libs) == {"cuBLAS", "Thrust"}
    assert libs["cuBLAS"]["oneapi"] == "oneMath BLAS" and libs["cuBLAS"]["files"] == ["gemm.cu"]
    assert libs["Thrust"]["oneapi"] == "oneDPL"


def test_constructs_and_tiers(report):
    f = files(report)
    assert f["gemm.cu"]["tier"] == 3
    assert f["kernels/reduce.cu"]["tier"] == 2
    assert {"warp_shuffle", "cooperative_groups", "shared_memory"} <= set(
        f["kernels/reduce.cu"]["constructs"])
    simple = f["kernels/simple.cu"]
    assert simple["tier"] == 1 and set(simple["constructs"]) == {"atomics"}


def test_inherited_from_local_headers_is_reported_but_does_not_raise_the_tier(report):
    simple = files(report)["kernels/simple.cu"]
    assert simple["includes"] == ["kernels/common.h"]
    assert simple["inherited_libraries"] == {"cuBLAS": ["kernels/common.h"]}
    assert set(simple["inherited_constructs"]) == {"warp_size_32", "warp_shuffle"}
    assert simple["tier"] == 1


def test_markdown_and_cli(capsys, tmp_path):
    md = render_markdown(scan(MINI))
    assert "3 CUDA file(s)" in md and "| cuBLAS |" in md and "*cuBLAS*" in md
    assert main(["scan", str(MINI), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["totals"]["kernels"] == 3
    assert main(["scan", str(tmp_path / "missing")]) == 2
