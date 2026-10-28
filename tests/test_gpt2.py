import json

import pytest

from museport.gpt2 import HARNESS, STATE, WEIGHTS, make_task


def _llmc(tmp_path):
    llmc = tmp_path / "llm.c"
    (llmc / "llmc").mkdir(parents=True)
    (llmc / "train_gpt2_fp32.cu").write_text("// training file\n")
    for h in ("utils.h", "tokenizer.h", "dataloader.h", "rand.h"):
        (llmc / "llmc" / h).write_text(f"// {h}\n")
    return llmc


def test_stage1_task_links_the_data_and_hides_the_reference(tmp_path):
    llmc = _llmc(tmp_path)
    data = tmp_path / "data"
    data.mkdir()
    with pytest.raises(FileNotFoundError, match="download_starter_pack"):
        make_task(llmc, data, tmp_path / "t")
    for name in (WEIGHTS, STATE):
        (data / name).write_bytes(b"x")
    task = json.loads(make_task(llmc, data, tmp_path / "t").read_text())
    out = tmp_path / "t"
    assert (out / WEIGHTS).is_symlink() and (out / STATE).is_symlink()  # never copied
    assert (out / "llmc" / "rand.h").exists() and (out / "train_gpt2_fp32.cu").exists()
    harness = (out / HARNESS).read_text()
    assert "#define TESTING" in harness and '#include "train_gpt2_fp32.cu"' in harness
    assert "CUBLAS_DEFAULT_MATH" in harness  # TF32 off, as llm.c's test
    assert task["inputs"] == [STATE] and (task["tolerance"], task["rtol"]) == (1e-2, 0.0)
    assert task["candidate_cmd"] == f"build/candidate {{input}} {{output}} build/{WEIGHTS}"
    assert "{task_dir}/gpt2_ref.cpp" in task["reference_build"][-1]
    assert "-lonemath" in task["base_flags"] and "delete_file" in task["goal"]
    # the CPU proof shares the contract main with the harness
    proof = (out / "cpu_proof.c").read_text()
    assert '#include "train_gpt2.c"' in proof and proof.split("#define MUSEPORT_COPY")[1]
    make_task(llmc, data, out)  # rewriting replaces the links
    assert (out / WEIGHTS).is_symlink()


def test_stage2_writes_the_backward_task_with_llmc_gradient_tolerance(tmp_path):
    llmc = _llmc(tmp_path)
    data = tmp_path / "data"
    data.mkdir()
    for name in (WEIGHTS, STATE):
        (data / name).write_bytes(b"x")
    task = json.loads(make_task(llmc, data, tmp_path / "b", "backward").read_text())
    out = tmp_path / "b"
    assert task["cuda_path"] == "gpt2_backward_harness.cu" and task["tolerance"] == 2e-2
    harness = (out / task["cuda_path"]).read_text()
    assert "gpt2_backward(&model);" in harness and "model.grads_memory" in harness
    assert "the gradient of every parameter" in task["goal"] and "within 0.02" in task["goal"]
    assert "grads_memory" in (out / "cpu_proof.c").read_text()
    assert "gradients fill the rest of the file" in (out / "gpt2_ref.cpp").read_text()


def test_stage3_writes_the_training_task_with_llmc_expected_losses(tmp_path):
    from museport.gpt2 import EXPECTED_LOSSES

    llmc = _llmc(tmp_path)
    data = tmp_path / "data"
    data.mkdir()
    for name in (WEIGHTS, STATE):
        (data / name).write_bytes(b"x")
    task = json.loads(make_task(llmc, data, tmp_path / "s3", "train").read_text())
    out = tmp_path / "s3"
    assert task["cuda_path"] == "gpt2_train_harness.cu" and task["tolerance"] == 1e-2
    harness = (out / task["cuda_path"]).read_text()
    assert "gpt2_update(&model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.01f, step + 1);" in harness
    assert "AdamW update" in task["goal"] and "10 values" in task["goal"]
    ref = (out / "gpt2_ref.cpp").read_text()
    assert len(EXPECTED_LOSSES) == 10 and all(f"{v!r}f" in ref for v in EXPECTED_LOSSES)
    assert "gpt2_update" in (out / "cpu_proof.c").read_text()


def test_gpu_task_fetches_the_data_and_builds_onemath_on_the_machine(tmp_path):
    from museport.gpt2 import DATA_SHA256, make_gpu_task

    llmc = _llmc(tmp_path)
    data = tmp_path / "data"
    data.mkdir()
    for name in (WEIGHTS, STATE):
        (data / name).write_bytes(b"x")
    stage = tmp_path / "stage"
    make_task(llmc, data, stage / "task", "train")
    (stage / "run1" / "port").mkdir(parents=True)
    (stage / "run1" / "port" / "gpt2_train.cpp").write_text("// sycl")
    (stage / "run1" / "trace.jsonl").write_text(json.dumps(
        {"type": "tool_call", "name": "compile", "input": {"sources": ["gpt2_train.cpp"]}}) + "\n")
    task = json.loads(make_gpu_task(stage, tmp_path / "gpu", "m").read_text())
    gpu = tmp_path / "gpu"
    assert not (gpu / WEIGHTS).exists() and not (gpu / STATE).exists()  # fetched, not shipped
    assert {f["path"]: f["sha256"] for f in task["fetch"]} == DATA_SHA256
    assert all(f["url"].startswith("https://huggingface.co/") for f in task["fetch"])
    assert task["sycl_sources"] == ["agent_port/gpt2_train.cpp"]
    assert task["native_flags"] == ["-lcublas"]
    setup = (gpu / task["gpu_setup"]).read_text()
    assert "-DENABLE_CUBLAS_BACKEND=ON" in setup and "export ONEMATH_PREFIX" in setup


def test_amd_gpu_task_builds_onemath_with_rocblas():
    from museport.gpt2 import onemath_gpu_setup

    amd, nv = onemath_gpu_setup("amd"), onemath_gpu_setup("nvidia")
    assert "-DENABLE_ROCBLAS_BACKEND=ON" in amd and 'HIP_TARGETS="$HIP_ARCH"' in amd
    assert "rocm_agent_enumerator" in amd and "onemath-rocblas" in amd
    # attempt 4 on the MI350X: hip-config needs AMDDeviceLibs' CMake config (J3 hit it again)
    assert "amdrocm-llvm-dev" in amd and "lib/llvm/lib/cmake/" in amd
    assert "-DENABLE_CUBLAS_BACKEND=ON" in nv and "ROCBLAS" not in nv


def test_gpu_setup_leaves_the_toolkit_runners_variables_alone():
    # the toolkit sources gpu_setup in its runner; a stray D= deleted the results (AMD, Oct 28)
    import re

    from oneapi_agent.bundle import RUNNER

    from museport.gpt2 import onemath_gpu_setup

    runner = set(re.findall(r"^\s*([A-Za-z_][A-Za-z0-9_]*)=", RUNNER, re.M))
    for vendor in ("nvidia", "amd"):
        ours = set(re.findall(r"(?:^|[\s;(])([A-Za-z_][A-Za-z0-9_]*)=", onemath_gpu_setup(vendor)))
        assert not ours & runner, (vendor, ours & runner)
