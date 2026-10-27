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
