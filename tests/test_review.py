import json
from pathlib import Path

from museport.review import REVIEW_SCHEMA, messages_for, review_targets, run_reviews

ROOT = Path(__file__).resolve().parents[1]


def test_review_prompt_has_the_code_and_no_test_results():
    [it] = [i for i in review_targets(ROOT / "evidence", ("port-llmc-tier23-run2",))
            if i.id == "t2-layernorm-backward"]
    text = messages_for(it)[1]["content"]
    assert "=== CUDA program: layernorm_backward_harness.cu ===" in text
    assert "=== SYCL port:" in text and "__global__" in text
    # nothing the hidden grade or the variant check found reaches the reviewer
    for leak in ("verify", "PASS", "duplicated", "hidden"):
        assert leak not in messages_for(it)[0]["content"] + text.split("===")[0]


def test_reviews_are_compared_with_the_hidden_grade(tmp_path):
    answer = {"variants": [{"variant": 1, "keeps_cuda_algorithm": False, "note": "n"}],
              "copies_one_implementation": True, "idiomatic_sycl": True,
              "likely_correct": True, "concerns": ["c"], "verdict": "reject"}

    def fake(url, headers, body):
        assert body["response_format"]["json_schema"]["schema"] == REVIEW_SCHEMA
        return {"choices": [{"message": {"content": json.dumps(answer)},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 100}}

    s = run_reviews(ROOT / "evidence", tmp_path, model="muse-spark-1.3",
                    batches=("port-llmc-tier23-run2",), transport=fake, log=lambda m: None)
    rows = {r["item"]: r for r in s["reviews"]}
    lb = rows["t2-layernorm-backward"]["compare"]
    assert lb == {"hidden_grade_correct": True, "reviewer_likely_correct": True,
                  "variant_check_duplicated": True, "reviewer_copies": True,
                  "reviewer_verdict": "reject", "variants_not_kept": [1]}
    assert rows["t2-softmax-forward"]["compare"]["hidden_grade_correct"] is False
    assert s["spent_usd"] > 0 and (tmp_path / "summary.json").exists()
