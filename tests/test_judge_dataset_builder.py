import json

from scripts.build_judge_dataset import build_dataset, split_records
from scripts.convert_judge_dataset_fireworks import convert_row


def test_dataset_builder_extracts_real_and_curated_examples():
    records = build_dataset()
    assert records
    assert {record.failure_mode for record in records} == {"hallucination", "jailbreak", "bias"}
    assert any(record.source == "amd_proof_w7900" for record in records)
    assert any(record.source == "real_runs_leaderboard" for record in records)
    assert any(record.source == "gemmajudge_curated_seed" for record in records)


def test_dataset_builder_dedupes_fingerprints():
    records = build_dataset()
    fingerprints = [record.fingerprint for record in records]
    assert len(fingerprints) == len(set(fingerprints))


def test_split_has_all_failure_modes_in_validation():
    records = build_dataset()
    train, validation = split_records(records)
    assert train
    assert validation
    assert {record.failure_mode for record in validation} == {"hallucination", "jailbreak", "bias"}


def test_score_three_examples_are_validation_only():
    records = build_dataset()
    assert any(record.score == 3 for record in records)
    train, validation = split_records(records)
    assert all(record.score != 3 for record in train)
    assert any(record.score == 3 for record in validation)


def test_jsonl_example_uses_gemma_model_role_and_valid_answer_json():
    record = build_dataset()[0]
    row = record.to_jsonl_example("train")
    assert row["messages"][0]["role"] == "user"
    assert row["messages"][1]["role"] == "model"
    answer = json.loads(row["messages"][1]["content"])
    assert answer["passed"] == (answer["score"] <= 2)
    assert answer["violation_detected"] == (answer["score"] >= 4)
    assert answer["failure_mode"] in {"hallucination", "jailbreak", "bias"}


def test_fireworks_converter_maps_model_to_assistant_and_drops_metadata():
    row = build_dataset()[0].to_jsonl_example("train")
    converted = convert_row(row)
    assert list(converted) == ["messages"]
    assert converted["messages"][1]["role"] == "assistant"
