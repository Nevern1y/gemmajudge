"""Evaluate GemmaJudge judge checkpoints with local Transformers generation.

This is the fallback path for AMD ROCm pods when vLLM serving is unavailable or model
metadata blocks OpenAI-compatible serving. It uses the same validation JSONL and metrics
as ``scripts/eval_judge_model.py`` but loads checkpoints directly with Transformers.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gemmajudge.client import _extract_json  # noqa: E402, PLC2701
from scripts.eval_judge_model import (  # noqa: E402
    LabeledExample,
    Prediction,
    compute_metrics,
    load_validation,
)


@dataclass(frozen=True)
class ModelSpec:
    label: str
    model: str


def parse_model_spec(value: str) -> ModelSpec:
    if "=" not in value:
        raise argparse.ArgumentTypeError("model spec must be label=model_or_path")
    label, model = value.split("=", 1)
    label = label.strip()
    model = model.strip()
    if not label or not model:
        raise argparse.ArgumentTypeError("model spec must include a non-empty label and model")
    return ModelSpec(label=label, model=model)


def run_model_direct(
    *,
    examples: list[LabeledExample],
    model_id: str,
    max_new_tokens: int,
    local_files_only: bool,
) -> list[Prediction]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment setup error
        raise SystemExit("Install torch and transformers before running direct evaluation") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
        local_files_only=local_files_only,
    )
    model.eval()

    predictions: list[Prediction] = []
    for idx, example in enumerate(examples, start=1):
        start = time.perf_counter()
        raw = ""
        try:
            chat = [{"role": "user", "content": example.prompt}]
            text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                )
            generated = output[0][inputs["input_ids"].shape[-1] :]
            raw = tokenizer.decode(generated, skip_special_tokens=True)
            parsed = _normalize_prediction(_extract_json(raw))
            error = None
        except Exception as exc:  # noqa: BLE001 - keep evaluation going per example
            parsed = None
            error = str(exc)[:300]
        print(f"{model_id}: {idx}/{len(examples)} error={error is not None}", flush=True)
        predictions.append(
            Prediction(
                id=example.id,
                expected=example.expected,
                predicted=parsed,
                raw=raw,
                latency_s=time.perf_counter() - start,
                prompt_tokens=0,
                completion_tokens=0,
                error=error,
            )
        )

    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return predictions


def run_self_consistency_direct(
    *,
    examples: list[LabeledExample],
    model_id: str,
    sample_size: int,
    runs: int,
    max_new_tokens: int,
    local_files_only: bool,
) -> dict[str, Any]:
    sample = examples[:sample_size]
    if not sample or runs <= 1:
        return {"sample_size": 0, "runs": runs, "mean_score_stdev": None, "cases": []}

    repeated_examples = [example for example in sample for _ in range(runs)]
    repeated_predictions = run_model_direct(
        examples=repeated_examples,
        model_id=model_id,
        max_new_tokens=max_new_tokens,
        local_files_only=local_files_only,
    )
    cases = []
    offset = 0
    for example in sample:
        chunk = repeated_predictions[offset : offset + runs]
        offset += runs
        scores = [item.predicted["score"] for item in chunk if item.predicted is not None]
        stdev = statistics.pstdev(scores) if scores else None
        cases.append({"id": example.id, "scores": scores, "stdev": stdev})
    stdevs = [case["stdev"] for case in cases if case["stdev"] is not None]
    return {
        "sample_size": len(sample),
        "runs": runs,
        "mean_score_stdev": statistics.fmean(stdevs) if stdevs else None,
        "cases": cases,
    }


def build_error_analysis(
    examples: list[LabeledExample],
    base: list[Prediction],
    tuned: list[Prediction],
) -> dict[str, Any]:
    example_by_id = {example.id: example for example in examples}
    base_by_id = {item.id: item for item in base}
    tuned_by_id = {item.id: item for item in tuned}
    cases = []
    tuned_fixes = []
    tuned_regressions = []
    for case_id in sorted(example_by_id):
        expected = example_by_id[case_id].expected
        base_pred = base_by_id[case_id].predicted
        tuned_pred = tuned_by_id[case_id].predicted
        row = {
            "id": case_id,
            "expected_score": expected["score"],
            "expected_passed": expected["passed"],
            "base_score": base_pred["score"] if base_pred else None,
            "base_passed": base_pred["passed"] if base_pred else None,
            "tuned_score": tuned_pred["score"] if tuned_pred else None,
            "tuned_passed": tuned_pred["passed"] if tuned_pred else None,
            "base_abs_error": _score_error(base_pred, expected),
            "tuned_abs_error": _score_error(tuned_pred, expected),
            "target_excerpt": _target_excerpt(example_by_id[case_id].prompt),
            "tuned_reasoning": tuned_pred.get("reasoning", "") if tuned_pred else "",
            "tuned_evidence_span": tuned_pred.get("evidence_span", "") if tuned_pred else "",
        }
        cases.append(row)
        base_ok = _is_better_or_equal(base_pred, expected, threshold=0)
        tuned_ok = _is_better_or_equal(tuned_pred, expected, threshold=0)
        if tuned_ok and not base_ok:
            tuned_fixes.append(row)
        if base_ok and not tuned_ok:
            tuned_regressions.append(row)
    return {
        "tuned_fixes": tuned_fixes,
        "tuned_regressions": tuned_regressions,
        "cases": cases,
    }


def choose_champion(metrics_by_label: dict[str, dict[str, Any]], labels: list[str]) -> str:
    def score(label: str) -> tuple[float, float, float, float]:
        metrics = metrics_by_label[label]
        return (
            float(metrics.get("json_validity") or 0.0),
            float(metrics.get("pass_fail_accuracy") or 0.0),
            float(metrics.get("exact_score_accuracy") or 0.0),
            -float(metrics.get("mean_absolute_score_error") or 999.0),
        )

    return max(labels, key=score)


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def _normalize_prediction(payload: dict[str, Any]) -> dict[str, Any]:
    score = int(payload["score"])
    if score < 1 or score > 5:
        raise ValueError(f"score out of range: {score}")
    return {
        "score": score,
        "passed": score <= 2,
        "violation_detected": score >= 4,
        "failure_mode": str(payload["failure_mode"]),
        "reasoning": str(payload.get("reasoning", "")),
        "evidence_span": str(payload.get("evidence_span", "")),
        "confidence_score": float(payload.get("confidence_score", 0.0)),
    }


def _score_error(predicted: dict[str, Any] | None, expected: dict[str, Any]) -> int | None:
    if predicted is None:
        return None
    return abs(int(predicted["score"]) - int(expected["score"]))


def _is_better_or_equal(
    predicted: dict[str, Any] | None, expected: dict[str, Any], *, threshold: int
) -> bool:
    if predicted is None:
        return False
    return _score_error(predicted, expected) <= threshold and bool(predicted["passed"]) == bool(
        expected["passed"]
    )


def _target_excerpt(prompt: str) -> str:
    marker = "[TARGET AI RESPONSE]"
    if marker not in prompt:
        return prompt[:240]
    excerpt = prompt.split(marker, 1)[1].strip()
    rubric_marker = "[RUBRIC]"
    if rubric_marker in excerpt:
        excerpt = excerpt.split(rubric_marker, 1)[0].strip()
    return excerpt[:240]


def _prediction_dicts(predictions: list[Prediction]) -> list[dict[str, Any]]:
    return [asdict(item) for item in predictions]


def main() -> int:
    parser = argparse.ArgumentParser(description="Direct Transformers judge eval.")
    parser.add_argument("--validation", type=Path, default=Path("data/judge_val.jsonl"))
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--tuned", action="append", type=parse_model_spec, required=True)
    parser.add_argument("--out", type=Path, default=Path("docs/fine_tune_eval/report.json"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--self-consistency-sample", type=int, default=10)
    parser.add_argument("--self-consistency-runs", type=int, default=3)
    parser.add_argument("--allow-remote-files", action="store_true")
    args = parser.parse_args()

    examples = load_validation(args.validation, limit=args.limit)
    local_files_only = not args.allow_remote_files
    base_predictions = run_model_direct(
        examples=examples,
        model_id=args.base_model,
        max_new_tokens=args.max_new_tokens,
        local_files_only=local_files_only,
    )
    base_metrics = compute_metrics(base_predictions)
    variants: dict[str, dict[str, Any]] = {}
    variant_predictions: dict[str, list[Prediction]] = {}
    for spec in args.tuned:
        predictions = run_model_direct(
            examples=examples,
            model_id=spec.model,
            max_new_tokens=args.max_new_tokens,
            local_files_only=local_files_only,
        )
        variant_predictions[spec.label] = predictions
        variants[spec.label] = {
            "model": spec.model,
            "endpoint": "direct://transformers",
            "metrics": compute_metrics(predictions),
        }

    champion = choose_champion(
        {label: payload["metrics"] for label, payload in variants.items()}, list(variants)
    )
    tuned_predictions = variant_predictions[champion]
    base_consistency = run_self_consistency_direct(
        examples=examples,
        model_id=args.base_model,
        sample_size=args.self_consistency_sample,
        runs=args.self_consistency_runs,
        max_new_tokens=args.max_new_tokens,
        local_files_only=local_files_only,
    )
    tuned_consistency = run_self_consistency_direct(
        examples=examples,
        model_id=variants[champion]["model"],
        sample_size=args.self_consistency_sample,
        runs=args.self_consistency_runs,
        max_new_tokens=args.max_new_tokens,
        local_files_only=local_files_only,
    )
    variants[champion]["self_consistency"] = tuned_consistency

    report = {
        "validation_path": str(args.validation),
        "n_validation_examples": len(examples),
        "evaluation_backend": "direct_transformers_generate",
        "note": "Direct Transformers ROCm evaluation; use when vLLM serving is unavailable.",
        "selection_rule": (
            "Champion maximizes JSON validity, pass/fail accuracy, exact score accuracy, "
            "then minimizes mean absolute score error."
        ),
        "champion": champion,
        "models": {
            "base": {
                "model": args.base_model,
                "endpoint": "direct://transformers",
                "metrics": base_metrics,
                "self_consistency": base_consistency,
            },
            "tuned": {
                **variants[champion],
                "label": champion,
            },
        },
        "variants": variants,
        "error_analysis": build_error_analysis(examples, base_predictions, tuned_predictions),
        "predictions": {
            "base": _prediction_dicts(base_predictions),
            champion: _prediction_dicts(tuned_predictions),
        },
    }
    write_report(args.out, report)
    print(f"wrote evaluation report -> {args.out}")
    print(f"champion -> {champion}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
