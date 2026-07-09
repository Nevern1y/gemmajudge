"""Evaluate a base judge model against a tuned GemmaJudge model.

The script expects OpenAI-compatible chat endpoints (local vLLM or hosted private
backend). It never calls closed judge APIs for labels; labels come from validation JSONL.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


JUDGE_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "minimum": 1, "maximum": 5},
        "passed": {"type": "boolean"},
        "violation_detected": {"type": "boolean"},
        "failure_mode": {"type": "string", "enum": ["hallucination", "jailbreak", "bias"]},
        "reasoning": {"type": "string"},
        "evidence_span": {"type": "string"},
        "confidence_score": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": [
        "score",
        "passed",
        "violation_detected",
        "failure_mode",
        "reasoning",
        "evidence_span",
        "confidence_score",
    ],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class LabeledExample:
    id: str
    prompt: str
    expected: dict[str, Any]


@dataclass(frozen=True)
class Prediction:
    id: str
    expected: dict[str, Any]
    predicted: dict[str, Any] | None
    raw: str
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    error: str | None = None


def load_validation(path: Path, *, limit: int | None = None) -> list[LabeledExample]:
    examples: list[LabeledExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            expected = json.loads(row["messages"][1]["content"])
            examples.append(
                LabeledExample(
                    id=row.get("metadata", {}).get("id", f"example_{len(examples) + 1}"),
                    prompt=row["messages"][0]["content"],
                    expected=expected,
                )
            )
            if limit is not None and len(examples) >= limit:
                break
    return examples


def run_model(
    *,
    examples: list[LabeledExample],
    endpoint: str,
    api_key: str,
    model: str,
    temperature: float,
) -> list[Prediction]:
    from gemmajudge.client import _extract_json, json_schema_response_format  # noqa: PLC2701

    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - environment setup error
        raise SystemExit("Install openai>=1.40.0 to run model evaluation") from exc

    client = OpenAI(base_url=endpoint, api_key=api_key, timeout=60, max_retries=0)
    predictions: list[Prediction] = []
    for example in examples:
        start = time.perf_counter()
        raw = ""
        usage_prompt = 0
        usage_completion = 0
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": example.prompt}],
                response_format=json_schema_response_format(
                    "gemmajudge_verdict", JUDGE_OUTPUT_SCHEMA
                ),
                temperature=temperature,
                max_tokens=320,
            )
            raw = response.choices[0].message.content or ""
            usage = response.usage
            if usage is not None:
                usage_prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
                usage_completion = int(getattr(usage, "completion_tokens", 0) or 0)
            parsed = _normalize_prediction(_extract_json(raw))
            error = None
        except Exception as exc:  # noqa: BLE001 - keep evaluation going per example
            parsed = None
            error = str(exc)[:300]
        predictions.append(
            Prediction(
                id=example.id,
                expected=example.expected,
                predicted=parsed,
                raw=raw,
                latency_s=time.perf_counter() - start,
                prompt_tokens=usage_prompt,
                completion_tokens=usage_completion,
                error=error,
            )
        )
    return predictions


def compute_metrics(predictions: list[Prediction]) -> dict[str, Any]:
    total = len(predictions)
    valid = [item for item in predictions if item.predicted is not None]
    if total == 0:
        return {}

    exact_score = sum(
        1 for item in valid if int(item.predicted["score"]) == int(item.expected["score"])
    )
    pass_fail = sum(
        1 for item in valid if bool(item.predicted["passed"]) == bool(item.expected["passed"])
    )
    mae_values = [
        abs(int(item.predicted["score"]) - int(item.expected["score"])) for item in valid
    ]
    evidence_required = [item for item in valid if int(item.predicted["score"]) >= 4]
    evidence_non_empty = sum(1 for item in evidence_required if item.predicted["evidence_span"])
    return {
        "n": total,
        "json_validity": len(valid) / total,
        "exact_score_accuracy": exact_score / total,
        "pass_fail_accuracy": pass_fail / total,
        "macro_f1_violation_detected": _binary_macro_f1(valid),
        "mean_absolute_score_error": statistics.fmean(mae_values) if mae_values else None,
        "evidence_span_non_empty_rate_score_ge_4": (
            evidence_non_empty / len(evidence_required) if evidence_required else None
        ),
        "avg_latency_s": statistics.fmean(item.latency_s for item in predictions),
        "total_prompt_tokens": sum(item.prompt_tokens for item in predictions),
        "total_completion_tokens": sum(item.completion_tokens for item in predictions),
        "errors": [
            {"id": item.id, "error": item.error} for item in predictions if item.error is not None
        ],
    }


def run_self_consistency(
    *,
    examples: list[LabeledExample],
    endpoint: str,
    api_key: str,
    model: str,
    sample_size: int,
    runs: int,
) -> dict[str, Any]:
    sample = examples[:sample_size]
    if not sample or runs <= 1:
        return {"sample_size": 0, "runs": runs, "mean_score_stdev": None, "cases": []}

    cases = []
    for example in sample:
        repeated = run_model(
            examples=[example] * runs,
            endpoint=endpoint,
            api_key=api_key,
            model=model,
            temperature=0.0,
        )
        scores = [item.predicted["score"] for item in repeated if item.predicted is not None]
        stdev = statistics.pstdev(scores) if scores else None
        cases.append({"id": example.id, "scores": scores, "stdev": stdev})
    stdevs = [case["stdev"] for case in cases if case["stdev"] is not None]
    return {
        "sample_size": len(sample),
        "runs": runs,
        "mean_score_stdev": statistics.fmean(stdevs) if stdevs else None,
        "cases": cases,
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    readme = path.with_name("README.md")
    tuned = report["models"]["tuned"]["metrics"]
    base = report["models"]["base"]["metrics"]
    readme.write_text(
        "\n".join(
            [
                "# GemmaJudge Fine-Tune Evaluation",
                "",
                f"Validation examples: {report['n_validation_examples']}",
                "",
                "| Metric | Base | Tuned |",
                "|---|---:|---:|",
                _metric_row("JSON validity", base, tuned, "json_validity"),
                _metric_row("Exact score accuracy", base, tuned, "exact_score_accuracy"),
                _metric_row("Pass/fail accuracy", base, tuned, "pass_fail_accuracy"),
                _metric_row("Violation macro-F1", base, tuned, "macro_f1_violation_detected"),
                _metric_row("Mean absolute score error", base, tuned, "mean_absolute_score_error"),
                "",
                "This report compares Gemma-family judge outputs against the validation "
                "JSONL labels.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


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


def _binary_macro_f1(valid: list[Prediction]) -> float | None:
    if not valid:
        return None
    f1s: list[float] = []
    for klass in (False, True):
        tp = sum(
            1
            for item in valid
            if bool(item.expected["violation_detected"]) is klass
            and bool(item.predicted["violation_detected"]) is klass
        )
        fp = sum(
            1
            for item in valid
            if bool(item.expected["violation_detected"]) is not klass
            and bool(item.predicted["violation_detected"]) is klass
        )
        fn = sum(
            1
            for item in valid
            if bool(item.expected["violation_detected"]) is klass
            and bool(item.predicted["violation_detected"]) is not klass
        )
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return statistics.fmean(f1s)


def _metric_row(label: str, base: dict[str, Any], tuned: dict[str, Any], key: str) -> str:
    return f"| {label} | {_fmt(base.get(key))} | {_fmt(tuned.get(key))} |"


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _api_key(env_name: str) -> str:
    value = os.environ.get(env_name)
    if not value:
        raise SystemExit(f"Set {env_name} for the requested endpoint before running eval")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare base Gemma judge vs tuned GemmaJudge.")
    parser.add_argument("--validation", type=Path, default=Path("data/judge_val.jsonl"))
    parser.add_argument("--base-endpoint", default=os.environ.get("BASE_JUDGE_ENDPOINT"))
    parser.add_argument("--tuned-endpoint", default=os.environ.get("TUNED_JUDGE_ENDPOINT"))
    parser.add_argument("--base-api-key-env", default="BASE_JUDGE_API_KEY")
    parser.add_argument("--tuned-api-key-env", default="TUNED_JUDGE_API_KEY")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--tuned-model", required=True)
    parser.add_argument("--out", type=Path, default=Path("docs/fine_tune_eval/report.json"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--self-consistency-sample", type=int, default=10)
    parser.add_argument("--self-consistency-runs", type=int, default=3)
    args = parser.parse_args()

    if not args.base_endpoint or not args.tuned_endpoint:
        raise SystemExit("Set --base-endpoint/--tuned-endpoint or matching env vars")

    examples = load_validation(args.validation, limit=args.limit)
    base_key = _api_key(args.base_api_key_env)
    tuned_key = _api_key(args.tuned_api_key_env)
    base_predictions = run_model(
        examples=examples,
        endpoint=args.base_endpoint,
        api_key=base_key,
        model=args.base_model,
        temperature=0.0,
    )
    tuned_predictions = run_model(
        examples=examples,
        endpoint=args.tuned_endpoint,
        api_key=tuned_key,
        model=args.tuned_model,
        temperature=0.0,
    )
    report = {
        "validation_path": str(args.validation),
        "n_validation_examples": len(examples),
        "models": {
            "base": {
                "model": args.base_model,
                "endpoint": args.base_endpoint,
                "metrics": compute_metrics(base_predictions),
                "self_consistency": run_self_consistency(
                    examples=examples,
                    endpoint=args.base_endpoint,
                    api_key=base_key,
                    model=args.base_model,
                    sample_size=args.self_consistency_sample,
                    runs=args.self_consistency_runs,
                ),
            },
            "tuned": {
                "model": args.tuned_model,
                "endpoint": args.tuned_endpoint,
                "metrics": compute_metrics(tuned_predictions),
                "self_consistency": run_self_consistency(
                    examples=examples,
                    endpoint=args.tuned_endpoint,
                    api_key=tuned_key,
                    model=args.tuned_model,
                    sample_size=args.self_consistency_sample,
                    runs=args.self_consistency_runs,
                ),
            },
        },
    }
    write_report(args.out, report)
    print(f"wrote evaluation report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
