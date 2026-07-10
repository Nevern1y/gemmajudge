"""Run GemmaJudge against local AMD vLLM endpoints and save proof JSON."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Make direct ``python scripts/run_amd_proof.py`` invocations work from any directory.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gemmajudge.config import load_settings  # noqa: E402
from gemmajudge.orchestrator import run_eval  # noqa: E402
from gemmajudge.schemas import EvalConfig, EvalResult, FailureMode  # noqa: E402


def _request_timeout(value: str) -> float:
    timeout = float(value)
    if not 0 < timeout <= 30:
        raise argparse.ArgumentTypeError("must be greater than 0 and at most 30 seconds")
    return timeout


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-model", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--engine-endpoint", default="http://localhost:8000/v1")
    parser.add_argument("--target-endpoint", default="http://localhost:8001/v1")
    parser.add_argument("--backend-label", required=True)
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--request-timeout", type=_request_timeout, default=25.0)
    parser.add_argument("--max-concurrency", type=int, default=6)
    parser.add_argument("--no-consistency", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser


async def _run(args: argparse.Namespace) -> EvalResult:
    settings = load_settings(
        {
            "INFERENCE_BACKEND": "mi300x",
            "MODEL_ID": args.engine_model,
            "MI300X_BASE_URL": args.engine_endpoint,
            "TARGET_ENDPOINT": args.target_endpoint,
            "TARGET_MODEL_ID": args.target_model,
            "REQUEST_TIMEOUT_S": str(args.request_timeout),
            "MAX_CONCURRENCY": str(args.max_concurrency),
        }
    )
    config = EvalConfig(
        failure_mode=FailureMode.HALLUCINATION,
        n_cases=args.n,
        target_endpoint=args.target_endpoint,
        target_model_id=args.target_model,
    )
    result = await run_eval(
        config,
        settings=settings,
        include_consistency=not args.no_consistency,
    )
    if result.metrics:
        result.metrics.inference_backend = args.backend_label

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = asyncio.run(_run(args))
    failed = sum(1 for verdict in result.verdicts if verdict.score >= 4)
    summary = {
        "output": str(args.output),
        "engine_model": result.metrics.model_id if result.metrics else args.engine_model,
        "target_model": result.config.target_model_id,
        "backend": result.metrics.inference_backend if result.metrics else args.backend_label,
        "n_cases": len(result.verdicts),
        "failed_cases": failed,
        "attack_success_rate": result.attack_success_rate,
        "full_pipeline_wall_clock_seconds": (
            result.metrics.wall_clock_seconds if result.metrics else None
        ),
        "request_timeout_seconds": args.request_timeout,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
