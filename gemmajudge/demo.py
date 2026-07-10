"""Command-line eval runner — proves the whole loop from the terminal.

Usage::

    python -m gemmajudge.demo --offline            # zero keys, simulated backend
    python -m gemmajudge.demo --n 20               # real backend, from .env / env
    python -m gemmajudge.demo --endpoint http://localhost:8000/v1 --model my-model

Real mode reads configuration via :func:`gemmajudge.config.load_settings` (fails
loudly if under-configured). ``--offline`` swaps in the simulated backend and prints
a loud ``SIMULATED`` banner — never mistake its output for a real Gemma/AMD run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from pydantic import ValidationError

from gemmajudge.config import ConfigError, load_settings
from gemmajudge.offline import OfflineEngineClient, OfflineTargetClient
from gemmajudge.orchestrator import run_eval
from gemmajudge.schemas import EvalConfig, EvalResult, FailureMode


def _fraction(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m gemmajudge.demo",
        description="Run one GemmaJudge adversarial evaluation and print the report.",
    )
    parser.add_argument("--n", type=int, default=10, help="number of test cases (default 10)")
    parser.add_argument(
        "--mode",
        default=FailureMode.HALLUCINATION.value,
        choices=[m.value for m in FailureMode],
        help="failure mode (default hallucination)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="use the SIMULATED backend (no keys, no network)",
    )
    parser.add_argument("--endpoint", help="target endpoint (real mode; overrides env)")
    parser.add_argument("--model", help="target model id (real mode; overrides env)")
    parser.add_argument(
        "--no-consistency",
        action="store_true",
        help="skip the F9b judge self-consistency pass",
    )
    parser.add_argument(
        "--max-asr",
        type=_fraction,
        help="fail with exit code 1 when attack success rate exceeds this fraction",
    )
    parser.add_argument(
        "--max-failed-cases",
        type=_non_negative_int,
        help="fail with exit code 1 when more than this many cases score 4 or 5",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print one machine-readable JSON summary instead of the text report",
    )
    return parser


def _bar(count: int, total: int, width: int = 24) -> str:
    filled = round(width * count / total) if total else 0
    return "#" * filled + "." * (width - filled)


def _print_report(result: EvalResult, *, offline: bool) -> None:
    m = result.metrics
    c = result.cost
    line = "=" * 60
    print(line)
    if offline:
        print("  [!] SIMULATED RUN - illustrative only, NOT a real model/AMD run")
        print(line)
    print(f"  Failure mode : {result.config.failure_mode.value}")
    print(f"  Backend      : {m.inference_backend or '(injected)'}")
    print(f"  Judge/Attack : {m.model_id}")
    print(f"  Target       : {m.target_model_id}")
    print(line)
    asr = result.attack_success_rate
    failed = sum(1 for v in result.verdicts if v.score >= 4)
    print(f"  Attack Success Rate : {asr:.0%}  ({failed}/{len(result.verdicts)} failed)")
    print(f"  Full pipeline clock : {m.wall_clock_seconds:.2f}s"
          f"   |  throughput {m.throughput_evals_per_sec:.1f} evals/s")
    tokens = c.total.total_tokens if c else 0
    usd = c.usd if c else 0.0
    src = f"  (price src: {c.price_source})" if c and c.price_source else ""
    print(f"  Cost                : ${usd:.4f} over {tokens} tokens{src}")
    print(line)

    print("  Score distribution (1=safe .. 5=confident&false):")
    counts = {s: 0 for s in range(1, 6)}
    for v in result.verdicts:
        counts[v.score] += 1
    total = len(result.verdicts)
    for score in range(1, 6):
        print(f"    {score} | {_bar(counts[score], total)} {counts[score]}")
    print(line)

    if result.consistency:
        print("  Judge self-consistency (F9b) on showcase cases:")
        for cr in result.consistency:
            spread = ",".join(str(s) for s in cr.scores)
            print(f"    {cr.test_id}: [{spread}] -> mean {cr.mean:.2f}, stdev {cr.stdev:.2f}")
        print(line)

    worst = _worst_case(result)
    if worst is not None:
        print("  Worst case (drill-down):")
        print(f"    Attacker prompt : {worst.attack.prompt}")
        print(f"    Target response : {worst.verdict.target_response}")
        print(f"    Judge score     : {worst.verdict.score}/5  (passed={worst.verdict.passed})")
        print(f"    Judge reasoning : {worst.verdict.reasoning}")
        print(f"    Evidence span   : {worst.verdict.evidence_span!r}")
        print(line)


def _worst_case(result: EvalResult):
    cases = result.cases
    if not cases:
        return None
    return max(cases, key=lambda c: c.verdict.score)


def _gate_failures(result: EvalResult, args: argparse.Namespace) -> list[str]:
    failures: list[str] = []
    failed_cases = sum(1 for verdict in result.verdicts if verdict.score >= 4)
    if args.max_asr is not None and result.attack_success_rate > args.max_asr:
        failures.append(
            f"ASR {result.attack_success_rate:.3f} exceeds maximum {args.max_asr:.3f}"
        )
    if args.max_failed_cases is not None and failed_cases > args.max_failed_cases:
        failures.append(
            f"failed cases {failed_cases} exceeds maximum {args.max_failed_cases}"
        )
    return failures


def _print_gate(failures: list[str]) -> None:
    if failures:
        print("  REGRESSION GATE: FAILED")
        for failure in failures:
            print(f"    - {failure}")
    else:
        print("  REGRESSION GATE: PASSED")


def _print_json_summary(
    result: EvalResult,
    *,
    offline: bool,
    args: argparse.Namespace,
    failures: list[str],
) -> None:
    metrics = result.metrics
    failed_cases = sum(1 for verdict in result.verdicts if verdict.score >= 4)
    summary = {
        "schema_version": 1,
        "status": "failed" if failures else "passed",
        "simulated": offline,
        "failure_mode": result.config.failure_mode.value,
        "n_cases": len(result.verdicts),
        "failed_cases": failed_cases,
        "attack_success_rate": result.attack_success_rate,
        "wall_clock_seconds": metrics.wall_clock_seconds if metrics else None,
        "inference_backend": metrics.inference_backend if metrics else None,
        "model_id": metrics.model_id if metrics else None,
        "target_model_id": result.config.target_model_id,
        "thresholds": {
            "max_asr": args.max_asr,
            "max_failed_cases": args.max_failed_cases,
        },
        "failures": failures,
    }
    print(json.dumps(summary, sort_keys=True))


async def _run(args: argparse.Namespace) -> EvalResult:
    mode = FailureMode(args.mode)
    if args.offline:
        config = EvalConfig(
            failure_mode=mode,
            n_cases=args.n,
            target_endpoint=args.endpoint or "offline://simulated",
            target_model_id=args.model or "weak-model-sim",
        )
        return await run_eval(
            config,
            engine_client=OfflineEngineClient(),
            target_client=OfflineTargetClient(),
            include_consistency=not args.no_consistency,
        )

    settings = load_settings()
    config = EvalConfig(
        failure_mode=mode,
        n_cases=args.n,
        target_endpoint=args.endpoint or settings.target.base_url,
        target_model_id=args.model or settings.target.model_id,
    )
    return await run_eval(
        config,
        settings=settings,
        include_consistency=not args.no_consistency,
    )


def main(argv: list[str] | None = None) -> int:
    # Prefer UTF-8 output where the terminal supports it; never crash if it doesn't.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):  # pragma: no cover - non-reconfigurable stream
        pass
    args = _build_parser().parse_args(argv)
    try:
        result = asyncio.run(_run(args))
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        print("Tip: run with --offline to try the simulated demo without keys.", file=sys.stderr)
        return 2
    except ValidationError as exc:
        # e.g. --n outside EvalConfig's allowed [1, 100] range.
        print(f"Invalid arguments: {exc}", file=sys.stderr)
        return 2
    failures = _gate_failures(result, args)
    if args.json:
        _print_json_summary(result, offline=args.offline, args=args, failures=failures)
    else:
        _print_report(result, offline=args.offline)
        if args.max_asr is not None or args.max_failed_cases is not None:
            _print_gate(failures)
    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
