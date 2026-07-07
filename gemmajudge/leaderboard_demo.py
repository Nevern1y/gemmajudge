"""CLI: rank many target models by hallucination robustness with one Gemma judge.

Usage::

    # engine (Gemma attacker+judge) + endpoint/key come from env / .env
    python -m gemmajudge.leaderboard_demo --n 8 \
        --targets accounts/fireworks/models/glm-5p1,accounts/fireworks/models/kimi-k2p6

Targets default to ``LEADERBOARD_TARGETS`` (comma-separated) or, failing that, the
single ``TARGET_MODEL_ID`` from the environment. Each target is reached on the same
endpoint/key as the configured target (they are all OpenAI-compatible). Prints a
ranked table and writes the full :class:`LeaderboardResult` JSON to ``--out``.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from pydantic import SecretStr, ValidationError

from gemmajudge.client import LLMClient, make_engine_client
from gemmajudge.config import ConfigError, EndpointSettings, Settings, load_settings
from gemmajudge.leaderboard import run_leaderboard
from gemmajudge.schemas import EvalConfig, FailureMode, LeaderboardResult, TargetReport


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m gemmajudge.leaderboard_demo",
        description="Rank target models by ASR under one Gemma-generated attack set.",
    )
    parser.add_argument("--n", type=int, default=8, help="attack-set size (default 8)")
    parser.add_argument(
        "--mode",
        default=FailureMode.HALLUCINATION.value,
        choices=[m.value for m in FailureMode],
        help="failure mode (default hallucination)",
    )
    parser.add_argument(
        "--targets",
        help="comma-separated target model ids (overrides LEADERBOARD_TARGETS / env)",
    )
    parser.add_argument("--out", help="write the LeaderboardResult JSON to this path")
    return parser


def _target_ids(args: argparse.Namespace, settings: Settings) -> list[str]:
    """Resolve the target model-id list from --targets, env, or the single target."""
    import os

    raw = args.targets or os.environ.get("LEADERBOARD_TARGETS") or settings.target.model_id
    ids = [t.strip() for t in raw.split(",") if t.strip()]
    # De-dup while preserving order.
    seen: set[str] = set()
    ordered: list[str] = []
    for mid in ids:
        if mid not in seen:
            seen.add(mid)
            ordered.append(mid)
    return ordered


def _make_target_clients(settings: Settings, model_ids: list[str]) -> list[LLMClient]:
    """One client per target model id, all on the configured target endpoint/key."""
    clients: list[LLMClient] = []
    for mid in model_ids:
        endpoint = EndpointSettings(
            base_url=settings.target.base_url,
            api_key=SecretStr(settings.target.api_key.get_secret_value()),
            model_id=mid,
        )
        clients.append(LLMClient.from_endpoint(endpoint, timeout=settings.request_timeout_s))
    return clients


def _short(model_id: str) -> str:
    """Trim ``accounts/fireworks/models/gemma-3-4b-it`` to ``gemma-3-4b-it``."""
    return model_id.rsplit("/", 1)[-1]


def _print_report(result: LeaderboardResult) -> None:
    line = "=" * 66
    print(line)
    print("  GemmaJudge - Robustness Leaderboard")
    print(f"  Failure mode : {result.failure_mode.value}")
    print(f"  Judge/Attack : {_short(result.engine_model_id)}  (backend: "
          f"{result.inference_backend or 'injected'})")
    print(f"  Attack set   : {len(result.attacks)} Gemma-generated adversarial prompts")
    print(line)
    print(f"  {'#':<3}{'target model':<26}{'ASR':>7}{'failed':>9}{'mean':>7}{'time':>8}")
    print("  " + "-" * 60)
    for rank, tgt in enumerate(result.ranked, start=1):
        if tgt.error:
            print(f"  {rank:<3}{_short(tgt.target_model_id):<26}{'ERR':>7}   {tgt.error[:26]}")
            continue
        print(
            f"  {rank:<3}{_short(tgt.target_model_id):<26}"
            f"{tgt.attack_success_rate:>6.0%}"
            f"{tgt.n_failed:>4}/{tgt.n_cases:<4}"
            f"{tgt.mean_score:>7.2f}"
            f"{tgt.wall_clock_seconds:>7.1f}s"
        )
    print(line)
    print("  ASR = fraction of prompts the target failed (judge score >= 4).")
    print("  Higher ASR = more hallucination-prone under Gemma's adversarial probing.")
    c = result.cost
    if c:
        print(f"  Measured tokens: {c.total.total_tokens:,}"
              + (f"  |  cost ${c.usd:.4f} ({c.price_source})" if c.price_source else ""))
    print(line)
    _print_headline_drilldown(result)


def _print_headline_drilldown(result: LeaderboardResult) -> None:
    """Show the single most-confident failure across the whole board."""
    worst_target: TargetReport | None = None
    worst_score = 0
    for tgt in result.targets:
        for v in tgt.verdicts:
            if v.score > worst_score:
                worst_score, worst_target = v.score, tgt
    if worst_target is None or worst_score < 4:
        print("  No failures surfaced - every target resisted the attack set.")
        print("=" * 66)
        return
    cases = {c.verdict.test_id: c for c in result.cases_for(worst_target)}
    top = max(worst_target.verdicts, key=lambda v: v.score)
    case = cases.get(top.test_id)
    print("  Headline failure (most confident hallucination):")
    print(f"    Target      : {_short(worst_target.target_model_id)}  (score {top.score}/5)")
    if case:
        print(f"    Gemma attack: {case.attack.prompt[:88]}")
    print(f"    Judge says  : {top.reasoning[:88]}")
    print(f"    Evidence    : {top.evidence_span[:88]!r}")
    print("=" * 66)


async def _run(args: argparse.Namespace) -> LeaderboardResult:
    settings = load_settings()
    engine_client = make_engine_client(settings)
    target_ids = _target_ids(args, settings)
    target_clients = _make_target_clients(settings, target_ids)
    config = EvalConfig(
        failure_mode=FailureMode(args.mode),
        n_cases=args.n,
        target_endpoint=settings.target.base_url,
        target_model_id=target_ids[0],
    )
    return await run_leaderboard(
        config,
        engine_client=engine_client,
        target_clients=target_clients,
        settings=settings,
        max_concurrency=settings.max_concurrency,
        close_clients=True,
    )


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):  # pragma: no cover
        pass
    args = _build_parser().parse_args(argv)
    try:
        result = asyncio.run(_run(args))
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    except ValidationError as exc:
        print(f"Invalid arguments: {exc}", file=sys.stderr)
        return 2
    _print_report(result)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(result.model_dump_json(indent=2))
        print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
