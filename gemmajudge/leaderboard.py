"""Robustness leaderboard: one Gemma attack set, many targets, ranked by ASR.

This is GemmaJudge's "unique product" surface. Where :func:`orchestrator.run_eval`
evaluates a *single* target, ``run_leaderboard`` generates **one** adversarial set
with the Gemma attacker and runs it against **every** target model, then ranks them
by Attack Success Rate. Because every target sees the identical prompts, the ranking
is apples-to-apples: a self-hosted, open-weight red-team + judge that benchmarks any
model's hallucination robustness on your own AMD hardware.

It reuses the same primitives as the orchestrator (``generate_attacks``,
``query_target``, ``judge``) — nothing about the frozen ``run_eval`` seam changes.
Everything is injectable (``engine_client``, ``target_clients``) so it is unit-testable
with mocks and zero network, and callable from a CLI or the UI.
"""

from __future__ import annotations

import asyncio
import time

from gemmajudge.attacker import generate_attacks
from gemmajudge.client import LLMClient
from gemmajudge.config import Settings
from gemmajudge.judge import fallback_verdict, judge
from gemmajudge.schemas import (
    AttackCase,
    EvalConfig,
    FailureMode,
    JudgeVerdict,
    LeaderboardResult,
    TargetReport,
    TokenUsage,
)
from gemmajudge.target import query_target
from gemmajudge.utils.cost import build_cost


async def _judge_one(
    engine_client: LLMClient,
    target_client: LLMClient,
    case: AttackCase,
    failure_mode: FailureMode,
    semaphore: asyncio.Semaphore,
) -> tuple[JudgeVerdict, TokenUsage, TokenUsage]:
    """Target -> Judge for one case; never raises (mirrors orchestrator._run_one_case).

    Returns ``(verdict, target_usage, judge_usage)``. A target error yields a sentinel
    response; a judge error degrades to an explicit fallback verdict, and any tokens
    the failed judge already spent are recovered from ``LLMError.usage``."""
    async with semaphore:
        response, target_usage = await query_target(target_client, case.prompt)
        try:
            verdict, judge_usage = await judge(
                engine_client, case, response, failure_mode=failure_mode
            )
        except Exception as exc:  # noqa: BLE001 - degrade, don't abort the board
            verdict = fallback_verdict(
                case,
                response,
                reason=str(exc)[:120],
                failure_mode=failure_mode,
            )
            judge_usage = getattr(exc, "usage", None) or TokenUsage()
    return verdict, target_usage, judge_usage


async def _evaluate_target(
    engine_client: LLMClient,
    target_client: LLMClient,
    attacks: list[AttackCase],
    failure_mode: FailureMode,
    max_concurrency: int,
) -> tuple[TargetReport, TokenUsage, TokenUsage]:
    """Run the shared attack set against one target; return its report + usage.

    A transport failure that escapes the per-case guard (unexpected) is captured on
    the report's ``error`` field rather than propagated, so the remaining targets
    still get scored."""
    semaphore = asyncio.Semaphore(max_concurrency)
    start = time.monotonic()
    try:
        results = await asyncio.gather(
            *(
                _judge_one(engine_client, target_client, case, failure_mode, semaphore)
                for case in attacks
            )
        )
    except Exception as exc:  # noqa: BLE001 - one target dying must not sink the board
        elapsed = time.monotonic() - start
        report = TargetReport(
            target_model_id=target_client.model_id,
            wall_clock_seconds=elapsed,
            error=str(exc)[:200],
        )
        return report, TokenUsage(), TokenUsage()

    elapsed = time.monotonic() - start
    verdicts = [r[0] for r in results]
    target_usage = sum((r[1] for r in results), TokenUsage())
    judge_usage = sum((r[2] for r in results), TokenUsage())
    report = TargetReport(
        target_model_id=target_client.model_id,
        verdicts=verdicts,
        wall_clock_seconds=elapsed,
    )
    return report, target_usage, judge_usage


async def run_leaderboard(
    config: EvalConfig,
    engine_client: LLMClient,
    target_clients: list[LLMClient],
    *,
    settings: Settings | None = None,
    max_concurrency: int = 6,
    close_clients: bool = False,
) -> LeaderboardResult:
    """Generate one Gemma attack set and rank every target by ASR.

    Args:
        config: run config; ``failure_mode`` selects the attacker prompt and
            ``n_cases`` the attack-set size. (``target_endpoint``/``target_model_id``
            are unused here — targets come from ``target_clients``.)
        engine_client: the Gemma Attacker + Judge.
        target_clients: the systems-under-test; each carries its own ``model_id``.
        settings: optional, only for the on-screen backend label and cost pricing.
        max_concurrency: per-target fan-out cap; request duration is bounded by the client
            timeout.
        close_clients: if True, close the engine and every target client on exit
            (CLI owns its clients; the UI/tests manage their own).

    Returns:
        A :class:`LeaderboardResult` with the shared ``attacks``, one ``TargetReport``
        per target (ranked via ``.ranked``), and an aggregate cost.
    """
    attacker_usage = TokenUsage()
    target_usage_total = TokenUsage()
    judge_usage_total = TokenUsage()
    reports: list[TargetReport] = []
    try:
        attacks, attacker_usage = await generate_attacks(engine_client, config)

        # Targets are evaluated one after another (not fanned out across targets):
        # a shared dedicated Gemma judge is the bottleneck, and sequential targets
        # keep each one's latency clean for the per-target throughput number.
        for target_client in target_clients:
            report, t_usage, j_usage = await _evaluate_target(
                engine_client,
                target_client,
                attacks,
                config.failure_mode,
                max_concurrency,
            )
            reports.append(report)
            target_usage_total = target_usage_total + t_usage
            judge_usage_total = judge_usage_total + j_usage
    finally:
        if close_clients:
            await _safe_close(engine_client)
            for target_client in target_clients:
                await _safe_close(target_client)

    cost = build_cost(settings, attacker_usage, target_usage_total, judge_usage_total)
    return LeaderboardResult(
        failure_mode=config.failure_mode,
        engine_model_id=engine_client.model_id,
        inference_backend=settings.backend.value if settings else "",
        attacks=attacks,
        targets=reports,
        cost=cost,
    )


async def _safe_close(client: LLMClient | None) -> None:
    if client is None:
        return
    try:
        await client.aclose()
    except Exception:  # noqa: BLE001 - closing must never mask the real result
        pass
