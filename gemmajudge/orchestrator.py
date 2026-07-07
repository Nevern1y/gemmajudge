"""The orchestrator: the single entrypoint the UI calls.

``run_eval(config) -> EvalResult`` is the frozen integration seam (AGENTS.md §3).
It runs the full closed loop:

1. **Attack** — one call: Gemma generates N adversarial cases.
2. **Run + Judge** — fan out over the N cases with an ``asyncio`` concurrency cap;
   each case queries the target then judges the response. Bounded concurrency keeps
   us under the **30s/request** hard rule (WORK_SPLIT standing rule) while still
   parallelizing.
3. **Aggregate** — wall-clock, throughput, per-role token usage → $ (cost meter),
   and the on-screen AMD backend/model labels.
4. **F9b self-consistency** *(off the live path)* — re-judge 1–3 showcase cases K
   times and report the score spread, the cheap credible answer to "can the judge be
   trusted?".

Everything is injectable (``settings``, ``engine_client``, ``target_client``), so the
whole loop is unit-testable with mocks and zero network. In production, pass nothing
and it builds clients from the environment via :func:`gemmajudge.config.load_settings`.
"""

from __future__ import annotations

import asyncio
import time

from gemmajudge.attacker import generate_attacks
from gemmajudge.client import LLMClient, make_engine_client, make_target_client
from gemmajudge.config import Settings, load_settings
from gemmajudge.judge import fallback_verdict, judge
from gemmajudge.schemas import (
    AttackCase,
    ConsistencyResult,
    CostReport,
    EvalConfig,
    EvalResult,
    JudgeVerdict,
    RunMetrics,
    TokenUsage,
)
from gemmajudge.target import query_target

# F9b defaults: re-judge this many showcase cases, this many times each. Small and
# fixed so it stays cheap and off the live-latency budget.
_CONSISTENCY_CASES = 3
_CONSISTENCY_REPEATS = 3


async def _run_one_case(
    engine_client: LLMClient,
    target_client: LLMClient,
    case: AttackCase,
    failure_mode,
    semaphore: asyncio.Semaphore,
) -> tuple[JudgeVerdict, TokenUsage, TokenUsage]:
    """Target → Judge for a single case. Returns (verdict, target_usage, judge_usage).

    Never raises for expected model failures: the target module returns a sentinel
    on error, and a judge failure degrades to an explicit fallback verdict — so one
    bad case can't abort the batch."""
    async with semaphore:
        response, target_usage = await query_target(target_client, case.prompt)
        try:
            verdict, judge_usage = await judge(
                engine_client, case, response, failure_mode=failure_mode
            )
        except Exception as exc:  # noqa: BLE001 - degrade, don't abort the run
            verdict = fallback_verdict(case, response, reason=str(exc)[:120])
            judge_usage = TokenUsage()
    return verdict, target_usage, judge_usage


async def _compute_consistency(
    engine_client: LLMClient,
    cases: list[AttackCase],
    verdicts_by_id: dict[str, JudgeVerdict],
    failure_mode,
    *,
    n_cases: int = _CONSISTENCY_CASES,
    repeats: int = _CONSISTENCY_REPEATS,
) -> tuple[list[ConsistencyResult], TokenUsage]:
    """Re-judge a few showcase cases K times; report the score spread (PRD F9b).

    Runs AFTER the timed live path, so its extra judge calls never count against the
    30s rule. Picks the highest-scoring (most dramatic) cases as the showcase."""
    showcase = sorted(
        cases,
        key=lambda c: verdicts_by_id[c.id].score if c.id in verdicts_by_id else 0,
        reverse=True,
    )[:n_cases]

    results: list[ConsistencyResult] = []
    usage = TokenUsage()
    for case in showcase:
        base = verdicts_by_id.get(case.id)
        response = base.target_response if base else ""
        # First data point is the live verdict we already have; re-judge (repeats-1)×.
        scores: list[int] = [base.score] if base else []
        for _ in range(max(0, repeats - 1)):
            try:
                verdict, u = await judge(
                    engine_client, case, response, failure_mode=failure_mode
                )
                scores.append(verdict.score)
                usage = usage + u
            except Exception:  # noqa: BLE001 - a dropped re-judge just shrinks the sample
                continue
        if scores:
            results.append(ConsistencyResult(test_id=case.id, scores=scores))
    return results, usage


async def run_eval(
    config: EvalConfig,
    *,
    settings: Settings | None = None,
    engine_client: LLMClient | None = None,
    target_client: LLMClient | None = None,
    include_consistency: bool = True,
) -> EvalResult:
    """Run one full evaluation and return the aggregated result (the UI seam).

    Args:
        config: user-selected run config (failure mode, N, target endpoint/model).
        settings: engine settings; loaded from env if not supplied.
        engine_client / target_client: injected for tests; built from ``settings``
            otherwise.
        include_consistency: run the F9b self-consistency pass (off the live path).

    Returns:
        An :class:`EvalResult` populated with ``attacks``, ``verdicts``, ``cost``,
        ``metrics`` and ``consistency`` — everything the report/drill-down/AMD panel
        need.
    """
    owns_clients = engine_client is None or target_client is None
    if owns_clients:
        settings = settings or load_settings()
        engine_client = engine_client or make_engine_client(settings)
        target_client = target_client or make_target_client(settings)

    max_concurrency = settings.max_concurrency if settings else 8
    backend_label = settings.backend.value if settings else ""
    engine_model = engine_client.model_id

    try:
        # --- timed live path (this is what the 30s rule governs) -------------
        start = time.monotonic()

        attacks, attacker_usage = await generate_attacks(engine_client, config)

        semaphore = asyncio.Semaphore(max_concurrency)
        results = await asyncio.gather(
            *(
                _run_one_case(
                    engine_client, target_client, case, config.failure_mode, semaphore
                )
                for case in attacks
            )
        )
        wall_clock = time.monotonic() - start

        verdicts = [r[0] for r in results]
        target_usage = sum((r[1] for r in results), TokenUsage())
        judge_usage = sum((r[2] for r in results), TokenUsage())

        # --- F9b: off the timed path -----------------------------------------
        consistency: list[ConsistencyResult] = []
        if include_consistency and verdicts:
            verdicts_by_id = {v.test_id: v for v in verdicts}
            consistency, consistency_usage = await _compute_consistency(
                engine_client, attacks, verdicts_by_id, config.failure_mode
            )
            judge_usage = judge_usage + consistency_usage
    finally:
        if owns_clients:
            # We created the clients; we close them. Injected clients are the
            # caller's to manage.
            await _safe_close(engine_client)
            await _safe_close(target_client)

    # --- aggregate cost + metrics -------------------------------------------
    cost = _build_cost(settings, attacker_usage, target_usage, judge_usage)
    metrics = RunMetrics(
        wall_clock_seconds=wall_clock,
        n_cases=len(verdicts),
        inference_backend=backend_label,
        model_id=engine_model,
        target_model_id=config.target_model_id,
    )

    return EvalResult(
        config=config,
        verdicts=verdicts,
        attacks=attacks,
        cost=cost,
        metrics=metrics,
        consistency=consistency,
    )


def _build_cost(
    settings: Settings | None,
    attacker: TokenUsage,
    target: TokenUsage,
    judge_usage: TokenUsage,
) -> CostReport:
    """Turn measured per-role usage into a CostReport with a $ figure.

    The $ figure prices the **engine** (Attacker+Judge) tokens, which is what runs
    on the AMD-hosted Gemma; the target is a separate system whose price we don't
    assume. If no pricing is configured, ``usd`` is a truthful ``0.0``."""
    pricing = settings.pricing if settings else None
    if pricing is not None:
        engine_tokens = attacker + judge_usage
        usd = pricing.cost_usd(engine_tokens.prompt_tokens, engine_tokens.completion_tokens)
        source = pricing.source
    else:
        usd = 0.0
        source = None
    return CostReport(
        attacker=attacker,
        target=target,
        judge=judge_usage,
        usd=usd,
        price_source=source,
    )


async def _safe_close(client: LLMClient | None) -> None:
    if client is None:
        return
    try:
        await client.aclose()
    except Exception:  # noqa: BLE001 - closing must never mask the real result
        pass
