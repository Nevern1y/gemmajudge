from __future__ import annotations

from gemmajudge.config import Settings
from gemmajudge.schemas import CostReport, TokenUsage


def build_cost(
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
