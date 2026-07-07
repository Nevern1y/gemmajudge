"""Tests for the robustness leaderboard: one Gemma attack set, many targets, ranked.

Uses the same mock style as ``test_orchestrator``: one engine mock serves both the
attacker JSON and the judge JSON, and each target returns scripted text. The judge
mock scores 5 when the target's response carries a ``HALLU`` marker and 1 otherwise,
so different targets earn different ASRs and the ranking is deterministic.
"""

import asyncio

from gemmajudge.client import ChatResult, LLMClient, LLMError
from gemmajudge.leaderboard import run_leaderboard
from gemmajudge.schemas import (
    AttackCase,
    EvalConfig,
    JudgeVerdict,
    LeaderboardResult,
    TargetReport,
    TokenUsage,
)


def _run(coro):
    return asyncio.run(coro)


def _config(n: int) -> EvalConfig:
    return EvalConfig(
        target_endpoint="http://targets/v1",
        target_model_id="unused-here",
        n_cases=n,
    )


class LbEngine(LLMClient):
    """Attacker: N cases. Judge: score 5 if the target response contains 'HALLU'."""

    def __init__(self, n: int, model_id: str = "gemma-3-27b-it") -> None:
        self.model_id = model_id
        self._n = n
        self.attacker_calls = 0
        self.judge_calls = 0
        self.closed = False

    async def complete_json(self, **kwargs):
        if kwargs["schema_name"] == "adversarial_test_cases":
            self.attacker_calls += 1
            tcs = [
                {"id": f"seed{i}", "prompt": f"Q{i}?", "rationale": "r",
                 "targeted_weakness": "w"}
                for i in range(1, self._n + 1)
            ]
            return (
                {"failure_mode": "hallucination", "test_cases": tcs},
                TokenUsage(prompt_tokens=100, completion_tokens=80),
            )
        # judge
        self.judge_calls += 1
        user = kwargs["user"]
        test_id = user.split("test_id: ")[1].split("\n")[0]
        score = 5 if "HALLU" in user else 1
        return (
            {"test_id": test_id, "target_response": "x", "score": score,
             "passed": score <= 2, "reasoning": "r", "evidence_span": "e"},
            TokenUsage(prompt_tokens=20, completion_tokens=10),
        )

    async def aclose(self):
        self.closed = True


class AlwaysHallu(LLMClient):
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.closed = False

    async def complete(self, **kwargs):
        return ChatResult(text="HALLU: confidently wrong.",
                          usage=TokenUsage(prompt_tokens=10, completion_tokens=8))

    async def aclose(self):
        self.closed = True


class NeverHallu(LLMClient):
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.closed = False

    async def complete(self, **kwargs):
        return ChatResult(text="That premise is false; I won't guess.",
                          usage=TokenUsage(prompt_tokens=10, completion_tokens=8))

    async def aclose(self):
        self.closed = True


class HalfHallu(LLMClient):
    """Hallucinates on even-numbered prompts (Q2, Q4, ...) -> deterministic 50%."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.closed = False

    async def complete(self, **kwargs):
        prompt = kwargs["user"]
        n = int("".join(ch for ch in prompt if ch.isdigit()) or "1")
        text = "HALLU: made up." if n % 2 == 0 else "I don't know."
        return ChatResult(text=text, usage=TokenUsage(prompt_tokens=10, completion_tokens=8))

    async def aclose(self):
        self.closed = True


class DeadTarget(LLMClient):
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.closed = False

    async def complete(self, **kwargs):
        raise LLMError("target endpoint unreachable")

    async def aclose(self):
        self.closed = True


def test_attack_set_generated_once_and_shared():
    engine = LbEngine(4)
    targets = [AlwaysHallu("a"), NeverHallu("b")]
    result = _run(run_leaderboard(_config(4), engine_client=engine, target_clients=targets))
    assert engine.attacker_calls == 1  # ONE shared attack set for all targets
    assert len(result.attacks) == 4
    # both targets scored over the full shared set
    assert all(len(t.verdicts) == 4 for t in result.targets)
    # judge ran once per (target, case)
    assert engine.judge_calls == 2 * 4


def test_targets_ranked_by_asr():
    engine = LbEngine(4)
    targets = [NeverHallu("safe"), AlwaysHallu("bad"), HalfHallu("mid")]
    result = _run(run_leaderboard(_config(4), engine_client=engine, target_clients=targets))

    ranked = result.ranked
    assert [t.target_model_id for t in ranked] == ["bad", "mid", "safe"]
    asr = {t.target_model_id: t.attack_success_rate for t in result.targets}
    assert asr == {"bad": 1.0, "mid": 0.5, "safe": 0.0}


def test_target_report_shapes():
    engine = LbEngine(4)
    result = _run(run_leaderboard(_config(4), engine_client=engine,
                                  target_clients=[HalfHallu("mid")]))
    report = result.targets[0]
    assert report.n_cases == 4
    assert report.n_failed == 2
    assert report.attack_success_rate == 0.5
    assert 1.0 <= report.mean_score <= 5.0
    assert report.error is None


def test_dead_target_scored_via_sentinel_not_error():
    # query_target swallows the transport error into a sentinel, which the judge
    # scores low -> the target still gets a full report (ASR 0), no error field set.
    engine = LbEngine(3)
    result = _run(run_leaderboard(_config(3), engine_client=engine,
                                  target_clients=[DeadTarget("down")]))
    report = result.targets[0]
    assert report.error is None
    assert report.n_cases == 3
    assert report.attack_success_rate == 0.0


def test_close_clients_flag():
    engine = LbEngine(2)
    targets = [AlwaysHallu("a"), NeverHallu("b")]
    _run(run_leaderboard(_config(2), engine_client=engine, target_clients=targets,
                         close_clients=True))
    assert engine.closed is True
    assert all(t.closed for t in targets)


def test_injected_clients_not_closed_by_default():
    engine = LbEngine(2)
    targets = [AlwaysHallu("a")]
    _run(run_leaderboard(_config(2), engine_client=engine, target_clients=targets))
    assert engine.closed is False
    assert targets[0].closed is False


def test_cost_and_engine_label_populated(settings):
    engine = LbEngine(3)
    result = _run(run_leaderboard(_config(3), engine_client=engine,
                                  target_clients=[AlwaysHallu("a")], settings=settings))
    assert result.engine_model_id == "gemma-3-27b-it"
    assert result.inference_backend == "fireworks"
    assert result.cost is not None
    assert result.cost.usd > 0  # settings fixture prices engine tokens


# --- schema-level unit tests (no run) --------------------------------------


def _verdict(test_id: str, score: int) -> JudgeVerdict:
    return JudgeVerdict(
        test_id=test_id, target_response="r", score=score,
        passed=score <= 2, reasoning="x", evidence_span="y",
    )


def test_leaderboard_ranked_tiebreak_on_mean():
    # Two targets with equal ASR (both 1/2) rank by mean score next.
    hi = TargetReport(target_model_id="hi", verdicts=[_verdict("tc_001", 5), _verdict("tc_002", 3)])
    lo = TargetReport(target_model_id="lo", verdicts=[_verdict("tc_001", 4), _verdict("tc_002", 1)])
    board = LeaderboardResult(targets=[lo, hi])
    assert hi.attack_success_rate == lo.attack_success_rate  # both 0.5
    assert [t.target_model_id for t in board.ranked] == ["hi", "lo"]  # mean breaks tie


def test_cases_for_joins_verdicts_to_attacks():
    attacks = [
        AttackCase(id="tc_001", prompt="p1", rationale="r", targeted_weakness="w"),
        AttackCase(id="tc_002", prompt="p2", rationale="r", targeted_weakness="w"),
    ]
    target = TargetReport(target_model_id="t", verdicts=[_verdict("tc_002", 5)])
    board = LeaderboardResult(attacks=attacks, targets=[target])
    cases = board.cases_for(target)
    assert len(cases) == 1
    assert cases[0].attack.id == "tc_002"
    assert cases[0].verdict.score == 5


def test_empty_target_report_is_safe():
    report = TargetReport(target_model_id="t", error="boom")
    assert report.n_cases == 0
    assert report.attack_success_rate == 0.0
    assert report.mean_score == 0.0
