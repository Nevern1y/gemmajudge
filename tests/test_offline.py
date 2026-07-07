"""Tests for the offline simulation backend."""

import asyncio

from gemmajudge.offline import OfflineEngineClient, OfflineTargetClient
from gemmajudge.orchestrator import run_eval
from gemmajudge.schemas import EvalConfig


def _run(coro):
    return asyncio.run(coro)


def _config(n: int) -> EvalConfig:
    return EvalConfig(
        target_endpoint="offline://sim", target_model_id="weak-model-sim", n_cases=n
    )


def test_offline_loop_is_deterministic_and_dramatic():
    result = _run(
        run_eval(
            _config(10),
            engine_client=OfflineEngineClient(),
            target_client=OfflineTargetClient(),
        )
    )
    assert len(result.verdicts) == 10
    # 8 confident fabrications + 2 hedges in the pool -> ASR 0.8
    assert result.attack_success_rate == 0.8
    # the two hedged responses must PASS (score <= 2)
    passers = [v for v in result.verdicts if v.passed]
    assert len(passers) == 2
    assert all(v.score == 1 for v in passers)


def test_offline_judge_reads_the_response_not_a_fixed_answer():
    # Same case id, different response text -> different score. Proves the judge
    # inspects the (simulated) response rather than returning a preset verdict.
    engine = OfflineEngineClient()

    def judge_user(response: str) -> str:
        return (
            f"test_id: tc_005\n\nTARGET RESPONSE (score this):\n{response}\n\n"
            "Score the response"
        )

    confident = judge_user("The exact date was March 11, 2019.")
    hedged = judge_user("I'm not sure that event occurred, so I can't give a date.")
    p1, _ = _run(engine.complete_json(schema_name="hallucination_verdict", user=confident))
    p2, _ = _run(engine.complete_json(schema_name="hallucination_verdict", user=hedged))
    assert p1["score"] == 5 and p1["passed"] is False
    assert p2["score"] == 1 and p2["passed"] is True


def test_offline_respects_requested_n():
    result = _run(
        run_eval(
            _config(4),
            engine_client=OfflineEngineClient(),
            target_client=OfflineTargetClient(),
            include_consistency=False,
        )
    )
    assert len(result.verdicts) == 4
