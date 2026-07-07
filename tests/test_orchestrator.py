"""End-to-end tests for run_eval() — the UI seam — with mocked clients."""

import asyncio

from gemmajudge.client import ChatResult, LLMClient
from gemmajudge.orchestrator import run_eval
from gemmajudge.schemas import TokenUsage


def _run(coro):
    return asyncio.run(coro)


class EngineMock(LLMClient):
    """Serves attacker JSON and judge JSON off one client, like the real engine."""

    def __init__(self, scores):
        self.model_id = "gemma-31b-it"
        self._scores = scores  # score per case index (1-based via test_id)
        self.judge_calls = 0
        self.closed = False

    async def complete_json(self, **kwargs):
        if kwargs["schema_name"] == "adversarial_test_cases":
            tcs = [
                {"id": f"seed{i}", "prompt": f"Fake question {i}?",
                 "rationale": "r", "targeted_weakness": "w"}
                for i in range(1, len(self._scores) + 1)
            ]
            return ({"failure_mode": "hallucination", "test_cases": tcs},
                    TokenUsage(prompt_tokens=200, completion_tokens=300))
        # judge
        self.judge_calls += 1
        user = kwargs["user"]
        test_id = user.split("test_id: ")[1].split("\n")[0]
        idx = int(test_id.split("_")[1])
        score = self._scores[idx - 1]
        return ({"test_id": test_id, "target_response": "echo", "score": score,
                 "passed": score <= 2, "reasoning": "because", "evidence_span": "ev"},
                TokenUsage(prompt_tokens=50, completion_tokens=10))

    async def aclose(self):
        self.closed = True


class TargetMock(LLMClient):
    def __init__(self):
        self.model_id = "weak-demo-model"
        self.closed = False

    async def complete(self, **kwargs):
        return ChatResult(
            text="The answer is definitely March 11, 2019.",
            usage=TokenUsage(prompt_tokens=15, completion_tokens=12),
        )

    async def aclose(self):
        self.closed = True


def test_full_loop_asr_and_shape(eval_config, settings):
    # scores: 5,1,4,2,3 -> 2 of 5 >= 4 -> ASR 0.4
    engine = EngineMock([5, 1, 4, 2, 3])
    target = TargetMock()
    result = _run(
        run_eval(eval_config, settings=settings, engine_client=engine, target_client=target)
    )

    assert [v.score for v in result.verdicts] == [5, 1, 4, 2, 3]
    assert result.attack_success_rate == 0.4
    assert len(result.attacks) == 5
    assert len(result.cases) == 5  # drill-down join complete
    # every verdict joins to an attack
    assert {c.verdict.test_id for c in result.cases} == {v.test_id for v in result.verdicts}


def test_cost_prices_engine_tokens_with_source(eval_config, settings):
    engine = EngineMock([5, 5, 5, 5, 5])
    result = _run(
        run_eval(
            eval_config, settings=settings, engine_client=engine, target_client=TargetMock()
        )
    )
    cost = result.cost
    assert cost.price_source == "example pricing"
    # engine tokens = attacker(200+300) + judge(live 5x60 + consistency). usd > 0.
    assert cost.usd > 0
    assert cost.attacker.total_tokens == 500
    assert cost.target.total_tokens == 5 * 27  # 15+12 per case


def test_metrics_carry_backend_and_models(eval_config, settings):
    engine = EngineMock([3, 3, 3, 3, 3])
    result = _run(
        run_eval(
            eval_config, settings=settings, engine_client=engine, target_client=TargetMock()
        )
    )
    m = result.metrics
    assert m.inference_backend == "fireworks"
    assert m.model_id == "gemma-31b-it"
    assert m.target_model_id == "weak-demo-model"
    assert m.n_cases == 5
    assert m.wall_clock_seconds >= 0.0


def test_consistency_rescore_off_live_path(eval_config, settings):
    engine = EngineMock([5, 4, 3, 2, 1])
    result = _run(
        run_eval(
            eval_config, settings=settings, engine_client=engine, target_client=TargetMock()
        )
    )
    # 3 showcase cases, each re-judged (repeats-1)=2 extra times on top of live verdict
    assert len(result.consistency) == 3
    # showcase = top scoring cases (5,4,3)
    means = sorted(c.mean for c in result.consistency)
    assert means == [3.0, 4.0, 5.0]
    # live judge calls = 5, plus 3 cases * 2 re-judges = 6 -> 11 total
    assert engine.judge_calls == 11


def test_consistency_can_be_disabled(eval_config, settings):
    engine = EngineMock([5, 4, 3, 2, 1])
    result = _run(
        run_eval(eval_config, settings=settings, engine_client=engine,
                 target_client=TargetMock(), include_consistency=False)
    )
    assert result.consistency == []
    assert engine.judge_calls == 5  # only the live path


def test_injected_clients_not_closed(eval_config, settings):
    # When the caller injects clients, run_eval must not close them (caller owns them).
    engine = EngineMock([1, 1, 1, 1, 1])
    target = TargetMock()
    _run(run_eval(eval_config, settings=settings, engine_client=engine, target_client=target))
    assert engine.closed is False
    assert target.closed is False


def test_judge_failure_degrades_to_fallback(eval_config, settings):
    class FlakyEngine(EngineMock):
        async def complete_json(self, **kwargs):
            if kwargs["schema_name"] != "adversarial_test_cases":
                from gemmajudge.client import LLMError
                raise LLMError("judge down")
            return await super().complete_json(**kwargs)

    engine = FlakyEngine([5, 5, 5, 5, 5])
    result = _run(
        run_eval(eval_config, settings=settings, engine_client=engine,
                 target_client=TargetMock(), include_consistency=False)
    )
    # all judged via fallback -> score 1, passed True, ASR 0 (conservative)
    assert all(v.score == 1 and v.passed for v in result.verdicts)
    assert result.attack_success_rate == 0.0
    assert len(result.verdicts) == 5  # run still completed
