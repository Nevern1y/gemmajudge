"""Regression tests for the 7 findings confirmed by the adversarial review.

Each test pins a specific fix so the bug cannot silently return.
"""

import asyncio

import pytest
from pydantic import SecretStr

from gemmajudge.client import ChatResult, LLMClient, LLMError, _extract_json
from gemmajudge.config import ConfigError, EndpointSettings, load_settings
from gemmajudge.judge import judge
from gemmajudge.orchestrator import run_eval
from gemmajudge.schemas import AttackCase, TokenUsage


def _run(coro):
    return asyncio.run(coro)


class _Engine(LLMClient):
    """Engine mock: serves attacker + a passing judge; tracks close()."""

    def __init__(self):
        self.model_id = "gemma"
        self.closed = False

    async def complete_json(self, **kwargs):
        if kwargs["schema_name"] == "adversarial_test_cases":
            return (
                {"test_cases": [
                    {"id": "a", "prompt": "P1?", "rationale": "r", "targeted_weakness": "w"}
                ]},
                TokenUsage(prompt_tokens=10, completion_tokens=5),
            )
        test_id = kwargs["user"].split("test_id: ")[1].split("\n")[0]
        return (
            {"test_id": test_id, "target_response": "x", "score": 5,
             "passed": False, "reasoning": "r", "evidence_span": "e"},
            TokenUsage(prompt_tokens=3, completion_tokens=2),
        )

    async def aclose(self):
        self.closed = True


class _Target(LLMClient):
    def __init__(self):
        self.model_id = "t"
        self.closed = False

    async def complete(self, **kwargs):
        return ChatResult(
            text="a confident answer",
            usage=TokenUsage(prompt_tokens=2, completion_tokens=2),
        )

    async def aclose(self):
        self.closed = True


# --- Findings 1 + 2: per-client ownership (never close an injected client) ---


def test_only_injected_engine_is_not_closed(monkeypatch, eval_config, settings):
    built_target = _Target()
    monkeypatch.setattr("gemmajudge.orchestrator.make_target_client", lambda s: built_target)
    injected_engine = _Engine()
    _run(run_eval(eval_config, settings=settings, engine_client=injected_engine,
                  include_consistency=False))
    assert injected_engine.closed is False  # injected -> caller's to manage
    assert built_target.closed is True       # created by run_eval -> closed


def test_only_injected_target_is_not_closed(monkeypatch, eval_config, settings):
    built_engine = _Engine()
    monkeypatch.setattr("gemmajudge.orchestrator.make_engine_client", lambda s: built_engine)
    injected_target = _Target()
    _run(run_eval(eval_config, settings=settings, target_client=injected_target,
                  include_consistency=False))
    assert injected_target.closed is False
    assert built_engine.closed is True


# --- Finding 3: REQUEST_TIMEOUT_S upper bound (30s rule) ---


def _fireworks_env(**extra):
    base = {
        "INFERENCE_BACKEND": "fireworks",
        "MODEL_ID": "gemma",
        "FIREWORKS_API_KEY": "k",
        "TARGET_ENDPOINT": "http://t/v1",
        "TARGET_MODEL_ID": "m",
    }
    base.update(extra)
    return base


def test_timeout_over_30s_is_rejected():
    with pytest.raises(ConfigError) as exc:
        load_settings(_fireworks_env(REQUEST_TIMEOUT_S="45"))
    assert "30" in str(exc.value)


def test_timeout_at_boundary_accepted():
    s = load_settings(_fireworks_env(REQUEST_TIMEOUT_S="30"))
    assert s.request_timeout_s == 30.0


# --- Finding 4: client built with max_retries=0 ---


def test_client_disables_sdk_retries():
    client = LLMClient.from_endpoint(
        EndpointSettings(base_url="http://x/v1", api_key=SecretStr("k"), model_id="m"),
        timeout=25.0,
    )
    assert client._backend.max_retries == 0
    assert float(client._backend.timeout) == 25.0


# --- Finding 5: judge carries spent tokens on total failure ---


class _BadJudgeEngine(_Engine):
    async def complete_json(self, **kwargs):
        if kwargs["schema_name"] == "adversarial_test_cases":
            return await super().complete_json(**kwargs)
        # always-invalid score -> retried, tokens accrue, then exhausts
        return (
            {"test_id": "a", "target_response": "x", "score": 99,
             "passed": False, "reasoning": "r", "evidence_span": "e"},
            TokenUsage(prompt_tokens=7, completion_tokens=3),
        )


def test_judge_exhaustion_error_carries_usage():
    case = AttackCase(id="tc_001", prompt="P?", rationale="r", targeted_weakness="w")
    engine = _BadJudgeEngine()
    with pytest.raises(LLMError) as exc:
        _run(judge(engine, case, "resp", max_attempts=3))
    assert exc.value.usage is not None
    assert exc.value.usage.total_tokens == 30  # 3 attempts x (7+3)


def test_orchestrator_counts_failed_judge_tokens(eval_config, settings):
    result = _run(run_eval(eval_config, settings=settings, engine_client=_BadJudgeEngine(),
                           target_client=_Target(), include_consistency=False))
    # judge failed -> fallback verdict, but the tokens it spent are still billed
    assert all(v.score == 1 and v.passed for v in result.verdicts)
    assert result.cost.judge.total_tokens > 0


# --- Finding 6: _extract_json tolerates prose around a fence ---


def test_extract_json_with_trailing_prose_after_fence():
    text = '```json\n{"score": 5}\n```\nHope that helps!'
    assert _extract_json(text) == {"score": 5}


def test_extract_json_embedded_in_prose():
    text = 'Sure, here is the verdict: {"score": 4, "passed": false} — done.'
    assert _extract_json(text) == {"score": 4, "passed": False}


def test_extract_json_same_line_fence():
    text = '```json {"ok": true} ```'
    assert _extract_json(text) == {"ok": True}


def test_extract_json_still_rejects_non_object():
    with pytest.raises(LLMError):
        _extract_json("no json here")


# --- Finding 7: demo --n out of range fails cleanly, not with a traceback ---


def test_demo_out_of_range_n_exits_two(capsys):
    from gemmajudge import demo

    rc = demo.main(["--offline", "--n", "500"])  # EvalConfig caps n_cases at 100
    assert rc == 2
    assert "Invalid arguments" in capsys.readouterr().err
