"""Tests for the Attacker: JSON validation, id re-stamping, dedupe, retry."""

import asyncio

import pytest

from gemmajudge.attacker import generate_attacks
from gemmajudge.client import LLMError
from gemmajudge.schemas import TokenUsage
from tests.conftest import ScriptedClient


def _run(coro):
    return asyncio.run(coro)


def _good_payload():
    return {
        "failure_mode": "hallucination",
        "test_cases": [
            {"id": "X", "prompt": "Fake Q1?", "rationale": "r1", "targeted_weakness": "w1"},
            {"id": "Y", "prompt": "fake q1?", "rationale": "r2", "targeted_weakness": "w2"},
            {"id": "Z", "prompt": "Fake Q2?", "rationale": "r3", "targeted_weakness": "w3"},
        ],
    }


def test_ids_are_restamped_in_order(eval_config):
    client = ScriptedClient(
        "gemma", lambda kw: (_good_payload(), TokenUsage(prompt_tokens=100, completion_tokens=50))
    )
    cases, usage = _run(generate_attacks(client, eval_config))
    # dup ("fake q1?") dropped -> 2 unique, ids normalized
    assert [c.id for c in cases] == ["tc_001", "tc_002"]
    assert usage.total_tokens == 150


def test_duplicate_prompts_dropped(eval_config):
    client = ScriptedClient(
        "gemma", lambda kw: (_good_payload(), TokenUsage())
    )
    cases, _ = _run(generate_attacks(client, eval_config))
    prompts = [c.prompt for c in cases]
    assert len(prompts) == len(set(p.lower() for p in prompts))


def test_missing_fields_get_defaults(eval_config):
    payload = {
        "test_cases": [
            {"prompt": "Only a prompt?"},  # no rationale / weakness / id
        ]
    }
    client = ScriptedClient("gemma", lambda kw: (payload, TokenUsage()))
    cases, _ = _run(generate_attacks(client, eval_config))
    assert cases[0].id == "tc_001"
    assert cases[0].rationale  # non-empty default
    assert cases[0].targeted_weakness


def test_over_delivery_trimmed_to_n(eval_config):
    many = {
        "test_cases": [
            {"prompt": f"Unique question {i}?", "rationale": "r", "targeted_weakness": "w"}
            for i in range(20)
        ]
    }
    client = ScriptedClient("gemma", lambda kw: (many, TokenUsage()))
    cases, _ = _run(generate_attacks(client, eval_config))
    assert len(cases) == eval_config.n_cases  # 5


def test_retry_on_empty_then_success(eval_config):
    calls = {"n": 0}

    def handler(kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"test_cases": []}, TokenUsage(prompt_tokens=10, completion_tokens=1)
        return _good_payload(), TokenUsage(prompt_tokens=10, completion_tokens=1)

    client = ScriptedClient("gemma", handler)
    cases, usage = _run(generate_attacks(client, eval_config))
    assert calls["n"] == 2
    assert len(cases) == 2
    assert usage.total_tokens == 22  # usage accumulated across both attempts


def test_exhaustion_raises(eval_config):
    client = ScriptedClient("gemma", lambda kw: ({"test_cases": []}, TokenUsage()))
    with pytest.raises(LLMError):
        _run(generate_attacks(client, eval_config, max_attempts=2))


def test_llm_error_during_generation_is_retried(eval_config):
    calls = {"n": 0}

    def handler(kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise LLMError("boom")
        return _good_payload(), TokenUsage()

    client = ScriptedClient("gemma", handler)
    cases, _ = _run(generate_attacks(client, eval_config))
    assert calls["n"] == 2
    assert cases
