"""Tests for the target module: resilience and sentinel behavior."""

import asyncio

from gemmajudge.client import ChatResult, LLMClient, LLMError
from gemmajudge.schemas import TokenUsage
from gemmajudge.target import TARGET_ERROR_SENTINEL, query_target


def _run(coro):
    return asyncio.run(coro)


class _Client(LLMClient):
    def __init__(self, behavior):
        self.model_id = "weak"
        self._behavior = behavior

    async def complete(self, **kwargs):
        return self._behavior(kwargs)


def test_returns_text_and_usage():
    client = _Client(
        lambda kw: ChatResult(
            text="  a confident wrong answer  ",
            usage=TokenUsage(prompt_tokens=8, completion_tokens=4),
        )
    )
    text, usage = _run(query_target(client, "Q?"))
    assert text == "a confident wrong answer"  # stripped
    assert usage.total_tokens == 12


def test_error_returns_sentinel_not_raise():
    def boom(kw):
        raise LLMError("connection refused")

    client = _Client(boom)
    text, usage = _run(query_target(client, "Q?"))
    assert text == TARGET_ERROR_SENTINEL
    assert usage.total_tokens == 0


def test_empty_response_returns_sentinel():
    client = _Client(
        lambda kw: ChatResult(text="   ", usage=TokenUsage(prompt_tokens=3, completion_tokens=0))
    )
    text, _ = _run(query_target(client, "Q?"))
    assert text == TARGET_ERROR_SENTINEL
