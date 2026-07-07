"""Shared test fakes — no network, no real models.

The engine is designed so every model call goes through ``LLMClient``. These fakes
implement just enough of that surface to exercise the whole loop deterministically.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import SecretStr

from gemmajudge.client import LLMClient
from gemmajudge.config import (
    EndpointSettings,
    InferenceBackend,
    PricingSettings,
    Settings,
)
from gemmajudge.schemas import EvalConfig, TokenUsage


class FakeBackend:
    """A stand-in for ``AsyncOpenAI`` exposing ``chat.completions.create``.

    Returns a scripted content string (or raises a scripted exception) per call.
    """

    def __init__(
        self,
        contents: Sequence[str | Exception],
        usage: tuple[int, int] = (10, 5),
    ) -> None:
        self._contents = list(contents)
        self._usage = usage
        self.calls: list[dict[str, Any]] = []
        self.completions = SimpleNamespace(create=self._create)
        self.chat = SimpleNamespace(completions=self.completions)

    async def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        item = self._contents.pop(0) if self._contents else ""
        if isinstance(item, Exception):
            raise item
        message = SimpleNamespace(content=item)
        choice = SimpleNamespace(message=message)
        usage = SimpleNamespace(
            prompt_tokens=self._usage[0], completion_tokens=self._usage[1]
        )
        return SimpleNamespace(choices=[choice], usage=usage)


class ScriptedClient(LLMClient):
    """An ``LLMClient`` whose ``complete_json`` is driven by a callable.

    Bypasses the JSON string layer so tests can hand back dicts directly and focus
    on module logic (attacker/judge/orchestrator) rather than JSON plumbing (that's
    covered separately against the real ``LLMClient`` + ``FakeBackend``).
    """

    def __init__(
        self,
        model_id: str,
        json_handler: Callable[[dict[str, Any]], tuple[dict[str, Any], TokenUsage]],
    ) -> None:
        self.model_id = model_id
        self._json_handler = json_handler
        self.closed = False

    async def complete_json(self, **kwargs: Any) -> tuple[dict[str, Any], TokenUsage]:
        return self._json_handler(kwargs)

    async def aclose(self) -> None:
        self.closed = True


@pytest.fixture
def eval_config() -> EvalConfig:
    return EvalConfig(
        target_endpoint="http://localhost:8000/v1",
        target_model_id="weak-demo-model",
        n_cases=5,
    )


@pytest.fixture
def settings() -> Settings:
    return Settings(
        backend=InferenceBackend.FIREWORKS,
        engine=EndpointSettings(
            base_url="http://engine/v1",
            api_key=SecretStr("engine-key"),
            model_id="gemma-31b-it",
        ),
        target=EndpointSettings(
            base_url="http://target/v1",
            api_key=SecretStr("target-key"),
            model_id="weak-demo-model",
        ),
        pricing=PricingSettings(
            prompt_usd_per_1k=0.2,
            completion_usd_per_1k=0.6,
            source="example pricing",
        ),
        max_concurrency=4,
    )
