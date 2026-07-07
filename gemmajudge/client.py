"""One OpenAI-compatible async client for every model GemmaJudge talks to.

Fireworks, self-hosted vLLM on MI300X, and the target system-under-test are all
OpenAI-compatible, so a single thin wrapper over :class:`openai.AsyncOpenAI`
serves all three — the only thing that changes is the endpoint, which comes from
:mod:`gemmajudge.config`.

The wrapper adds the two things the raw SDK call doesn't give us directly:

* **token accounting** — every call returns a :class:`~gemmajudge.schemas.TokenUsage`
  parsed from ``response.usage`` (the cost meter, PRD F8, is built on this), and
* **structured output** — :meth:`LLMClient.complete_json` asks the server for a
  ``json_schema``-shaped response and parses it, so the Attacker and Judge get
  validated JSON instead of free text.

Testability: :class:`LLMClient` wraps *any* object exposing
``chat.completions.create`` (the real ``AsyncOpenAI`` in production, a mock in
tests), so the whole engine is unit-testable with zero network calls.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from openai import AsyncOpenAI

from gemmajudge.config import EndpointSettings, Settings
from gemmajudge.schemas import TokenUsage


class _CompletionsCreate(Protocol):
    async def create(self, **kwargs: Any) -> Any: ...


class _Chat(Protocol):
    @property
    def completions(self) -> _CompletionsCreate: ...


class ChatBackend(Protocol):
    """The minimal surface :class:`LLMClient` needs — satisfied by ``AsyncOpenAI``.

    Declaring it as a Protocol is what lets tests inject a mock with just a
    ``chat.completions.create`` coroutine and nothing else."""

    @property
    def chat(self) -> _Chat: ...


class LLMError(RuntimeError):
    """A model call failed or returned an unusable payload.

    ``usage`` optionally carries any tokens already spent before the failure (e.g.
    across retry attempts), so callers can still account for real cost on error."""

    usage: TokenUsage | None = None


@dataclass(slots=True)
class ChatResult:
    """The text of a completion plus the tokens it cost."""

    text: str
    usage: TokenUsage


def _parse_usage(raw: Any) -> TokenUsage:
    """Turn a possibly-absent ``response.usage`` into a TokenUsage (never raises)."""
    if raw is None:
        return TokenUsage()
    # SDK objects expose attributes; some servers hand back a plain dict.
    prompt = getattr(raw, "prompt_tokens", None)
    completion = getattr(raw, "completion_tokens", None)
    if prompt is None and isinstance(raw, dict):
        prompt = raw.get("prompt_tokens")
        completion = raw.get("completion_tokens")
    return TokenUsage(
        prompt_tokens=int(prompt or 0),
        completion_tokens=int(completion or 0),
    )


def json_schema_response_format(name: str, schema: dict[str, Any]) -> dict[str, Any]:
    """Build the ``response_format`` payload for strict json_schema output.

    Works on Fireworks and recent vLLM. Callers that hit a server without strict
    support can fall back to plain JSON mode via :func:`json_object_response_format`.
    """
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "schema": schema, "strict": True},
    }


def json_object_response_format() -> dict[str, Any]:
    """Build the ``response_format`` for plain JSON mode (broadest compatibility)."""
    return {"type": "json_object"}


class LLMClient:
    """An async, usage-accounting wrapper around one OpenAI-compatible endpoint."""

    def __init__(self, backend: ChatBackend, model_id: str) -> None:
        self._backend = backend
        self.model_id = model_id

    # --- construction --------------------------------------------------------

    @classmethod
    def from_endpoint(cls, endpoint: EndpointSettings, *, timeout: float) -> LLMClient:
        """Build a client backed by a real ``AsyncOpenAI`` for the given endpoint.

        ``max_retries=0`` is deliberate: the SDK's default (2) would let one logical
        request take up to ~3× ``timeout`` on a slow/flaky endpoint, blowing the
        30s/request hard rule. We retry at the application layer instead (attacker /
        judge re-issue whole calls), so every *individual* HTTP request stays bounded
        by ``timeout``."""
        backend = AsyncOpenAI(
            base_url=endpoint.base_url,
            api_key=endpoint.api_key.get_secret_value(),
            timeout=timeout,
            max_retries=0,
        )
        return cls(backend, endpoint.model_id)

    # --- calls ---------------------------------------------------------------

    async def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> ChatResult:
        """One chat completion. Returns text + measured token usage.

        Raises :class:`LLMError` on transport failure or an empty/malformed
        response, so callers can retry or degrade a single case without the whole
        run crashing.
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        kwargs: dict[str, Any] = {
            "model": model or self.model_id,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if response_format is not None:
            kwargs["response_format"] = response_format

        try:
            response = await self._backend.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - normalize any SDK/transport error
            raise LLMError(f"model call failed: {exc}") from exc

        try:
            text = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as exc:
            raise LLMError(f"malformed completion response: {exc}") from exc

        if text is None:
            raise LLMError("model returned empty content")

        return ChatResult(text=text, usage=_parse_usage(getattr(response, "usage", None)))

    async def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        schema_name: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> tuple[dict[str, Any], TokenUsage]:
        """Complete and parse a JSON object, requesting ``json_schema`` shaping.

        Returns the parsed dict and the usage. Raises :class:`LLMError` if the
        payload isn't valid JSON (the caller decides whether to retry).
        """
        result = await self.complete(
            system=system,
            user=user,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=json_schema_response_format(schema_name, schema),
            model=model,
        )
        payload = _extract_json(result.text)
        return payload, result.usage

    async def aclose(self) -> None:
        """Close the underlying client if it holds network resources."""
        close = getattr(self._backend, "close", None)
        if close is not None:
            await close()


def _extract_json(text: str) -> dict[str, Any]:
    """Parse a JSON object from a completion, tolerating fences and prose.

    Real servers wrap JSON in ```json ... ``` fences, prepend "Here you go:", or
    append a trailing remark. We try, in order: the whole text, the contents of a
    code fence, and the first balanced ``{...}`` object anywhere in the text —
    returning the first candidate that parses to a dict. Raises :class:`LLMError`
    if none do."""
    for candidate in _json_candidates(text):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    preview = text.strip()[:200]
    raise LLMError(f"expected a JSON object, got: {preview!r}")


def _json_candidates(text: str) -> list[str]:
    """Yield progressively more permissive JSON candidate substrings."""
    stripped = text.strip()
    candidates = [stripped]
    if stripped.startswith("```"):
        body = stripped[3:]
        if "\n" in body:  # drop an optional language tag (```json) on the first line
            body = body.split("\n", 1)[1]
        end = body.find("```")  # cut at the closing fence, ignoring any trailing prose
        if end != -1:
            body = body[:end]
        candidates.append(body.strip())
    obj = _first_json_object(stripped)
    if obj is not None:
        candidates.append(obj)
    return candidates


def _first_json_object(text: str) -> str | None:
    """Return the first balanced ``{...}`` substring, respecting string literals."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def make_engine_client(settings: Settings) -> LLMClient:
    """Client for the Attacker + Judge Gemma (the selected inference backend)."""
    return LLMClient.from_endpoint(settings.engine, timeout=settings.request_timeout_s)


def make_target_client(settings: Settings) -> LLMClient:
    """Client for the system-under-test (a separate endpoint)."""
    return LLMClient.from_endpoint(settings.target, timeout=settings.request_timeout_s)
