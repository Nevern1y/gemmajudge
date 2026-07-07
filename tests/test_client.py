"""Tests for the OpenAI-compatible client wrapper (against a fake backend)."""

import pytest

from gemmajudge.client import (
    LLMClient,
    LLMError,
    json_schema_response_format,
)
from tests.conftest import FakeBackend


def _run(coro):
    import asyncio

    return asyncio.run(coro)


def test_complete_returns_text_and_usage():
    backend = FakeBackend(["hello world"], usage=(12, 3))
    client = LLMClient(backend, "gemma")
    result = _run(client.complete(system="s", user="u"))
    assert result.text == "hello world"
    assert result.usage.prompt_tokens == 12
    assert result.usage.completion_tokens == 3
    assert result.usage.total_tokens == 15


def test_complete_json_parses_plain_object():
    backend = FakeBackend(['{"a": 1, "b": [2, 3]}'])
    client = LLMClient(backend, "gemma")
    payload, _usage = _run(
        client.complete_json(system="s", user="u", schema={"type": "object"}, schema_name="x")
    )
    assert payload == {"a": 1, "b": [2, 3]}


def test_complete_json_strips_markdown_fence():
    backend = FakeBackend(['```json\n{"ok": true}\n```'])
    client = LLMClient(backend, "gemma")
    payload, _usage = _run(
        client.complete_json(system="s", user="u", schema={}, schema_name="x")
    )
    assert payload == {"ok": True}


def test_complete_json_sends_json_schema_response_format():
    backend = FakeBackend(['{"x": 1}'])
    client = LLMClient(backend, "gemma")
    _run(
        client.complete_json(
            system="s", user="u", schema={"type": "object"}, schema_name="verdict"
        )
    )
    sent = backend.calls[0]["response_format"]
    assert sent["type"] == "json_schema"
    assert sent["json_schema"]["name"] == "verdict"


def test_none_content_raises_llm_error():
    # Some servers return a completion with content=None; complete() must reject it
    # rather than pass None downstream.
    backend = FakeBackend([None])  # type: ignore[list-item]
    client = LLMClient(backend, "gemma")
    with pytest.raises(LLMError):
        _run(client.complete(system="s", user="u"))


def test_transport_error_becomes_llm_error():
    backend = FakeBackend([RuntimeError("connection refused")])
    client = LLMClient(backend, "gemma")
    with pytest.raises(LLMError) as exc:
        _run(client.complete(system="s", user="u"))
    assert "connection refused" in str(exc.value)


def test_bad_json_raises_llm_error():
    backend = FakeBackend(["not json at all"])
    client = LLMClient(backend, "gemma")
    with pytest.raises(LLMError):
        _run(client.complete_json(system="s", user="u", schema={}, schema_name="x"))


def test_json_array_rejected_as_not_object():
    backend = FakeBackend(["[1, 2, 3]"])
    client = LLMClient(backend, "gemma")
    with pytest.raises(LLMError):
        _run(client.complete_json(system="s", user="u", schema={}, schema_name="x"))


def test_json_schema_response_format_shape():
    rf = json_schema_response_format("myname", {"type": "object"})
    assert rf["json_schema"]["strict"] is True
    assert rf["json_schema"]["name"] == "myname"
