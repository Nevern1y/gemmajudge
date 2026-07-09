"""Tests for env loading + validation, including the secret-safety guarantee."""

import pytest

from gemmajudge.config import (
    ConfigError,
    InferenceBackend,
    load_settings,
)

_FIREWORKS_OK = {
    "INFERENCE_BACKEND": "fireworks",
    "MODEL_ID": "accounts/fireworks/models/gemma",
    "FIREWORKS_API_KEY": "sk-super-secret-value",
    "TARGET_ENDPOINT": "http://target/v1",
    "TARGET_MODEL_ID": "weak-model",
}


def test_fireworks_happy_path():
    s = load_settings(_FIREWORKS_OK)
    assert s.backend is InferenceBackend.FIREWORKS
    assert s.engine.model_id == "accounts/fireworks/models/gemma"
    assert s.engine.base_url.endswith("/inference/v1")  # default filled in
    assert s.target.model_id == "weak-model"


def test_missing_vars_are_aggregated_and_loud():
    with pytest.raises(ConfigError) as exc:
        load_settings({"INFERENCE_BACKEND": "fireworks"})
    msg = str(exc.value)
    # every missing var named in a single message
    for var in ("MODEL_ID", "FIREWORKS_API_KEY"):
        assert var in msg
    assert "TARGET_ENDPOINT" not in msg
    assert "TARGET_MODEL_ID" not in msg


def test_fireworks_defaults_target_to_same_backend():
    s = load_settings(
        {
            "INFERENCE_BACKEND": "fireworks",
            "MODEL_ID": "accounts/demo/deployments/gemma-live",
            "FIREWORKS_API_KEY": "sk-super-secret-value",
        }
    )
    assert s.target.base_url == s.engine.base_url
    assert s.target.model_id == "accounts/fireworks/models/gpt-oss-120b"
    assert s.target.api_key.get_secret_value() == "sk-super-secret-value"


def test_secret_never_appears_in_repr_or_str():
    s = load_settings(_FIREWORKS_OK)
    assert "sk-super-secret-value" not in repr(s)
    assert "sk-super-secret-value" not in str(s)
    assert str(s.engine.api_key) == "**********"
    # but it IS retrievable explicitly when needed
    assert s.engine.api_key.get_secret_value() == "sk-super-secret-value"


def test_error_message_contains_no_secret_values():
    env = dict(_FIREWORKS_OK)
    del env["MODEL_ID"]  # force an error while a secret is present
    with pytest.raises(ConfigError) as exc:
        load_settings(env)
    assert "sk-super-secret-value" not in str(exc.value)
    assert "MODEL_ID" in str(exc.value)


def test_mi300x_requires_base_url():
    with pytest.raises(ConfigError) as exc:
        load_settings(
            {
                "INFERENCE_BACKEND": "mi300x",
                "MODEL_ID": "gemma",
                "TARGET_ENDPOINT": "http://t/v1",
                "TARGET_MODEL_ID": "y",
            }
        )
    assert "MI300X_BASE_URL" in str(exc.value)


def test_mi300x_key_optional_uses_placeholder():
    s = load_settings(
        {
            "INFERENCE_BACKEND": "mi300x",
            "MODEL_ID": "gemma",
            "MI300X_BASE_URL": "http://mi300x/v1",
            "TARGET_ENDPOINT": "http://t/v1",
            "TARGET_MODEL_ID": "y",
        }
    )
    assert s.backend is InferenceBackend.MI300X
    # A placeholder key is present so the OpenAI SDK doesn't choke on empty auth.
    assert s.engine.api_key.get_secret_value()


def test_unknown_backend_rejected():
    with pytest.raises(ConfigError) as exc:
        load_settings(
            {
                "INFERENCE_BACKEND": "cuda",
                "MODEL_ID": "g",
                "TARGET_ENDPOINT": "x",
                "TARGET_MODEL_ID": "y",
            }
        )
    assert "cuda" in str(exc.value)


def test_pricing_and_runtime_knobs_parsed():
    env = dict(_FIREWORKS_OK)
    env.update(
        {
            "PRICE_PER_1K_PROMPT_TOKENS": "0.15",
            "PRICE_PER_1K_COMPLETION_TOKENS": "0.60",
            "PRICE_SOURCE": "Fireworks 2026-07",
            "MAX_CONCURRENCY": "12",
            "REQUEST_TIMEOUT_S": "20",
        }
    )
    s = load_settings(env)
    assert s.pricing.cost_usd(1000, 1000) == pytest.approx(0.75)
    assert s.pricing.source == "Fireworks 2026-07"
    assert s.max_concurrency == 12
    assert s.request_timeout_s == 20.0


def test_invalid_numeric_knob_is_loud():
    env = dict(_FIREWORKS_OK)
    env["MAX_CONCURRENCY"] = "0"
    with pytest.raises(ConfigError):
        load_settings(env)


def test_whitespace_only_var_treated_as_missing():
    env = dict(_FIREWORKS_OK)
    env["MODEL_ID"] = "   "
    with pytest.raises(ConfigError) as exc:
        load_settings(env)
    assert "MODEL_ID" in str(exc.value)
