"""Environment-driven configuration for the GemmaJudge engine.

Everything the engine needs to reach a model comes from here, and *only* from
here — no endpoint, key, or model id is hardcoded anywhere else (AGENTS.md §5.3,
PRD F9). Call :func:`load_settings` once at startup; it:

* loads ``.env`` in local dev (via ``python-dotenv``; a real environment's vars
  always win over the file),
* resolves ``INFERENCE_BACKEND`` into a concrete ``(base_url, api_key, model_id)``
  for the Attacker+Judge Gemma,
* resolves the separate system-under-test (target) endpoint,
* **fails loudly** with one aggregated, secret-free message listing every missing
  required variable — so a misconfigured run dies at startup, not mid-demo.

Secrets are held as :class:`pydantic.SecretStr`, so a stray ``print(settings)``
or a logged traceback shows ``SecretStr('**********')``, never the key itself.
"""

from __future__ import annotations

import os
from enum import StrEnum

from pydantic import BaseModel, SecretStr

try:  # dotenv is a convenience for local dev; absence must not break prod.
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dotenv is a listed dep, this is defensive

    def load_dotenv(*_args: object, **_kwargs: object) -> bool:
        return False


# Fireworks' documented OpenAI-compatible base URL — used if the env var is unset.
_DEFAULT_FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
# Placeholder key for endpoints that don't authenticate (e.g. a local vLLM server).
# The OpenAI SDK requires *some* non-empty api_key even when the server ignores it.
_NO_AUTH_PLACEHOLDER = "EMPTY"


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid.

    The message never contains secret values — only variable *names*."""


class InferenceBackend(StrEnum):
    """Where the Attacker + Judge Gemma runs (env ``INFERENCE_BACKEND``)."""

    FIREWORKS = "fireworks"  # Gemma on Fireworks' AMD-hosted infra → live URL
    MI300X = "mi300x"  # Gemma self-hosted on AMD Dev Cloud MI300X → AMD proof


class EndpointSettings(BaseModel):
    """A resolved OpenAI-compatible endpoint: where to call and as what model."""

    base_url: str
    api_key: SecretStr
    model_id: str


class PricingSettings(BaseModel):
    """Prices for the cost meter (PRD F8). USD per 1,000 tokens.

    Defaults are ``0.0`` so the meter shows a truthful ``$0.00`` until real prices
    are supplied via env — we never invent a number. ``source`` is surfaced in the
    UI next to any figure so a comparison always carries its citation."""

    prompt_usd_per_1k: float = 0.0
    completion_usd_per_1k: float = 0.0
    source: str | None = None

    def cost_usd(self, prompt_tokens: int, completion_tokens: int) -> float:
        """USD for a given token count, from measured usage only."""
        return (
            prompt_tokens / 1000.0 * self.prompt_usd_per_1k
            + completion_tokens / 1000.0 * self.completion_usd_per_1k
        )


class Settings(BaseModel):
    """Fully-resolved, validated engine configuration."""

    backend: InferenceBackend
    engine: EndpointSettings  # Attacker + Judge Gemma
    target: EndpointSettings  # system-under-test
    pricing: PricingSettings = PricingSettings()

    # Runtime knobs (safe defaults; overridable via env).
    max_concurrency: int = 8  # asyncio fan-out cap (30s-rule guardrail)
    request_timeout_s: float = 25.0  # per-call ceiling, under the 30s hard rule

    model_config = {"frozen": True}

    @property
    def model_id(self) -> str:
        """The Attacker + Judge Gemma model id (for the on-screen AMD label)."""
        return self.engine.model_id


def _get(env: dict[str, str], key: str) -> str | None:
    """Fetch an env var, treating empty/whitespace-only as absent."""
    val = env.get(key)
    if val is None:
        return None
    val = val.strip()
    return val or None


def _get_float(env: dict[str, str], key: str, default: float) -> float:
    raw = _get(env, key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{key} must be a number, got {raw!r}") from exc


def _get_int(env: dict[str, str], key: str, default: int) -> int:
    raw = _get(env, key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{key} must be an integer, got {raw!r}") from exc


def load_settings(env: dict[str, str] | None = None) -> Settings:
    """Load and validate configuration from the environment.

    Args:
        env: Mapping to read from. Defaults to ``os.environ`` (with ``.env``
            loaded first). Passing an explicit dict makes this fully testable
            without touching the process environment.

    Raises:
        ConfigError: if the selected backend or the target is under-configured.
            The message lists every offending variable at once (no secrets).
    """
    if env is None:
        load_dotenv()  # no-op if there's no .env; real env vars take precedence
        env = dict(os.environ)

    missing: list[str] = []

    # --- backend selection ---------------------------------------------------
    backend_raw = _get(env, "INFERENCE_BACKEND") or InferenceBackend.FIREWORKS.value
    try:
        backend = InferenceBackend(backend_raw.lower())
    except ValueError as exc:
        allowed = ", ".join(b.value for b in InferenceBackend)
        raise ConfigError(
            f"INFERENCE_BACKEND={backend_raw!r} is not valid. Choose one of: {allowed}."
        ) from exc

    # --- Attacker + Judge Gemma (the engine) ---------------------------------
    model_id = _get(env, "MODEL_ID")
    if model_id is None:
        missing.append("MODEL_ID")

    if backend is InferenceBackend.FIREWORKS:
        base_url = _get(env, "FIREWORKS_BASE_URL") or _DEFAULT_FIREWORKS_BASE_URL
        api_key = _get(env, "FIREWORKS_API_KEY")
        if api_key is None:
            missing.append("FIREWORKS_API_KEY")
    else:  # MI300X — self-hosted vLLM, key optional
        base_url = _get(env, "MI300X_BASE_URL")
        if base_url is None:
            missing.append("MI300X_BASE_URL")
        api_key = _get(env, "MI300X_API_KEY") or _NO_AUTH_PLACEHOLDER

    # --- target (system-under-test) ------------------------------------------
    target_endpoint = _get(env, "TARGET_ENDPOINT")
    if target_endpoint is None:
        missing.append("TARGET_ENDPOINT")
    target_model_id = _get(env, "TARGET_MODEL_ID")
    if target_model_id is None:
        missing.append("TARGET_MODEL_ID")
    target_api_key = _get(env, "TARGET_API_KEY") or _NO_AUTH_PLACEHOLDER

    if missing:
        raise ConfigError(
            "Missing required configuration: "
            + ", ".join(missing)
            + f". (backend={backend.value}) - set these in your environment or .env; "
            "see .env.example. No secrets are printed."
        )

    # --- pricing + runtime knobs ---------------------------------------------
    pricing = PricingSettings(
        prompt_usd_per_1k=_get_float(env, "PRICE_PER_1K_PROMPT_TOKENS", 0.0),
        completion_usd_per_1k=_get_float(env, "PRICE_PER_1K_COMPLETION_TOKENS", 0.0),
        source=_get(env, "PRICE_SOURCE"),
    )
    max_concurrency = _get_int(env, "MAX_CONCURRENCY", 8)
    if max_concurrency < 1:
        raise ConfigError(f"MAX_CONCURRENCY must be >= 1, got {max_concurrency}")
    request_timeout_s = _get_float(env, "REQUEST_TIMEOUT_S", 25.0)
    if request_timeout_s <= 0:
        raise ConfigError(f"REQUEST_TIMEOUT_S must be > 0, got {request_timeout_s}")

    # These locals are only reached when `missing` is empty, so they are all set.
    return Settings(
        backend=backend,
        engine=EndpointSettings(
            base_url=base_url,  # type: ignore[arg-type]
            api_key=SecretStr(api_key),  # type: ignore[arg-type]
            model_id=model_id,  # type: ignore[arg-type]
        ),
        target=EndpointSettings(
            base_url=target_endpoint,  # type: ignore[arg-type]
            api_key=SecretStr(target_api_key),
            model_id=target_model_id,  # type: ignore[arg-type]
        ),
        pricing=pricing,
        max_concurrency=max_concurrency,
        request_timeout_s=request_timeout_s,
    )
