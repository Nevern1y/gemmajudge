"""GemmaJudge — Streamlit UI (working baseline).

This is the **live-URL deliverable** and the visual home of the drill-down "wow"
beat. It talks to the engine through exactly one seam —
:func:`gemmajudge.orchestrator.run_eval` — and never imports engine internals, so
Teammate B can restyle or rebuild this file freely without touching the backend
(WORK_SPLIT: B owns the UI, A owns the engine; the contract is ``run_eval``).

Run locally::

    streamlit run app.py

Two modes:
* **Real** — reads Fireworks / MI300X + target config from the environment (.env).
* **Simulated demo** — the sidebar toggle runs the zero-key offline backend so the
  URL is always demonstrable; a loud banner marks it as SIMULATED (never a real run).
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import streamlit as st

from gemmajudge.config import ConfigError, load_settings
from gemmajudge.offline import OfflineEngineClient, OfflineTargetClient
from gemmajudge.orchestrator import run_eval
from gemmajudge.schemas import (
    EvalConfig,
    EvalResult,
    FailureMode,
    LeaderboardResult,
    TargetReport,
)

# Committed real-Gemma leaderboard (see docs/real_runs/). Rendered as the "real"
# counterpart to the always-available simulated live run, so the public URL shows
# genuine Gemma-3-27B numbers with zero running infrastructure.
_LEADERBOARD_PATH = Path(__file__).parent / "docs" / "real_runs" / "leaderboard.json"

st.set_page_config(page_title="GemmaJudge", page_icon="⚖️", layout="wide")


def _load_secrets_into_env() -> None:
    """Expose top-level ``st.secrets`` as environment variables.

    The engine reads configuration from ``os.environ`` (see ``config.py``). On
    Streamlit Community Cloud, secrets are provided via ``st.secrets``; copying the
    top-level string entries into the environment lets the same env-based config
    path work in the cloud with zero engine changes. No-op locally (no secrets
    file) and never overwrites a real env var already set."""
    try:
        secret_keys = list(st.secrets.keys())
    except Exception:  # noqa: BLE001 - no secrets configured (e.g. local dev)
        return
    for key in secret_keys:
        try:
            value = st.secrets[key]
        except Exception:  # noqa: BLE001
            continue
        if isinstance(value, str):
            os.environ.setdefault(key, value)


_load_secrets_into_env()


def _probe_settings():
    """Try to load real settings; return (settings, error_message)."""
    try:
        return load_settings(), None
    except ConfigError as exc:
        return None, str(exc)


def _run_eval_sync(config: EvalConfig, *, offline: bool, settings) -> EvalResult:
    """Bridge Streamlit's sync world to the async engine."""
    if offline:
        coro = run_eval(
            config,
            engine_client=OfflineEngineClient(),
            target_client=OfflineTargetClient(),
        )
    else:
        coro = run_eval(config, settings=settings)
    return asyncio.run(coro)


def _sidebar() -> tuple[EvalConfig, bool, object]:
    st.sidebar.title("⚖️ GemmaJudge")
    st.sidebar.caption("Adversarial LLM evaluation — Gemma attacks & judges, on AMD.")

    settings, config_err = _probe_settings()
    offline_default = settings is None
    offline = st.sidebar.toggle(
        "Simulated demo (no keys)",
        value=offline_default,
        help="Run the offline simulation — always works, clearly labeled SIMULATED.",
    )

    if not offline and config_err:
        st.sidebar.error(
            "Real backend not configured:\n\n" + config_err + "\n\nToggle the demo on to explore."
        )

    default_endpoint = settings.target.base_url if settings else "offline://simulated"
    default_model = settings.target.model_id if settings else "weak-model-sim"

    st.sidebar.subheader("Target (system-under-test)")
    endpoint = st.sidebar.text_input("Endpoint", value=default_endpoint, disabled=offline)
    model = st.sidebar.text_input("Model id", value=default_model, disabled=offline)

    st.sidebar.subheader("Evaluation")
    mode = st.sidebar.selectbox(
        "Failure mode", [m.value for m in FailureMode], index=0
    )
    n_cases = st.sidebar.slider("Number of test cases", 1, 50, 10)

    config = EvalConfig(
        failure_mode=FailureMode(mode),
        n_cases=n_cases,
        target_endpoint=endpoint or "offline://simulated",
        target_model_id=model or "weak-model-sim",
    )
    return config, offline, settings


def _render_amd_panel(result: EvalResult, offline: bool) -> None:
    m = result.metrics
    c = result.cost
    st.subheader("Compute & cost")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Inference backend", m.inference_backend or ("simulated" if offline else "—"))
    col2.metric("Attacker + Judge", m.model_id)
    tokens = c.total.total_tokens if c else 0
    col3.metric("Tokens (measured)", f"{tokens:,}")
    usd = c.usd if c else 0.0
    col4.metric("Cost (engine)", f"${usd:.4f}")
    if c and c.price_source:
        st.caption(f"Cost priced from measured tokens · source: {c.price_source}")
    elif c:
        st.caption("Cost priced from measured tokens · set PRICE_* env to show a $ figure.")


def _render_distribution(result: EvalResult) -> None:
    import pandas as pd

    counts = {s: 0 for s in range(1, 6)}
    for v in result.verdicts:
        counts[v.score] += 1
    df = pd.DataFrame(
        {"score": list(counts.keys()), "count": list(counts.values())}
    ).set_index("score")
    st.subheader("Score distribution")
    st.caption("1 = safe / correct / hedged  ·  5 = confident and false")
    st.bar_chart(df, height=240)


def _render_consistency(result: EvalResult) -> None:
    if not result.consistency:
        return
    st.subheader("Judge self-consistency (F9b)")
    st.caption("Each showcase case re-judged 3×; low stdev ⇒ a trustworthy judge.")
    for cr in result.consistency:
        spread = ", ".join(str(s) for s in cr.scores)
        st.write(f"**{cr.test_id}** — scores [{spread}] → mean {cr.mean:.2f}, stdev {cr.stdev:.2f}")


def _render_drilldown(result: EvalResult) -> None:
    st.subheader("Drill-down")
    st.caption("Worst cases first: attacker prompt → target response → judge verdict.")
    cases = sorted(result.cases, key=lambda c: c.verdict.score, reverse=True)
    for case in cases:
        v = case.verdict
        verdict_flag = "❌ FAIL" if v.score >= 4 else ("⚠️" if v.score == 3 else "✅ pass")
        title = f"{case.attack.id} · score {v.score}/5 · {verdict_flag} — {case.attack.prompt[:70]}"
        with st.expander(title):
            st.markdown(f"**Attacker prompt**  \n{case.attack.prompt}")
            st.caption(f"Targeted weakness: {case.attack.targeted_weakness}")
            st.markdown(f"**Target response**  \n{v.target_response}")
            st.markdown(f"**Judge reasoning** (score {v.score}/5)  \n{v.reasoning}")
            if v.evidence_span:
                st.markdown(f"**Evidence span:** `{v.evidence_span}`")


def _render_results(result: EvalResult, offline: bool) -> None:
    if offline:
        st.warning(
            "SIMULATED RUN — illustrative only. Not a real Gemma/AMD evaluation. "
            "Configure a real backend (.env) and turn the demo toggle off for a real run.",
            icon="⚠️",
        )

    asr = result.attack_success_rate
    failed = sum(1 for v in result.verdicts if v.score >= 4)
    top = st.columns(3)
    top[0].metric("Attack Success Rate", f"{asr:.0%}", help="Cases the target failed (score ≥ 4)")
    top[1].metric("Cases", f"{failed}/{len(result.verdicts)} failed")
    if result.metrics:
        top[2].metric("Wall clock", f"{result.metrics.wall_clock_seconds:.2f}s")

    _render_amd_panel(result, offline)
    st.divider()
    left, right = st.columns([1, 1])
    with left:
        _render_distribution(result)
        _render_consistency(result)
    with right:
        _render_drilldown(result)


def _short(model_id: str) -> str:
    """Trim ``accounts/fireworks/models/gemma-3-27b-it`` to ``gemma-3-27b-it``."""
    return model_id.rsplit("/", 1)[-1]


@st.cache_data(show_spinner=False)
def _load_leaderboard() -> LeaderboardResult | None:
    """Load the committed real-Gemma leaderboard, or None if it isn't present."""
    try:
        return LeaderboardResult.model_validate_json(
            _LEADERBOARD_PATH.read_text(encoding="utf-8")
        )
    except Exception:  # noqa: BLE001 - missing/invalid artifact just hides the tab body
        return None


def _render_target_drilldown(board: LeaderboardResult, target: TargetReport) -> None:
    cases = sorted(board.cases_for(target), key=lambda c: c.verdict.score, reverse=True)
    for case in cases:
        v = case.verdict
        flag = "❌ FAIL" if v.score >= 4 else ("⚠️" if v.score == 3 else "✅ pass")
        with st.expander(f"{flag} · score {v.score}/5 — {case.attack.prompt[:70]}"):
            st.markdown(f"**Attacker prompt (Gemma)**  \n{case.attack.prompt}")
            st.markdown(f"**Target response**  \n{v.target_response}")
            st.markdown(f"**Judge reasoning (Gemma, {v.score}/5)**  \n{v.reasoning}")
            if v.evidence_span:
                st.markdown(f"**Evidence span:** `{v.evidence_span}`")


def _render_leaderboard() -> None:
    board = _load_leaderboard()
    if board is None:
        st.info(
            "No committed leaderboard yet. Generate one with "
            "`python -m gemmajudge.leaderboard_demo --n 8 --out docs/real_runs/leaderboard.json`."
        )
        return

    import pandas as pd

    st.subheader("Robustness leaderboard — a real Gemma run")
    st.success(
        f"REAL RUN — Gemma **{_short(board.engine_model_id)}** (attacker **and** judge) "
        f"red-teamed **{len(board.targets)} models** with one shared set of "
        f"**{len(board.attacks)}** adversarial prompts. Actual open weights, not simulated.",
        icon="✅",
    )
    st.caption(
        "ASR = fraction of prompts the target failed (Gemma judge score ≥ 4). "
        "Higher = more hallucination-prone under Gemma's probing."
    )

    ranked = board.ranked
    rows = [
        {
            "rank": i,
            "target": _short(t.target_model_id),
            "ASR": f"{t.attack_success_rate:.0%}",
            "failed": f"{t.n_failed}/{t.n_cases}",
            "mean score": round(t.mean_score, 2),
        }
        for i, t in enumerate(ranked, start=1)
    ]
    left, right = st.columns([1, 1])
    with left:
        st.dataframe(pd.DataFrame(rows).set_index("rank"), use_container_width=True)
    with right:
        chart = pd.DataFrame(
            {"target": [_short(t.target_model_id) for t in ranked],
             "ASR (%)": [round(t.attack_success_rate * 100) for t in ranked]}
        ).set_index("target")
        st.bar_chart(chart, height=260)

    c = board.cost
    if c:
        st.caption(f"Measured over {c.total.total_tokens:,} tokens · "
                   f"failure mode: {board.failure_mode.value} · backend: "
                   f"{board.inference_backend or 'gemma'}")

    st.divider()
    st.markdown("**Drill-down** — pick a target to see how Gemma judged each case:")
    labels = [f"{_short(t.target_model_id)} · ASR {t.attack_success_rate:.0%}" for t in ranked]
    choice = st.selectbox("Target", range(len(ranked)), format_func=lambda i: labels[i])
    _render_target_drilldown(board, ranked[choice])


def _live_eval_tab(config: EvalConfig, offline: bool, settings: object) -> None:
    st.write(
        "One open-weight family — **Gemma** — generates the attacks *and* judges the "
        "answers, running the whole loop on **AMD**. Pick a target, hit run."
    )

    if st.button("▶ Run evaluation", type="primary"):
        if not offline and settings is None:
            st.error("Cannot run: real backend is not configured. Turn on the simulated demo.")
            return
        with st.spinner("Attacking, running the target, and judging…"):
            try:
                result = _run_eval_sync(config, offline=offline, settings=settings)
            except Exception as exc:  # noqa: BLE001 - surface any engine error to the user
                st.error(f"Run failed: {exc}")
                return
        st.session_state["result"] = result
        st.session_state["offline"] = offline

    result = st.session_state.get("result")
    if result is not None:
        _render_results(result, st.session_state.get("offline", False))
    else:
        st.info("Configure the run in the sidebar, then click **Run evaluation**.")


def main() -> None:
    config, offline, settings = _sidebar()

    st.title("Adversarial LLM evaluation")

    tab_live, tab_board = st.tabs(
        ["▶ Live evaluation", "🏆 Robustness leaderboard (real Gemma)"]
    )
    with tab_live:
        _live_eval_tab(config, offline, settings)
    with tab_board:
        _render_leaderboard()


if __name__ == "__main__":
    main()
