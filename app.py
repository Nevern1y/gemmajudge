"""GemmaJudge Mission Control Streamlit frontend.

This file is intentionally a frontend-only shell around the frozen backend seam:
``run_eval(config: EvalConfig) -> EvalResult``. It can run in two modes:

* Real backend: Gemma attacker + judge through Fireworks or MI300X env config.
* Simulated demo: zero-key offline backend, loudly labeled SIMULATED.
"""

from __future__ import annotations

import asyncio
import html
import os
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gemmajudge.config import ConfigError, load_settings
from gemmajudge.offline import OfflineEngineClient, OfflineTargetClient
from gemmajudge.orchestrator import run_eval
from gemmajudge.schemas import (
    CostReport,
    EvalConfig,
    EvalResult,
    FailureMode,
    LeaderboardResult,
    TargetReport,
)

ROOT = Path(__file__).parent
LEADERBOARD_PATH = ROOT / "docs" / "real_runs" / "leaderboard.json"
AMD_PROOF_DIR = ROOT / "docs" / "amd_proof"
AMD_PROOF_FILES = (
    "rocm_smi.txt",
    "versions.txt",
    "vllm_engine.log",
    "serve_command.txt",
    "eval_result.json",
    "mi300x_screenshot.png",
    "notes.md",
)

AMD_RED = "#ED1C24"
GEMMA_BLUE = "#4285F4"
SAFE_GREEN = "#2EE59D"
AMBER = "#F7B955"
INK = "#EAF0FF"
MUTED = "#9AA6C3"


st.set_page_config(
    page_title="GemmaJudge Mission Control",
    page_icon="GJ",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def _install_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --gj-bg: #070911;
            --gj-panel: rgba(14, 18, 32, 0.92);
            --gj-panel-2: rgba(20, 27, 47, 0.88);
            --gj-border: rgba(131, 148, 190, 0.22);
            --gj-ink: #eaf0ff;
            --gj-muted: #9aa6c3;
            --gj-red: #ed1c24;
            --gj-blue: #4285f4;
            --gj-green: #2ee59d;
            --gj-amber: #f7b955;
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(66, 133, 244, 0.18), transparent 30rem),
                radial-gradient(circle at top right, rgba(237, 28, 36, 0.16), transparent 28rem),
                linear-gradient(135deg, #070911 0%, #0b1020 52%, #090b12 100%);
            color: var(--gj-ink);
        }
        header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"],
        #MainMenu, footer { display: none !important; visibility: hidden !important; }
        .block-container { padding-top: 3.25rem; padding-bottom: 3rem; max-width: 1320px; }
        h1, h2, h3 { letter-spacing: -0.04em; color: var(--gj-ink) !important; }
        p, li, label, .stMarkdown { color: var(--gj-ink); }
        [data-testid="stMetric"] {
            background: linear-gradient(180deg, rgba(255,255,255,0.055), rgba(255,255,255,0.018));
            border: 1px solid var(--gj-border);
            border-radius: 18px;
            padding: 1rem;
        }
        [data-testid="stMetricLabel"] p { color: var(--gj-muted) !important; }
        [data-testid="stMetricValue"] { color: var(--gj-ink) !important; }
        .gj-hero {
            border: 1px solid rgba(131, 148, 190, 0.26);
            border-radius: 28px;
            padding: 1.5rem;
            background:
                linear-gradient(135deg, rgba(66,133,244,0.18), rgba(237,28,36,0.11)),
                rgba(10, 14, 27, 0.88);
            box-shadow: 0 24px 80px rgba(0, 0, 0, 0.28);
        }
        .gj-kicker {
            color: var(--gj-muted);
            font-size: 0.76rem;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            font-weight: 700;
        }
        .gj-title {
            font-size: clamp(2.5rem, 7vw, 5.6rem);
            line-height: 0.9;
            margin: 0.25rem 0 0.65rem;
            font-weight: 900;
        }
        .gj-lede {
            color: #c8d3ef;
            font-size: 1.08rem;
            max-width: 760px;
        }
        .gj-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            border: 1px solid var(--gj-border);
            border-radius: 999px;
            padding: 0.32rem 0.68rem;
            margin: 0.2rem 0.25rem 0.2rem 0;
            color: #dfe7fb;
            background: rgba(255,255,255,0.04);
            font-size: 0.82rem;
        }
        .gj-pipeline {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.7rem;
            margin-top: 1rem;
        }
        .gj-node {
            min-height: 118px;
            border: 1px solid var(--gj-border);
            border-radius: 18px;
            padding: 0.9rem;
            background: rgba(255,255,255,0.035);
        }
        .gj-node strong { display: block; margin-bottom: 0.35rem; }
        .gj-node span { color: var(--gj-muted); font-size: 0.88rem; }
        .gj-danger { color: var(--gj-red); }
        .gj-blue { color: var(--gj-blue); }
        .gj-green { color: var(--gj-green); }
        .gj-muted { color: var(--gj-muted); }
        .gj-dossier {
            border: 1px solid rgba(237, 28, 36, 0.38);
            border-radius: 24px;
            padding: 1rem 1.15rem;
            background: linear-gradient(145deg, rgba(237,28,36,0.12), rgba(14,18,32,0.95));
            margin-bottom: 1rem;
        }
        .gj-evidence {
            border-left: 3px solid var(--gj-red);
            padding: 0.75rem 0.9rem;
            background: rgba(237, 28, 36, 0.08);
            color: #ffd7d9;
            border-radius: 0 12px 12px 0;
        }
        @media (max-width: 900px) {
            .gj-pipeline { grid-template-columns: 1fr; }
            .block-container { padding-left: 1rem; padding-right: 1rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _short(model_id: str) -> str:
    return model_id.rsplit("/", 1)[-1]


def _pct(value: float) -> str:
    return f"{value:.0%}"


def _risk_color(asr: float) -> str:
    if asr >= 0.30:
        return AMD_RED
    if asr >= 0.10:
        return AMBER
    return SAFE_GREEN


def _load_secrets_into_env() -> None:
    try:
        secret_keys = list(st.secrets.keys())
    except Exception:  # noqa: B110 - no Streamlit secrets configured locally
        return
    for key in secret_keys:
        try:
            value = st.secrets[key]
        except Exception:  # noqa: B112 - one unreadable secret should not break boot
            continue
        if isinstance(value, str):
            os.environ.setdefault(key, value)


def _probe_settings() -> tuple[Any | None, str | None]:
    try:
        return load_settings(), None
    except ConfigError as exc:
        return None, str(exc)


@st.cache_data(show_spinner=False)
def _load_leaderboard() -> LeaderboardResult | None:
    try:
        return LeaderboardResult.model_validate_json(
            LEADERBOARD_PATH.read_text(encoding="utf-8")
        )
    except Exception:  # noqa: B110 - missing/invalid artifact hides only this surface
        return None


def _run_eval_sync(config: EvalConfig, *, offline: bool, settings: Any | None) -> EvalResult:
    if offline:
        coro = run_eval(
            config,
            engine_client=OfflineEngineClient(),
            target_client=OfflineTargetClient(),
        )
    else:
        coro = run_eval(config, settings=settings)
    return asyncio.run(coro)


def _default_config(settings: Any | None, offline: bool) -> EvalConfig:
    endpoint = "offline://simulated" if offline or settings is None else settings.target.base_url
    model = "weak-model-sim" if offline or settings is None else settings.target.model_id
    return EvalConfig(
        failure_mode=FailureMode.HALLUCINATION,
        n_cases=10,
        target_endpoint=endpoint,
        target_model_id=model,
    )


def _leaderboard_rows(board: LeaderboardResult) -> list[dict[str, Any]]:
    return [
        {
            "rank": idx,
            "target": _short(target.target_model_id),
            "ASR": _pct(target.attack_success_rate),
            "failed": f"{target.n_failed}/{target.n_cases}",
            "mean score": round(target.mean_score, 2),
            "time": f"{target.wall_clock_seconds:.1f}s",
        }
        for idx, target in enumerate(board.ranked, start=1)
    ]


def _render_header(board: LeaderboardResult | None) -> None:
    tokens = board.cost.total.total_tokens if board and board.cost else 0
    top_asr = (
        max((target.attack_success_rate for target in board.targets), default=0.0)
        if board
        else 0.0
    )
    safest = (
        min(board.ranked, key=lambda target: target.attack_success_rate).target_model_id
        if board and board.ranked
        else "pending"
    )
    st.markdown(
        f"""
        <section class="gj-hero">
          <div class="gj-kicker">Mission Control · AMD Developer Hackathon ACT II</div>
          <div class="gj-title">GemmaJudge</div>
          <p class="gj-lede">Adversarial LLM evaluation powered by Gemma on AMD.
          Gemma attacks. The target answers. Gemma judges.</p>
          <span class="gj-pill">Real Gemma run: {_escape(_pct(top_asr))} top ASR</span>
          <span class="gj-pill">Safest target: {_escape(_short(safest))}</span>
          <span class="gj-pill">Measured tokens: {tokens:,}</span>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### Mission Control")


def _render_pipeline() -> None:
    st.markdown(
        """
        <div class="gj-pipeline">
          <div class="gj-node">
            <strong class="gj-blue">Gemma Attacker</strong>
            <span>Generates targeted adversarial prompts.</span>
          </div>
          <div class="gj-node">
            <strong>Target Model</strong>
            <span>System under test answers each prompt.</span>
          </div>
          <div class="gj-node">
            <strong class="gj-blue">Gemma Judge</strong>
            <span>Scores 1-5 with reasoning and evidence.</span>
          </div>
          <div class="gj-node">
            <strong class="gj-danger">Risk Report</strong>
            <span>ASR, drill-down, cost, latency, AMD backend.</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_leaderboard_snapshot(board: LeaderboardResult | None) -> None:
    st.markdown("### Real Gemma run")
    if board is None:
        st.info(
            "No committed leaderboard artifact found. Generate it with "
            "`python -m gemmajudge.leaderboard_demo --n 8 --out docs/real_runs/leaderboard.json`."
        )
        return

    tokens = board.cost.total.total_tokens if board.cost else 0
    st.success(
        f"Real Gemma run · engine `{_short(board.engine_model_id)}` · "
        f"{len(board.targets)} targets · {len(board.attacks)} shared attacks · "
        f"{tokens:,} measured tokens · safest `{_short(board.ranked[-1].target_model_id)}`"
    )
    st.dataframe(
        pd.DataFrame(_leaderboard_rows(board)).set_index("rank"),
        use_container_width=True,
        height=220,
    )


def _render_live_console(
    settings: Any | None,
    config_error: str | None,
    *,
    key_prefix: str,
) -> tuple[EvalConfig, bool]:
    offline_default = settings is None
    offline = st.toggle(
        "Simulated demo (no keys)",
        value=offline_default,
        help="Runs the zero-key offline backend. It is illustrative only, not AMD proof.",
        key=f"{key_prefix}_offline",
    )

    default = _default_config(settings, offline)
    if not offline and config_error:
        st.error("Real backend is not configured. Missing setup:\n\n" + config_error)

    col_a, col_b = st.columns([1, 1])
    with col_a:
        endpoint = st.text_input(
            "Target endpoint",
            value=default.target_endpoint,
            disabled=offline,
            key=f"{key_prefix}_endpoint",
        )
    with col_b:
        model = st.text_input(
            "Target model id",
            value=default.target_model_id,
            disabled=offline,
            key=f"{key_prefix}_model",
        )

    col_c, col_d = st.columns([1, 1])
    with col_c:
        mode = st.selectbox(
            "Failure mode",
            [failure_mode.value for failure_mode in FailureMode],
            index=0,
            key=f"{key_prefix}_failure_mode",
        )
    with col_d:
        n_cases = st.slider(
            "Number of test cases",
            1,
            50,
            default.n_cases,
            key=f"{key_prefix}_n_cases",
        )

    config = EvalConfig(
        failure_mode=FailureMode(mode),
        n_cases=n_cases,
        target_endpoint=endpoint or default.target_endpoint,
        target_model_id=model or default.target_model_id,
    )

    if offline:
        st.warning(
            "SIMULATED RUN MODE - illustrative only. Not a real Gemma/AMD evaluation."
        )
    elif settings is not None:
        st.success(
            f"Real backend ready · engine `{_short(settings.model_id)}` · "
            f"backend `{settings.backend.value}`"
        )

    if st.button(
        "Run evaluation",
        type="primary",
        use_container_width=True,
        key=f"{key_prefix}_run",
    ):
        if not offline and settings is None:
            st.error("Cannot run: real backend is not configured. Turn on simulated demo mode.")
        else:
            with st.spinner("Gemma is attacking, the target is answering, Gemma is judging..."):
                try:
                    result = _run_eval_sync(config, offline=offline, settings=settings)
                except Exception as exc:  # noqa: BLE001 - UI must surface any engine failure
                    st.session_state["last_error"] = str(exc)
                else:
                    st.session_state["result"] = result
                    st.session_state["result_offline"] = offline
                    st.session_state.pop("last_error", None)

    if "last_error" in st.session_state:
        st.error("Run failed without exposing secrets: " + st.session_state["last_error"])

    return config, offline


def _score_distribution_fig(result: EvalResult) -> go.Figure:
    counts = {score: 0 for score in range(1, 6)}
    for verdict in result.verdicts:
        counts[verdict.score] += 1
    colors = [SAFE_GREEN, SAFE_GREEN, AMBER, AMD_RED, AMD_RED]
    fig = go.Figure(
        data=[
            go.Bar(
                x=list(counts.keys()),
                y=list(counts.values()),
                marker_color=colors,
                text=list(counts.values()),
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK),
        xaxis=dict(title="Judge score", dtick=1, gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(title="Cases", gridcolor="rgba(255,255,255,0.08)", rangemode="tozero"),
        showlegend=False,
    )
    return fig


def _token_usage_fig(cost: CostReport | None) -> go.Figure:
    labels = ["Attacker", "Target", "Judge"]
    values = [0, 0, 0]
    if cost:
        values = [
            cost.attacker.total_tokens,
            cost.target.total_tokens,
            cost.judge.total_tokens,
        ]
    fig = go.Figure(data=[go.Bar(x=labels, y=values, marker_color=[GEMMA_BLUE, MUTED, AMD_RED])])
    fig.update_layout(
        height=230,
        margin=dict(l=10, r=10, t=25, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK),
        yaxis=dict(title="Tokens", gridcolor="rgba(255,255,255,0.08)", rangemode="tozero"),
        showlegend=False,
    )
    return fig


def _verdict_label(score: int) -> str:
    if score >= 4:
        return "CONFIDENT FALSE"
    if score == 3:
        return "BORDERLINE"
    return "PASSED"


def _render_risk_report(result: EvalResult, *, offline: bool, key_prefix: str) -> None:
    st.markdown("## Risk Report")
    if offline:
        st.warning(
            "SIMULATED RUN - illustrative only. Not a real Gemma/AMD evaluation."
        )

    asr = result.attack_success_rate
    failed = sum(1 for verdict in result.verdicts if verdict.score >= 4)
    metrics = result.metrics
    cost = result.cost

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Attack Success Rate", _pct(asr), help="Cases the target failed: judge score >= 4")
    col2.metric("Failed cases", f"{failed}/{len(result.verdicts)}")
    col3.metric("Wall clock", f"{metrics.wall_clock_seconds:.2f}s" if metrics else "n/a")
    col4.metric("Throughput", f"{metrics.throughput_evals_per_sec:.2f}/s" if metrics else "n/a")

    backend_label = "simulated"
    if metrics and metrics.inference_backend:
        backend_label = metrics.inference_backend
    elif not offline:
        backend_label = "n/a"
    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Backend", backend_label)
    col6.metric("Engine model", _short(metrics.model_id) if metrics and metrics.model_id else "n/a")
    col7.metric("Measured tokens", f"{cost.total.total_tokens:,}" if cost else "0")
    col8.metric("Cost", f"${cost.usd:.4f}" if cost else "$0.0000")

    if cost and cost.price_source:
        st.caption(
            f"Cost is computed from measured engine tokens. Price source: {cost.price_source}"
        )
    elif cost:
        st.caption(
            "Cost is computed from measured tokens. Configure PRICE_* env vars to show "
            "a non-zero dollar value."
        )

    chart_left, chart_right = st.columns([1, 1])
    with chart_left:
        st.markdown("### Score Distribution")
        st.caption("1 = safe / correct / hedged · 5 = confident and false")
        st.plotly_chart(
            _score_distribution_fig(result),
            use_container_width=True,
            key=f"{key_prefix}_score_distribution",
        )
    with chart_right:
        st.markdown("### Token Usage")
        st.caption("Measured usage split by role")
        st.plotly_chart(
            _token_usage_fig(cost),
            use_container_width=True,
            key=f"{key_prefix}_token_usage",
        )

    _render_worst_case_dossier(result)
    _render_case_expanders(result)


def _render_worst_case_dossier(result: EvalResult) -> None:
    st.markdown("## Worst-Case Dossier")
    if not result.cases:
        st.info("No joined attack/verdict cases available for a dossier.")
        return

    case = max(result.cases, key=lambda item: item.verdict.score)
    verdict = case.verdict
    verdict_label = _escape(_verdict_label(verdict.score))
    st.markdown(
        f"""
        <div class="gj-dossier">
          <div class="gj-kicker">
            CASE {_escape(case.attack.id)} · SCORE {verdict.score}/5 · {verdict_label}
          </div>
          <h3>Gemma found the target's most brittle answer</h3>
          <p class="gj-muted">Targeted weakness: {_escape(case.attack.targeted_weakness)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("**Gemma attack**")
    st.write(_escape(case.attack.prompt))
    st.markdown("**Target response**")
    st.write(_escape(verdict.target_response))
    st.markdown("**Gemma judge**")
    st.write(_escape(verdict.reasoning))
    if verdict.evidence_span:
        st.markdown(
            f'<div class="gj-evidence"><strong>Evidence span:</strong> '
            f"{_escape(verdict.evidence_span)}</div>",
            unsafe_allow_html=True,
        )
    if result.consistency:
        bits = [
            f"{item.test_id}: [{', '.join(str(score) for score in item.scores)}] "
            f"stdev {item.stdev:.2f}"
            for item in result.consistency
        ]
        st.caption("Judge self-consistency · " + " · ".join(bits))


def _render_case_expanders(result: EvalResult) -> None:
    st.markdown("### Full Drill-Down")
    cases = sorted(result.cases, key=lambda item: item.verdict.score, reverse=True)
    for item in cases:
        verdict = item.verdict
        label = _verdict_label(verdict.score)
        with st.expander(
            f"{item.attack.id} · score {verdict.score}/5 · {label} · {item.attack.prompt[:82]}"
        ):
            st.markdown("**Gemma attack**")
            st.write(_escape(item.attack.prompt))
            st.caption(f"Rationale: {item.attack.rationale}")
            st.markdown("**Target response**")
            st.write(_escape(verdict.target_response))
            st.markdown("**Gemma judge**")
            st.write(_escape(verdict.reasoning))
            if verdict.evidence_span:
                st.markdown(f"**Evidence span:** `{verdict.evidence_span}`")


def _render_target_drilldown(board: LeaderboardResult, target: TargetReport) -> None:
    st.markdown(f"### Drill-down: `{_short(target.target_model_id)}`")
    cases = sorted(board.cases_for(target), key=lambda item: item.verdict.score, reverse=True)
    for item in cases:
        verdict = item.verdict
        label = _verdict_label(verdict.score)
        with st.expander(
            f"{item.attack.id} · score {verdict.score}/5 · {label} · {item.attack.prompt[:82]}"
        ):
            st.markdown("**Shared Gemma attack**")
            st.write(_escape(item.attack.prompt))
            st.markdown("**Target response**")
            st.write(_escape(verdict.target_response))
            st.markdown("**Gemma judge**")
            st.write(_escape(verdict.reasoning))
            if verdict.evidence_span:
                st.markdown(f"**Evidence span:** `{verdict.evidence_span}`")


def _render_full_leaderboard(board: LeaderboardResult | None) -> None:
    st.markdown("## Robustness Leaderboard")
    if board is None:
        st.info(
            "No committed real-Gemma leaderboard found. Generate one with "
            "`python -m gemmajudge.leaderboard_demo --n 8 --out docs/real_runs/leaderboard.json`."
        )
        return

    tokens = board.cost.total.total_tokens if board.cost else 0
    st.success(
        f"Real Gemma run · `{_short(board.engine_model_id)}` attacker+judge · "
        f"backend `{board.inference_backend or 'recorded'}` · "
        f"{len(board.attacks)} shared attacks · "
        f"{tokens:,} measured tokens"
    )
    st.dataframe(
        pd.DataFrame(_leaderboard_rows(board)).set_index("rank"),
        use_container_width=True,
        height=260,
    )

    ranked = board.ranked
    fig = go.Figure(
        data=[
            go.Bar(
                x=[_short(target.target_model_id) for target in ranked],
                y=[round(target.attack_success_rate * 100) for target in ranked],
                marker_color=[_risk_color(target.attack_success_rate) for target in ranked],
                text=[_pct(target.attack_success_rate) for target in ranked],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK),
        yaxis=dict(title="ASR (%)", gridcolor="rgba(255,255,255,0.08)", rangemode="tozero"),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    labels = [
        f"{_short(target.target_model_id)} · ASR {_pct(target.attack_success_rate)}"
        for target in ranked
    ]
    selected = st.selectbox(
        "Inspect target",
        range(len(ranked)),
        format_func=lambda idx: labels[idx],
        key="leaderboard_target",
    )
    _render_target_drilldown(board, ranked[selected])


def _render_amd_proof(settings: Any | None, config_error: str | None) -> None:
    st.markdown("## AMD Proof")
    st.markdown(
        "GemmaJudge uses one OpenAI-compatible client path for Fireworks and for "
        "self-hosted Gemma on AMD Instinct MI300X via vLLM + ROCm. The simulated "
        "mode is never counted as proof."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Backend", settings.backend.value if settings else "not configured")
    col2.metric("Engine model", _short(settings.model_id) if settings else "not configured")
    col3.metric("Request timeout", f"{settings.request_timeout_s:.0f}s" if settings else "<= 30s")
    col4.metric("Max concurrency", str(settings.max_concurrency) if settings else "8 default")

    if config_error:
        st.info("Real backend setup is pending: " + config_error)

    st.markdown("### Proof artifact checklist")
    rows = []
    for name in AMD_PROOF_FILES:
        path = AMD_PROOF_DIR / name
        rows.append(
            {
                "artifact": name,
                "status": "present" if path.exists() else "pending",
                "path": str(path.relative_to(ROOT)),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(
        "Pending files are shown honestly. The live demo can still run in simulated mode, "
        "but only real Fireworks/MI300X runs should be presented as Gemma/AMD evidence."
    )


def _render_stored_result(*, key_prefix: str) -> None:
    result = st.session_state.get("result")
    if result is not None:
        _render_risk_report(
            result,
            offline=bool(st.session_state.get("result_offline", False)),
            key_prefix=key_prefix,
        )


def main() -> None:
    _load_secrets_into_env()
    _install_theme()
    settings, config_error = _probe_settings()
    board = _load_leaderboard()

    tab_mission, tab_live, tab_board, tab_amd = st.tabs(
        ["Mission Control", "Live Evaluation", "Robustness Leaderboard", "AMD Proof"]
    )

    with tab_mission:
        _render_header(board)
        _render_pipeline()
        left, right = st.columns([1.1, 0.9], gap="large")
        with left:
            _render_leaderboard_snapshot(board)
        with right:
            st.markdown("### Live Evaluation")
            _render_live_console(settings, config_error, key_prefix="mission")
        _render_stored_result(key_prefix="mission_result")

    with tab_live:
        st.markdown("## Live Evaluation")
        _render_live_console(settings, config_error, key_prefix="live")
        _render_stored_result(key_prefix="live_result")

    with tab_board:
        _render_full_leaderboard(board)

    with tab_amd:
        _render_amd_proof(settings, config_error)


if __name__ == "__main__":
    main()
