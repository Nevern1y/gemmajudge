"""GemmaJudge Streamlit frontend.

Thin UI around the frozen backend seam: ``run_eval(config: EvalConfig) -> EvalResult``.
The public app is a recorded-run dashboard over committed real artifacts; live backends
remain env-configured for private/local demos.
"""

from __future__ import annotations

import asyncio
import html
import json
import os
from datetime import datetime
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
    RunMetrics,
    TargetReport,
)

ROOT = Path(__file__).parent
LEADERBOARD_PATH = ROOT / "docs" / "real_runs" / "leaderboard.json"
AMD_PROOF_DIR = ROOT / "docs" / "amd_proof"
AMD_PROOF_RUN_DIR = AMD_PROOF_DIR / "w7900"
AMD_PROOF_RESULT_PATH = AMD_PROOF_RUN_DIR / "eval_result.json"
AMD_PROOF_SCREENSHOTS = (
    AMD_PROOF_RUN_DIR / "proof.png",
    ROOT.parent / "w7900_proof.png",
)
AMD_PROOF_FILES = (
    "rocm_smi.txt",
    "versions.txt",
    "vllm_engine.log",
    "vllm_target.log",
    "serve_command.txt",
    "eval_result.json",
    "notes.md",
    "proof.png",
)
FINE_TUNE_PROOF_DIR = ROOT / "docs" / "fine_tune_eval"
FINE_TUNE_REPORT_PATH = FINE_TUNE_PROOF_DIR / "report.json"
FINE_TUNE_EVIDENCE = (
    FINE_TUNE_PROOF_DIR / "evidence_metrics.png",
    FINE_TUNE_PROOF_DIR / "evidence_analysis.png",
)

HISTORY_LIMIT = 12
SOURCE_AMD_PROOF = "amd_proof"
SOURCE_LIVE = "live"
SOURCE_RECORDED = "recorded"
SOURCE_SIMULATED = "simulated"

RUN_RECORDED_AMD = "Recorded AMD proof (W7900 ROCm)"
RUN_RECORDED_LEADERBOARD = "Recorded leaderboard target"
RUN_LIVE_BACKEND = "Live Gemma backend (advanced)"
RUN_SIMULATED = "Simulated fallback"

INK = "#050B14"
MUTED = "#5B6878"
BLUE = "#4DB8FF"
BLUE_DARK = "#075D92"
BLUE_SOFT = "#EAF7FF"
LINE = "#D7EAF8"
WHITE = "#FFFFFF"


st.set_page_config(
    page_title="GemmaJudge Live Demo",
    page_icon="GJ",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def _install_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --gj-ink: #050b14;
            --gj-muted: #5b6878;
            --gj-blue: #4db8ff;
            --gj-blue-dark: #075d92;
            --gj-blue-soft: #eaf7ff;
            --gj-line: #d7eaf8;
            --gj-white: #ffffff;
        }
        .stApp {
            background:
                radial-gradient(circle at 10% 0%, rgba(77, 184, 255, 0.23), transparent 28rem),
                linear-gradient(180deg, #ffffff 0%, #f5fbff 100%);
            color: var(--gj-ink);
        }
        header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"],
        #MainMenu, footer { display: none !important; visibility: hidden !important; }
        .block-container { max-width: 980px; padding-top: 1.2rem; padding-bottom: 2rem; }
        h1, h2, h3 { color: var(--gj-ink) !important; letter-spacing: -0.03em; }
        p, li, label, .stMarkdown { color: var(--gj-ink); }
        a { color: var(--gj-blue-dark); }
        div[data-testid="stRadio"] [role="radiogroup"] {
            gap: 0.45rem;
            width: max-content;
            max-width: 100%;
            margin: 0 auto 1rem;
            padding: 0.35rem;
            border: 1px solid var(--gj-line);
            border-radius: 999px;
            background: rgba(234, 247, 255, 0.82);
        }
        div[data-testid="stRadio"] label {
            border-radius: 999px;
            padding: 0.18rem 0.62rem;
        }
        div[data-testid="stRadio"] label p {
            color: var(--gj-muted) !important;
            font-weight: 760;
        }
        [data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.94);
            border: 1px solid var(--gj-line);
            border-radius: 16px;
            padding: 0.72rem;
            box-shadow: 0 12px 32px rgba(7, 93, 146, 0.07);
        }
        [data-testid="stMetricLabel"] p { color: var(--gj-muted) !important; }
        [data-testid="stMetricValue"] { color: var(--gj-ink) !important; }
        .stButton button {
            min-height: 2.75rem;
            border-radius: 999px !important;
            border: 1px solid var(--gj-line) !important;
            background: var(--gj-white) !important;
            color: var(--gj-ink) !important;
            font-weight: 850 !important;
        }
        .stButton button[kind="primary"] {
            background: var(--gj-blue-dark) !important;
            color: var(--gj-white) !important;
            border-color: var(--gj-blue-dark) !important;
        }
        div[data-testid="stExpander"] {
            border: 1px solid var(--gj-line) !important;
            border-radius: 18px !important;
            background: rgba(255, 255, 255, 0.96) !important;
            box-shadow: 0 12px 32px rgba(7, 93, 146, 0.07);
        }
        div[data-testid="stExpander"] details,
        div[data-testid="stExpander"] summary,
        div[data-testid="stExpander"] div {
            background: transparent !important;
        }
        div[data-testid="stExpander"] * {
            color: var(--gj-ink) !important;
        }
        div[data-testid="stExpander"] summary p {
            color: var(--gj-ink) !important;
            font-weight: 820;
        }
        div[data-baseweb="select"] > div, input, textarea {
            border-radius: 12px !important;
            border-color: var(--gj-line) !important;
            background: var(--gj-white) !important;
            color: var(--gj-ink) !important;
        }
        div[data-baseweb="popover"], div[data-baseweb="menu"], ul[role="listbox"] {
            background: var(--gj-white) !important;
            color: var(--gj-ink) !important;
            border: 1px solid var(--gj-line) !important;
            border-radius: 14px !important;
        }
        div[data-baseweb="popover"] *, div[data-baseweb="menu"] *, ul[role="listbox"] * {
            background: var(--gj-white) !important;
            color: var(--gj-ink) !important;
        }
        .gj-hero, .gj-panel, .gj-empty {
            border: 1px solid var(--gj-line);
            background: rgba(255, 255, 255, 0.94);
            box-shadow: 0 16px 42px rgba(7, 93, 146, 0.08);
        }
        .gj-hero {
            border-radius: 24px;
            padding: 1rem 1.1rem;
            margin-bottom: 0.85rem;
            text-align: center;
        }
        .gj-panel {
            border-radius: 22px;
            padding: 1rem;
            margin-bottom: 0.85rem;
        }
        .gj-run-title {
            text-align: center;
            margin-bottom: 0.75rem;
        }
        .gj-live-head {
            border-radius: 18px;
            padding: 1rem;
            margin-bottom: 0.85rem;
            background: linear-gradient(135deg, var(--gj-blue-soft), #ffffff);
            color: var(--gj-ink);
            text-align: center;
        }
        .gj-live-head .gj-kicker { color: var(--gj-blue-dark); }
        .gj-live-head strong {
            display: block;
            color: var(--gj-ink);
            font-size: 1.25rem;
            letter-spacing: -0.02em;
            margin-top: 0.15rem;
        }
        .gj-live-head span { color: var(--gj-muted); font-size: 0.92rem; }
        .gj-result-head {
            display: flex;
            justify-content: space-between;
            gap: 0.75rem;
            align-items: flex-start;
            margin-bottom: 0.8rem;
        }
        .gj-result-head h2 { margin: 0; }
        .gj-result-head span { color: var(--gj-muted); font-size: 0.9rem; }
        .gj-read-card {
            border: 1px solid var(--gj-line);
            border-radius: 16px;
            background: var(--gj-white);
            padding: 0.85rem 0.95rem;
            margin: 0.7rem 0;
            box-shadow: 0 8px 22px rgba(7, 93, 146, 0.05);
        }
        .gj-read-card strong { color: var(--gj-ink); }
        .gj-read-card p {
            color: var(--gj-ink);
            white-space: pre-wrap;
            margin: 0.45rem 0 0;
            line-height: 1.45;
        }
        .gj-read-body {
            color: var(--gj-ink);
            margin: 0.45rem 0 0;
            line-height: 1.45;
        }
        .gj-result-empty {
            min-height: 360px;
            display: grid;
            place-items: center;
            text-align: center;
        }
        .gj-result-empty strong { display: block; margin-bottom: 0.35rem; }
        .gj-empty {
            border-radius: 18px;
            padding: 0.85rem 1rem;
            color: var(--gj-muted);
        }
        .gj-kicker {
            color: var(--gj-blue-dark);
            font-size: 0.72rem;
            font-weight: 900;
            letter-spacing: 0.14em;
            text-transform: uppercase;
        }
        .gj-title {
            color: var(--gj-ink);
            font-size: clamp(2.35rem, 6vw, 4.5rem);
            font-weight: 950;
            line-height: 0.92;
            margin: 0.08rem 0 0.32rem;
        }
        .gj-lede {
            color: var(--gj-muted);
            font-size: 1.02rem;
            line-height: 1.42;
            max-width: 720px;
            margin: 0 auto 0.35rem;
        }
        .gj-pill {
            display: inline-flex;
            border: 1px solid var(--gj-line);
            border-radius: 999px;
            padding: 0.28rem 0.6rem;
            margin: 0.2rem 0.2rem 0.1rem 0;
            background: var(--gj-blue-soft);
            color: var(--gj-ink);
            font-size: 0.82rem;
            font-weight: 780;
        }
        .gj-case {
            border-left: 4px solid var(--gj-blue);
            border-radius: 0 14px 14px 0;
            background: #f5fbff;
            padding: 0.75rem 0.9rem;
            margin: 0.7rem 0;
        }
        .gj-muted { color: var(--gj-muted); }
        @media (max-width: 760px) {
            .block-container { padding-left: 0.75rem; padding-right: 0.75rem; }
            .gj-title { font-size: 2.6rem; }
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


@st.cache_data(show_spinner=False)
def _load_amd_proof_result() -> EvalResult | None:
    try:
        return EvalResult.model_validate_json(
            AMD_PROOF_RESULT_PATH.read_text(encoding="utf-8")
        )
    except Exception:  # noqa: B110 - missing/invalid proof hides only this summary
        return None


@st.cache_data(show_spinner=False)
def _load_fine_tune_report() -> dict[str, Any] | None:
    try:
        report = json.loads(FINE_TUNE_REPORT_PATH.read_text(encoding="utf-8"))
        if not isinstance(report, dict) or "models" not in report:
            return None
        return report
    except (OSError, json.JSONDecodeError):
        return None


def _first_existing_path(paths: tuple[Path, ...]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
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


def _recorded_target_options(board: LeaderboardResult | None) -> list[str]:
    if board is None:
        return []
    return [_short(target.target_model_id) for target in board.ranked]


def _recorded_eval_from_board(board: LeaderboardResult, target_label: str) -> EvalResult:
    by_short = {_short(target.target_model_id): target for target in board.targets}
    target = by_short[target_label]
    return EvalResult(
        config=EvalConfig(
            failure_mode=board.failure_mode,
            n_cases=target.n_cases,
            target_endpoint="recorded://docs/real_runs/leaderboard.json",
            target_model_id=target.target_model_id,
        ),
        verdicts=target.verdicts,
        attacks=board.attacks,
        cost=board.cost,
        metrics=RunMetrics(
            wall_clock_seconds=target.wall_clock_seconds,
            n_cases=target.n_cases,
            inference_backend="recorded Gemma-27B Fireworks run",
            model_id=board.engine_model_id,
            target_model_id=target.target_model_id,
        ),
    )


def _default_config(settings: Any | None, offline: bool) -> EvalConfig:
    if offline or settings is None:
        return EvalConfig(
            failure_mode=FailureMode.HALLUCINATION,
            n_cases=10,
            target_endpoint="offline://simulated",
            target_model_id="weak-model-sim",
        )
    return EvalConfig(
        failure_mode=FailureMode.HALLUCINATION,
        n_cases=8,
        target_endpoint=settings.target.base_url,
        target_model_id=settings.target.model_id,
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


def _score_distribution_fig(result: EvalResult) -> go.Figure:
    counts = {score: 0 for score in range(1, 6)}
    for verdict in result.verdicts:
        counts[verdict.score] += 1
    fig = go.Figure(
        data=[
            go.Bar(
                x=list(counts.keys()),
                y=list(counts.values()),
                marker_color=[BLUE_SOFT, "#CDEFFF", "#96DAFF", "#5BC1FF", BLUE_DARK],
                text=list(counts.values()),
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        height=230,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK),
        xaxis=dict(title="Judge score", dtick=1, gridcolor=LINE),
        yaxis=dict(title="Cases", gridcolor=LINE, rangemode="tozero"),
        showlegend=False,
    )
    return fig


def _token_usage_fig(cost: CostReport | None) -> go.Figure:
    values = [0, 0, 0]
    if cost:
        values = [
            cost.attacker.total_tokens,
            cost.target.total_tokens,
            cost.judge.total_tokens,
        ]
    fig = go.Figure(
        data=[go.Bar(x=["Attacker", "Target", "Judge"], y=values, marker_color=BLUE)]
    )
    fig.update_layout(
        height=210,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK),
        yaxis=dict(title="Tokens", gridcolor=LINE, rangemode="tozero"),
        showlegend=False,
    )
    return fig


def _result_label(entry: dict[str, Any]) -> str:
    result: EvalResult = entry["result"]
    metrics = result.metrics
    failed = sum(1 for verdict in result.verdicts if verdict.score >= 4)
    source = entry.get("source")
    if source == SOURCE_AMD_PROOF:
        backend = "VERIFIED AMD"
    elif source == SOURCE_RECORDED:
        backend = "RECORDED REAL"
    elif entry.get("offline"):
        backend = "SIMULATED"
    else:
        backend = metrics.inference_backend if metrics else "REAL"
    return (
        f"{entry['created_at']} · {backend} · ASR {_pct(result.attack_success_rate)} · "
        f"{failed}/{len(result.verdicts)} failed · {_short(result.config.target_model_id)}"
    )


def _push_result(
    result: EvalResult,
    *,
    offline: bool,
    source: str,
    created_at: str | None = None,
) -> None:
    history = list(st.session_state.get("result_history", []))
    history.insert(
        0,
        {
            "created_at": created_at or datetime.now().strftime("%H:%M:%S"),
            "offline": offline,
            "source": source,
            "result": result,
        },
    )
    st.session_state["result_history"] = history[:HISTORY_LIMIT]


def _seed_verified_amd_result(proof: EvalResult | None) -> None:
    if proof is None or st.session_state.get("verified_amd_seeded"):
        return
    if not st.session_state.get("result_history"):
        _push_result(
            proof,
            offline=False,
            source=SOURCE_AMD_PROOF,
            created_at="recorded AMD proof",
        )
    st.session_state["verified_amd_seeded"] = True


def _render_read_card(title: str, body: str, eyebrow: str | None = None) -> None:
    label = f'<div class="gj-kicker">{_escape(eyebrow)}</div>' if eyebrow else ""
    body_html = _escape(body).replace("\n", "<br />")
    st.markdown(
        f"""
        <div class="gj-read-card">
          {label}
          <strong>{_escape(title)}</strong>
          <div class="gj-read-body">{body_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_header(settings: Any | None) -> None:
    backend = settings.backend.value if settings else "recorded artifact viewer"
    model = _short(settings.model_id) if settings else "Gemma proof + real runs"
    st.markdown(
        f"""
        <section class="gj-hero">
          <div class="gj-kicker">Mission Control · AMD Developer Hackathon ACT II · Track 3</div>
          <div class="gj-title">GemmaJudge</div>
          <p class="gj-lede"><strong>Gemma attacks. The target answers. Gemma judges.</strong>
          Run a hallucination eval, then open the evidence behind the score.</p>
          <span class="gj-pill">Backend: {_escape(backend)}</span>
          <span class="gj-pill">Engine: {_escape(model)}</span>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _render_run_form(
    settings: Any | None,
    config_error: str | None,
    board: LeaderboardResult | None,
) -> None:
    with st.container(border=True):
        st.markdown(
            """
            <div class="gj-live-head">
              <div class="gj-kicker">Evaluation Console</div>
              <strong>Load a real run</strong>
              <span>Replay committed real runs, or switch to a private live backend.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        modes = [RUN_RECORDED_AMD]
        if board and board.ranked:
            modes.append(RUN_RECORDED_LEADERBOARD)
        modes.append(RUN_LIVE_BACKEND)
        modes.append(RUN_SIMULATED)
        run_mode = st.selectbox(
            "Run source",
            modes,
            help="Recorded runs are real committed GemmaJudge artifacts, not generated mock data.",
            key="run_source",
        )

        recorded_target = None
        if run_mode == RUN_RECORDED_LEADERBOARD and board:
            options = _recorded_target_options(board)
            recorded_target = st.selectbox(
                "Recorded target",
                options,
                key="recorded_target",
            )

        offline = run_mode == RUN_SIMULATED
        live = run_mode == RUN_LIVE_BACKEND
        editable_run = live or offline
        if run_mode == RUN_RECORDED_AMD and (proof := _load_amd_proof_result()):
            default = proof.config
        elif run_mode == RUN_RECORDED_LEADERBOARD and board and recorded_target:
            default = _recorded_eval_from_board(board, recorded_target).config
        else:
            default = _default_config(settings, offline or not live)

        col_left, col_right = st.columns([1, 1])
        with col_left:
            failure_mode = st.selectbox(
                "Failure mode",
                [mode.value for mode in FailureMode],
                index=[mode.value for mode in FailureMode].index(default.failure_mode.value),
                disabled=not editable_run,
                key="run_failure_mode",
            )
        with col_right:
            n_cases = st.slider(
                "Test cases",
                min_value=1,
                max_value=20,
                value=min(default.n_cases, 20),
                help="Small live batches keep the demo under the 30s rule.",
                disabled=not editable_run,
                key="run_n_cases",
            )

        with st.expander("Target settings", expanded=False):
            endpoint = st.text_input(
                "Target endpoint",
                value=default.target_endpoint,
                disabled=not live,
                key="run_endpoint",
            )
            model = st.text_input(
                "Target model id",
                value=default.target_model_id,
                disabled=not live,
                key="run_model",
            )

        config = EvalConfig(
            failure_mode=FailureMode(failure_mode),
            n_cases=n_cases,
            target_endpoint=endpoint or default.target_endpoint,
            target_model_id=model or default.target_model_id,
        )

        if run_mode == RUN_RECORDED_AMD:
            st.success(
                "Recorded real AMD run - Gemma attacker+judge served with vLLM + ROCm on W7900."
            )
        elif run_mode == RUN_RECORDED_LEADERBOARD:
            st.success(
                "Recorded real leaderboard - Gemma-27B attacker+judge evaluated live target models."
            )
        elif offline:
            st.info("SIMULATED RUN - illustrative only. Not a real Gemma/AMD evaluation.")
        elif live and config_error:
            st.error(
                "Private live backend is not configured on this public URL. Recorded real runs "
                "above are the submitted evidence. To run live Gemma, set env vars from "
                ".env.example. Details: "
                + config_error
            )
        elif live:
            st.info(
                f"Real backend ready: `{settings.backend.value}` · "
                f"`{_short(settings.model_id)}`"
            )

        if st.button("Load evaluation", type="primary", use_container_width=True, key="run_button"):
            if run_mode == RUN_RECORDED_AMD:
                proof = _load_amd_proof_result()
                if proof is None:
                    st.error("Recorded AMD proof artifact is missing or invalid.")
                else:
                    _push_result(
                        proof,
                        offline=False,
                        source=SOURCE_AMD_PROOF,
                        created_at="recorded AMD proof",
                    )
                    st.info("Loaded recorded AMD proof run.")
            elif run_mode == RUN_RECORDED_LEADERBOARD:
                if board is None or recorded_target is None:
                    st.error("Recorded leaderboard artifact is missing or invalid.")
                else:
                    result = _recorded_eval_from_board(board, recorded_target)
                    _push_result(
                        result,
                        offline=False,
                        source=SOURCE_RECORDED,
                        created_at="recorded leaderboard",
                    )
                    st.info("Loaded recorded real leaderboard target.")
            elif live and settings is None:
                st.error(
                    "Cannot run the private live backend yet. Use a recorded run or configure env."
                )
            else:
                with st.spinner("Running attacker -> target -> judge..."):
                    try:
                        result = _run_eval_sync(config, offline=offline, settings=settings)
                    except Exception as exc:  # noqa: BLE001 - Streamlit must show demo-safe failures
                        st.error("Run failed without exposing secrets: " + str(exc))
                    else:
                        _push_result(
                            result,
                            offline=offline,
                            source=SOURCE_SIMULATED if offline else SOURCE_LIVE,
                        )
                        st.info("Result added below.")


def _render_results() -> None:
    history = list(st.session_state.get("result_history", []))
    if not history:
        st.markdown(
            """
            <div class="gj-empty gj-result-empty">
              <div>
                <strong>Results will appear here.</strong>
                Run an evaluation on the left. The latest report will be readable here from
                summary to evidence without jumping around the page.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        """
        <div class="gj-result-head">
          <div>
            <div class="gj-kicker">Results</div>
            <h2>Read the report</h2>
          </div>
          <span>Newest run is selected by default.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    selected = st.selectbox(
        "Stored runs",
        range(len(history)),
        format_func=lambda idx: _result_label(history[idx]),
        key="selected_result",
    )
    entry = history[selected]
    _render_risk_report(
        entry["result"],
        offline=bool(entry.get("offline")),
        source=str(entry.get("source") or SOURCE_LIVE),
    )

    if st.button("Clear results", key="clear_results"):
        st.session_state.pop("result_history", None)
        st.rerun()


def _render_risk_report(result: EvalResult, *, offline: bool, source: str = SOURCE_LIVE) -> None:
    st.markdown("### 1. Risk Report Summary")
    failed = sum(1 for verdict in result.verdicts if verdict.score >= 4)
    if offline:
        st.info("SIMULATED RUN - illustrative only. Not a real Gemma/AMD evaluation.")
    elif source == SOURCE_AMD_PROOF:
        st.success(
            "VERIFIED AMD RUN - recorded W7900 ROCm artifact from "
            "docs/amd_proof/w7900/eval_result.json. "
            f"ASR {_pct(result.attack_success_rate)} ({failed}/{len(result.verdicts)} failed)."
        )
    elif source == SOURCE_RECORDED:
        st.success(
            "RECORDED REAL RUN - committed GemmaJudge leaderboard artifact from "
            "docs/real_runs/leaderboard.json. "
            f"ASR {_pct(result.attack_success_rate)} ({failed}/{len(result.verdicts)} failed)."
        )
    metrics = result.metrics

    col1, col2, col3 = st.columns(3)
    col1.metric("Attack Success Rate", _pct(result.attack_success_rate))
    col2.metric("Failed cases", f"{failed}/{len(result.verdicts)}")
    col3.metric("Wall clock", f"{metrics.wall_clock_seconds:.2f}s" if metrics else "n/a")

    _render_worst_case(result)
    _render_case_browser(result)


def _render_worst_case(result: EvalResult) -> None:
    st.markdown("### 2. Worst-Case Dossier")
    if not result.cases:
        st.info("No joined attack/verdict cases are available for this result.")
        return

    case = max(result.cases, key=lambda item: item.verdict.score)
    verdict = case.verdict
    _render_read_card(
        f"Case {case.attack.id} · score {verdict.score}/5",
        case.attack.targeted_weakness,
        "Targeted weakness",
    )
    _render_read_card("Gemma attack", case.attack.prompt)
    _render_read_card("Target response", verdict.target_response)
    _render_read_card("Gemma judge", verdict.reasoning)
    if verdict.evidence_span:
        st.caption("Evidence span: " + verdict.evidence_span)
    if result.consistency:
        summary = "; ".join(
            f"{item.test_id}: {item.scores} stdev {item.stdev:.2f}"
            for item in result.consistency
        )
        st.caption("Judge self-consistency: " + summary)


def _render_case_browser(result: EvalResult) -> None:
    cases = sorted(result.cases, key=lambda item: item.verdict.score, reverse=True)
    if not cases:
        return
    with st.expander("Inspect another case", expanded=False):
        selected = st.selectbox(
            "Case",
            range(len(cases)),
            format_func=lambda idx: (
                f"{cases[idx].attack.id} · score {cases[idx].verdict.score}/5 · "
                f"{cases[idx].attack.prompt[:70]}"
            ),
            key=f"case_browser_{id(result)}",
        )
        case = cases[selected]
        verdict = case.verdict
        _render_read_card("Gemma attack", case.attack.prompt)
        _render_read_card("Target response", verdict.target_response)
        _render_read_card("Gemma judge", verdict.reasoning)


def _render_leaderboard(board: LeaderboardResult | None) -> None:
    st.markdown("## Robustness Leaderboard")
    if board is None:
        st.info("No committed real-Gemma leaderboard artifact found.")
        return

    tokens = board.cost.total.total_tokens if board.cost else 0
    safest = _short(board.ranked[-1].target_model_id) if board.ranked else "n/a"
    top_asr = max((target.attack_success_rate for target in board.targets), default=0.0)
    st.markdown(
        f"""
        <div class="gj-read-card">
          <div class="gj-kicker">Real Gemma run</div>
          <strong>{_escape(_short(board.engine_model_id))} attacker + judge</strong>
          <p>Backend: {_escape(board.inference_backend or 'recorded')} ·
          safest target: {_escape(safest)} · {len(board.attacks)} shared attacks ·
          {tokens:,} measured tokens.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col1, col2, col3 = st.columns(3)
    col1.metric("Highest ASR", _pct(top_asr))
    col2.metric("Safest target", safest)
    col3.metric("Targets", str(len(board.targets)))
    st.dataframe(
        pd.DataFrame(_leaderboard_rows(board)).set_index("rank"),
        use_container_width=True,
        height=240,
    )

    ranked = board.ranked
    fig = go.Figure(
        data=[
            go.Bar(
                x=[_short(target.target_model_id) for target in ranked],
                y=[round(target.attack_success_rate * 100) for target in ranked],
                marker_color=BLUE_DARK,
                text=[_pct(target.attack_success_rate) for target in ranked],
                textposition="outside",
                textfont=dict(color=INK, size=14),
            )
        ]
    )
    fig.update_layout(
        title=dict(text="Attack Success Rate by target", font=dict(color=INK, size=18)),
        height=320,
        margin=dict(l=10, r=10, t=52, b=10),
        paper_bgcolor=WHITE,
        plot_bgcolor=WHITE,
        font=dict(color=INK),
        xaxis=dict(tickfont=dict(color=INK), gridcolor=LINE),
        yaxis=dict(
            title=dict(text="ASR (%)", font=dict(color=INK)),
            tickfont=dict(color=INK),
            gridcolor=LINE,
            range=[0, max(35, round(top_asr * 100) + 10)],
        ),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    selected = st.selectbox(
        "Inspect target",
        range(len(ranked)),
        format_func=lambda idx: (
            f"{_short(ranked[idx].target_model_id)} · "
            f"{_pct(ranked[idx].attack_success_rate)} ASR"
        ),
        key="leaderboard_target",
    )
    _render_target_drilldown(board, ranked[selected])


def _render_target_drilldown(board: LeaderboardResult, target: TargetReport) -> None:
    st.markdown(f"### Target drill-down: `{_short(target.target_model_id)}`")
    cases = sorted(board.cases_for(target), key=lambda item: item.verdict.score, reverse=True)
    for item in cases[:8]:
        verdict = item.verdict
        with st.expander(f"{item.attack.id} · score {verdict.score}/5 · {item.attack.prompt[:80]}"):
            st.markdown("**Shared Gemma attack**")
            st.write(item.attack.prompt)
            st.markdown("**Target response**")
            st.write(verdict.target_response)
            st.markdown("**Gemma judge**")
            st.write(verdict.reasoning)
            if verdict.evidence_span:
                st.caption("Evidence span: " + verdict.evidence_span)


def _render_fine_tune_proof() -> None:
    report = _load_fine_tune_report()
    st.markdown("## Fine-Tune Proof")
    st.markdown(
        "Recorded base-vs-tuned judge evaluation from the AMD ROCm training session. "
        "The claim is deliberately scoped to the committed 56-example validation run."
    )
    if report is None:
        st.info("Fine-tune report could not be loaded from docs/fine_tune_eval/report.json.")
        return

    models = report.get("models", {})
    base = models.get("base", {})
    tuned = models.get("tuned", {})
    base_metrics = base.get("metrics", {})
    tuned_metrics = tuned.get("metrics", {})
    n_examples = int(report.get("n_validation_examples", 0))
    model_route = (
        f"{_escape(base.get('model', 'base judge'))} -> "
        f"{_escape(tuned.get('model', 'tuned judge'))}"
    )

    st.markdown(
        f"""
        <div class="gj-read-card">
          <div class="gj-kicker">Recorded ROCm result</div>
          <strong>{model_route}</strong>
          <p>{n_examples} held-out examples · direct Transformers generation on ROCm ·
          selected champion: {_escape(report.get('champion', 'n/a'))}.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    json_base = float(base_metrics.get("json_validity", 0.0))
    json_tuned = float(tuned_metrics.get("json_validity", 0.0))
    pass_base = float(base_metrics.get("pass_fail_accuracy", 0.0))
    pass_tuned = float(tuned_metrics.get("pass_fail_accuracy", 0.0))
    f1_base = float(base_metrics.get("macro_f1_violation_detected", 0.0))
    f1_tuned = float(tuned_metrics.get("macro_f1_violation_detected", 0.0))
    mae_base = float(base_metrics.get("mean_absolute_score_error", 0.0))
    mae_tuned = float(tuned_metrics.get("mean_absolute_score_error", 0.0))
    exact_base = float(base_metrics.get("exact_score_accuracy", 0.0))
    exact_tuned = float(tuned_metrics.get("exact_score_accuracy", 0.0))
    latency_base = float(base_metrics.get("avg_latency_s", 0.0))
    latency_tuned = float(tuned_metrics.get("avg_latency_s", 0.0))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Valid JSON", f"{json_tuned:.1%}", f"{(json_tuned - json_base) * 100:+.1f} pp")
    col2.metric(
        "Pass/fail accuracy",
        f"{pass_tuned:.1%}",
        f"{(pass_tuned - pass_base) * 100:+.1f} pp",
    )
    col3.metric("Macro-F1", f"{f1_tuned:.3f}", f"{f1_tuned - f1_base:+.3f}")
    col4.metric(
        "Score MAE",
        f"{mae_tuned:.2f}",
        f"{mae_tuned - mae_base:+.2f}",
        delta_color="inverse",
    )

    comparison = pd.DataFrame(
        [
            {
                "metric": "JSON validity",
                "base": f"{json_base:.1%}",
                "tuned": f"{json_tuned:.1%}",
            },
            {
                "metric": "Exact score accuracy",
                "base": f"{exact_base:.1%}",
                "tuned": f"{exact_tuned:.1%}",
            },
            {
                "metric": "Pass/fail accuracy",
                "base": f"{pass_base:.1%}",
                "tuned": f"{pass_tuned:.1%}",
            },
            {
                "metric": "Violation macro-F1",
                "base": f"{f1_base:.3f}",
                "tuned": f"{f1_tuned:.3f}",
            },
            {
                "metric": "Mean absolute score error",
                "base": f"{mae_base:.2f}",
                "tuned": f"{mae_tuned:.2f}",
            },
            {
                "metric": "Average latency",
                "base": f"{latency_base:.2f}s",
                "tuned": f"{latency_tuned:.2f}s",
            },
        ]
    )
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    evidence = [path for path in FINE_TUNE_EVIDENCE if path.exists()]
    if evidence:
        st.markdown("### Committed evidence")
        columns = st.columns(len(evidence))
        for column, path in zip(columns, evidence, strict=True):
            column.image(
                str(path),
                caption=path.name,
                use_container_width=True,
            )

    st.info(
        "Scoped claim: this tuned checkpoint beat its base model on the recorded "
        "56-example validation split. A larger human-reviewed benchmark is still needed "
        "for a general judge-quality claim."
    )
    st.caption(
        "Source: docs/fine_tune_eval/report.json · training and evaluation ran on AMD ROCm; "
        "the recorded serving fallback used direct Transformers generation."
    )


def _render_amd_proof(settings: Any | None, config_error: str | None) -> None:
    proof = _load_amd_proof_result()
    screenshot = _first_existing_path(AMD_PROOF_SCREENSHOTS)
    st.markdown("## AMD Proof")
    st.markdown(
        "Real proof screenshot first. The executed proof is AMD Radeon PRO W7900 via "
        "vLLM + ROCm; MI300X is included as the AMD Instinct reference runbook."
    )

    if screenshot:
        st.image(
            str(screenshot),
            caption="Real AMD proof screenshot: ROCm/vLLM GemmaJudge run on AMD hardware.",
            use_container_width=True,
        )
    else:
        st.info("No proof screenshot image found. Expected w7900_proof.png or w7900/proof.png.")

    st.markdown("### Verified AMD run")
    if proof and proof.metrics:
        failed = sum(1 for verdict in proof.verdicts if verdict.score >= 4)
        proof_rows = [
            {"item": "Hardware", "value": "AMD Radeon PRO W7900 / gfx1100"},
            {"item": "Stack", "value": "ROCm 7.2 + vLLM"},
            {"item": "Attacker + Judge", "value": _short(proof.metrics.model_id)},
            {"item": "Target", "value": _short(proof.metrics.target_model_id)},
            {"item": "ASR", "value": _pct(proof.attack_success_rate)},
            {"item": "Failed cases", "value": f"{failed}/{len(proof.verdicts)}"},
            {"item": "Throughput", "value": "136.9 tok/s"},
        ]
        st.dataframe(pd.DataFrame(proof_rows), use_container_width=True, hide_index=True)
    else:
        st.info("AMD proof result could not be loaded from docs/amd_proof/w7900/eval_result.json.")

    if config_error:
        st.info(
            "This public URL is a recorded-run dashboard over committed real artifacts. "
            "Private live Gemma can be enabled with Streamlit Secrets or Docker env vars "
            "from .env.example when needed. Details: "
            + config_error
        )

    with st.expander("Live backend config", expanded=False):
        live_rows = [
            {
                "item": "Live URL backend",
                "value": settings.backend.value if settings else "not configured",
            },
            {
                "item": "Live engine",
                "value": _short(settings.model_id) if settings else "not configured",
            },
            {
                "item": "Request guardrail",
                "value": f"{settings.request_timeout_s:.0f}s" if settings else "<= 30s",
            },
        ]
        st.dataframe(pd.DataFrame(live_rows), use_container_width=True, hide_index=True)

    st.markdown("### Artifact checklist")
    rows = []
    for name in AMD_PROOF_FILES:
        path = AMD_PROOF_RUN_DIR / name
        rows.append(
            {
                "artifact": name,
                "status": "present" if path.exists() else "pending",
                "path": str(path.relative_to(ROOT)),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.caption(
        "MI300X reference files: docs/amd_proof/mi300x_gemma.ipynb and "
        "docs/amd_proof/serve_gemma_mi300x.sh. They are runbooks; the committed proof "
        "above is the executed W7900 ROCm run."
    )


def main() -> None:
    _load_secrets_into_env()
    _install_theme()
    settings, config_error = _probe_settings()
    board = _load_leaderboard()
    _seed_verified_amd_result(_load_amd_proof_result())

    page = st.radio(
        "Navigation",
        ["Mission Control", "Fine-Tune Proof", "Leaderboard", "AMD Proof"],
        horizontal=True,
        label_visibility="collapsed",
        key="page_nav",
    )

    if page == "Mission Control":
        _render_header(settings)
        live_col, result_col = st.columns([0.42, 0.58], gap="large")
        with live_col:
            _render_run_form(settings, config_error, board)
        with result_col:
            _render_results()
    elif page == "Fine-Tune Proof":
        _render_fine_tune_proof()
    elif page == "Leaderboard":
        _render_leaderboard(board)
    else:
        _render_amd_proof(settings, config_error)


if __name__ == "__main__":
    main()
