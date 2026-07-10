"""Integration smoke tests for the Streamlit Mission Control UI.

The app runs in-process with Streamlit's AppTest harness. Tests stay offline:
with no env configured, the UI defaults to committed recorded AMD artifacts and
renders the real-Gemma leaderboard and fine-tune report from docs/.
"""

from streamlit.testing.v1 import AppTest

_APP = "app.py"


def _texts(at: AppTest) -> list[str]:
    values: list[str] = []
    for group in (at.markdown, at.caption, at.info, at.success, at.warning, at.error):
        values.extend(getattr(item, "value", "") for item in group)
    values.extend(getattr(item, "label", "") for item in at.metric)
    return values


def test_app_loads_mission_control_without_exception():
    at = AppTest.from_file(_APP, default_timeout=60).run()
    assert not at.exception
    text = "\n".join(_texts(at))
    assert "GemmaJudge" in text
    assert "Mission Control" in text
    assert "Gemma attacks" in text
    metric_values = {metric.label: metric.value for metric in at.metric}
    assert "VERIFIED AMD RUN" in text
    assert metric_values["Attack Success Rate"] == "80%"
    assert metric_values["Failed cases"] == "4/5"


def test_recorded_amd_run_renders_risk_report():
    at = AppTest.from_file(_APP, default_timeout=60).run()
    assert not at.exception

    # With no env configured, the public app defaults to committed real artifacts,
    # not simulated or external live API calls.
    assert any(
        getattr(selectbox, "value", "") == "Recorded AMD proof (W7900 ROCm)"
        for selectbox in at.selectbox
    )
    at.button[0].click().run()

    assert not at.exception
    text = "\n".join(_texts(at))
    assert "VERIFIED AMD RUN" in text
    assert "Risk Report" in text
    assert "Attack Success Rate" in text
    assert "Worst-Case Dossier" in text
    assert len(at.expander) >= 1
    metric_values = {metric.label: metric.value for metric in at.metric}
    assert metric_values["Attack Success Rate"] == "80%"
    assert metric_values["Failed cases"] == "4/5"


def test_real_leaderboard_artifact_is_visible():
    at = AppTest.from_file(_APP, default_timeout=60).run()
    assert not at.exception
    at.radio[0].set_value("Leaderboard").run()
    assert not at.exception
    text = "\n".join(_texts(at))
    assert "Real Gemma run" in text
    assert "gemma-3-27b-it" in text
    assert "gpt-oss-120b" in text


def test_fine_tune_proof_surface_is_present():
    at = AppTest.from_file(_APP, default_timeout=60).run()
    assert not at.exception
    at.radio[0].set_value("Fine-Tune Proof").run()
    assert not at.exception
    text = "\n".join(_texts(at))
    assert "Fine-Tune Proof" in text
    assert "Recorded ROCm result" in text
    assert "56-example validation split" in text
    metric_values = {metric.label: metric.value for metric in at.metric}
    assert metric_values["Valid JSON"] == "100.0%"
    assert metric_values["Pass/fail accuracy"] == "75.0%"
    assert metric_values["Macro-F1"] == "0.622"
    assert metric_values["Score MAE"] == "1.30"


def test_amd_proof_surface_is_present():
    at = AppTest.from_file(_APP, default_timeout=60).run()
    assert not at.exception
    at.radio[0].set_value("AMD Proof").run()
    assert not at.exception
    text = "\n".join(_texts(at))
    assert "AMD Proof" in text
    assert "MI300X" in text
    assert "vLLM + ROCm" in text
