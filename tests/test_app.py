"""Integration smoke tests for the Streamlit Mission Control UI.

The app runs in-process with Streamlit's AppTest harness. Tests stay offline:
with no env configured, the UI defaults to the simulated backend for live runs
and renders the committed real-Gemma leaderboard artifact from docs/real_runs/.
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


def test_simulated_run_renders_risk_report_and_warning():
    at = AppTest.from_file(_APP, default_timeout=60).run()
    assert not at.exception

    # With no env configured, simulated mode is enabled and the first button is
    # the primary live-run action in the Mission Control console.
    assert any(getattr(toggle, "value", False) for toggle in at.toggle)
    at.button[0].click().run()

    assert not at.exception
    text = "\n".join(_texts(at))
    assert "SIMULATED" in text
    assert "Risk Report" in text
    assert "Attack Success Rate" in text
    assert "Worst-Case Dossier" in text
    assert len(at.expander) >= 1


def test_real_leaderboard_artifact_is_visible():
    at = AppTest.from_file(_APP, default_timeout=60).run()
    assert not at.exception
    text = "\n".join(_texts(at))
    assert "Real Gemma run" in text
    assert "gemma-3-27b-it" in text
    assert "gpt-oss-120b" in text


def test_amd_proof_surface_is_present():
    at = AppTest.from_file(_APP, default_timeout=60).run()
    assert not at.exception
    text = "\n".join(_texts(at))
    assert "AMD Proof" in text
    assert "MI300X" in text
    assert "vLLM + ROCm" in text
