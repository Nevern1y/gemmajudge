"""Integration smoke test for the Streamlit UI via the official AppTest harness.

Runs the app in-process (no browser, no network) in simulated mode, clicks Run, and
asserts the engine seam wired through to the report with no exceptions. This guards
the UI ↔ ``run_eval`` contract in CI.
"""

from streamlit.testing.v1 import AppTest

_APP = "app.py"


def test_app_loads_without_exception():
    at = AppTest.from_file(_APP, default_timeout=60).run()
    assert not at.exception


def test_simulated_run_renders_report():
    at = AppTest.from_file(_APP, default_timeout=60).run()
    # With no env configured, the simulated-demo toggle defaults on.
    assert any(t.value for t in at.toggle)
    at.button[0].click().run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert any("Attack Success Rate" in label for label in labels)
    # Drill-down expander per case + a loud SIMULATED banner.
    assert len(at.expander) >= 1
    assert any("SIMULATED" in w.value for w in at.warning)
