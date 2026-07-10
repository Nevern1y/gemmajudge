"""Tests for the CLI runner (offline path + config-error handling)."""

import json

from gemmajudge import demo
from gemmajudge.config import ConfigError


def test_offline_run_exits_zero(capsys):
    rc = demo.main(["--offline", "--n", "4"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "SIMULATED RUN" in out
    assert "Attack Success Rate" in out
    assert "Worst case (drill-down)" in out


def test_offline_no_consistency_flag(capsys):
    rc = demo.main(["--offline", "--n", "3", "--no-consistency"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "self-consistency" not in out.lower()


def test_regression_gate_exits_one_when_threshold_is_exceeded(capsys):
    rc = demo.main(["--offline", "--n", "4", "--max-asr", "0.5"])
    assert rc == 1
    out = capsys.readouterr().out
    assert "REGRESSION GATE: FAILED" in out
    assert "ASR 1.000 exceeds maximum 0.500" in out


def test_regression_gate_passes_at_threshold(capsys):
    rc = demo.main(
        ["--offline", "--n", "4", "--max-asr", "1", "--max-failed-cases", "4"]
    )
    assert rc == 0
    assert "REGRESSION GATE: PASSED" in capsys.readouterr().out


def test_json_summary_is_machine_readable(capsys):
    rc = demo.main(["--offline", "--n", "3", "--max-failed-cases", "0", "--json"])
    assert rc == 1
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "failed"
    assert summary["simulated"] is True
    assert summary["attack_success_rate"] == 1.0
    assert summary["thresholds"]["max_failed_cases"] == 0
    assert summary["failures"]


def test_real_mode_config_error_exits_two(monkeypatch, capsys):
    def _raise() -> None:
        raise ConfigError("Missing required configuration: MODEL_ID")

    monkeypatch.setattr(demo, "load_settings", _raise)
    rc = demo.main(["--n", "3"])  # no --offline -> real path
    assert rc == 2
    err = capsys.readouterr().err
    assert "Configuration error" in err
    assert "--offline" in err  # helpful tip is shown
