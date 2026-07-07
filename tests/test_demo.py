"""Tests for the CLI runner (offline path + config-error handling)."""

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


def test_real_mode_config_error_exits_two(monkeypatch, capsys):
    def _raise() -> None:
        raise ConfigError("Missing required configuration: MODEL_ID")

    monkeypatch.setattr(demo, "load_settings", _raise)
    rc = demo.main(["--n", "3"])  # no --offline -> real path
    assert rc == 2
    err = capsys.readouterr().err
    assert "Configuration error" in err
    assert "--offline" in err  # helpful tip is shown
