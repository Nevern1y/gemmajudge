"""Tests for the reproducible AMD proof runner."""

import asyncio
import json
import subprocess
import sys

import pytest

from gemmajudge.schemas import EvalResult, RunMetrics
from scripts import run_amd_proof


def _args(tmp_path, *extra):
    return run_amd_proof._build_parser().parse_args(
        [
            "--engine-model",
            "google/gemma-3-4b-it",
            "--target-model",
            "google/gemma-3-1b-it",
            "--backend-label",
            "AMD Instinct MI300X test",
            "--output",
            str(tmp_path / "eval_result.json"),
            *extra,
        ]
    )


def test_request_timeout_over_rule_limit_is_rejected():
    with pytest.raises(SystemExit):
        run_amd_proof._build_parser().parse_args(
            [
                "--engine-model",
                "engine",
                "--target-model",
                "target",
                "--backend-label",
                "AMD",
                "--output",
                "out.json",
                "--request-timeout",
                "31",
            ]
        )


def test_runner_can_be_invoked_directly_outside_repo(tmp_path):
    result = subprocess.run(
        [sys.executable, str(run_amd_proof.__file__), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Run GemmaJudge against local AMD vLLM endpoints" in result.stdout


def test_runner_saves_complete_result(monkeypatch, tmp_path):
    async def fake_run_eval(config, *, settings, include_consistency):
        assert settings.request_timeout_s == 25.0
        assert include_consistency is True
        return EvalResult(
            config=config,
            metrics=RunMetrics(
                wall_clock_seconds=2.5,
                n_cases=config.n_cases,
                model_id=settings.model_id,
                target_model_id=config.target_model_id,
            ),
        )

    monkeypatch.setattr(run_amd_proof, "run_eval", fake_run_eval)
    args = _args(tmp_path)
    result = asyncio.run(run_amd_proof._run(args))

    saved = json.loads(args.output.read_text(encoding="utf-8"))
    assert saved["config"]["target_model_id"] == "google/gemma-3-1b-it"
    assert saved["metrics"]["inference_backend"] == "AMD Instinct MI300X test"
    assert result.metrics.wall_clock_seconds == 2.5
