"""Preflight checks for a GemmaJudge ROCm fine-tune/eval host.

This script is intentionally lightweight: it imports optional training packages only
when they are installed, so it can also run on a local dev machine and explain why
that machine is not suitable for the AMD proof run.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from typing import Any

TRAINING_PACKAGES = ("torch", "transformers", "peft", "accelerate")
EVAL_PACKAGES = ("openai", "vllm")
OPTIONAL_PACKAGES = ("bitsandbytes", "bitsandbytes_rocm")


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def collect_environment() -> dict[str, Any]:
    packages = {
        name: module_available(name)
        for name in (*TRAINING_PACKAGES, *EVAL_PACKAGES, *OPTIONAL_PACKAGES)
    }
    report: dict[str, Any] = {
        "python": sys.version.split()[0],
        "packages": packages,
        "executables": {
            "rocm-smi": shutil.which("rocm-smi") is not None,
            "vllm": shutil.which("vllm") is not None,
        },
        "torch": None,
        "rocm_smi": None,
    }

    if packages["torch"]:
        report["torch"] = _torch_report()
    if report["executables"]["rocm-smi"]:
        report["rocm_smi"] = _run_rocm_smi()
    return report


def failure_reasons(report: dict[str, Any], *, require_gpu: bool) -> list[str]:
    reasons: list[str] = []
    packages = report["packages"]
    for package in TRAINING_PACKAGES:
        if not packages.get(package, False):
            reasons.append(f"missing training package: {package}")
    if not packages.get("openai", False):
        reasons.append("missing eval package: openai")
    if not packages.get("vllm", False):
        reasons.append("missing serving package: vllm")

    torch_report = report.get("torch") or {}
    if require_gpu:
        if not torch_report:
            reasons.append("torch is unavailable, so GPU visibility cannot be checked")
        elif torch_report.get("error"):
            reasons.append(f"torch import failed: {torch_report['error']}")
        else:
            if not torch_report.get("hip_version"):
                reasons.append("torch.version.hip is empty; install a ROCm PyTorch build")
            if not torch_report.get("device_available"):
                reasons.append("PyTorch cannot see an AMD GPU through torch.cuda")
            if int(torch_report.get("device_count") or 0) < 1:
                reasons.append("PyTorch reports zero visible GPU devices")
        if not report["executables"].get("rocm-smi", False):
            reasons.append("rocm-smi is not on PATH")
    return reasons


def _torch_report() -> dict[str, Any]:
    try:
        import torch

        device_count = torch.cuda.device_count()
        devices = []
        for idx in range(device_count):
            devices.append(
                {
                    "index": idx,
                    "name": torch.cuda.get_device_name(idx),
                    "capability": list(torch.cuda.get_device_capability(idx)),
                }
            )
        return {
            "version": torch.__version__,
            "hip_version": getattr(torch.version, "hip", None),
            "cuda_version": getattr(torch.version, "cuda", None),
            "device_available": torch.cuda.is_available(),
            "device_count": device_count,
            "devices": devices,
        }
    except Exception as exc:  # noqa: BLE001 - preflight should report any import/runtime issue
        return {"error": str(exc)[:500]}


def _run_rocm_smi() -> dict[str, Any]:
    try:
        completed = subprocess.run(
            ["rocm-smi"],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except Exception as exc:  # noqa: BLE001 - diagnostic only
        return {"error": str(exc)[:500]}
    return {
        "returncode": completed.returncode,
        "stdout_head": completed.stdout.splitlines()[:20],
        "stderr_head": completed.stderr.splitlines()[:20],
    }


def _print_text(report: dict[str, Any], reasons: list[str]) -> None:
    print(f"Python: {report['python']}")
    print("Packages:")
    for name, available in report["packages"].items():
        print(f"  {name}: {'yes' if available else 'no'}")
    print("Executables:")
    for name, available in report["executables"].items():
        print(f"  {name}: {'yes' if available else 'no'}")
    print("Torch:")
    print(json.dumps(report.get("torch"), indent=2))
    if report.get("rocm_smi") is not None:
        print("rocm-smi:")
        print(json.dumps(report["rocm_smi"], indent=2))
    if reasons:
        print("\nPreflight failed:")
        for reason in reasons:
            print(f"  - {reason}")
    else:
        print("\nPreflight passed.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check ROCm fine-tune/eval readiness.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Fail unless ROCm PyTorch can see at least one AMD GPU.",
    )
    args = parser.parse_args()

    report = collect_environment()
    reasons = failure_reasons(report, require_gpu=args.require_gpu)
    if args.json:
        print(json.dumps({"report": report, "failure_reasons": reasons}, indent=2))
    else:
        _print_text(report, reasons)
    return 1 if reasons else 0


if __name__ == "__main__":
    raise SystemExit(main())
