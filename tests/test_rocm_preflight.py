from scripts.check_rocm_training_env import failure_reasons


def test_preflight_reports_missing_required_packages():
    report = {
        "packages": {
            "torch": False,
            "transformers": False,
            "peft": False,
            "accelerate": False,
            "openai": True,
            "vllm": False,
        },
        "executables": {"rocm-smi": False},
        "torch": None,
    }

    reasons = failure_reasons(report, require_gpu=False)

    assert "missing training package: torch" in reasons
    assert "missing serving package: vllm" in reasons
    assert "rocm-smi is not on PATH" not in reasons


def test_preflight_requires_rocm_gpu_when_requested():
    report = {
        "packages": {
            "torch": True,
            "transformers": True,
            "peft": True,
            "accelerate": True,
            "openai": True,
            "vllm": True,
        },
        "executables": {"rocm-smi": False},
        "torch": {
            "hip_version": None,
            "device_available": False,
            "device_count": 0,
        },
    }

    reasons = failure_reasons(report, require_gpu=True)

    assert "torch.version.hip is empty; install a ROCm PyTorch build" in reasons
    assert "PyTorch cannot see an AMD GPU through torch.cuda" in reasons
    assert "rocm-smi is not on PATH" in reasons


def test_preflight_accepts_ready_rocm_host():
    report = {
        "packages": {
            "torch": True,
            "transformers": True,
            "peft": True,
            "accelerate": True,
            "openai": True,
            "vllm": True,
        },
        "executables": {"rocm-smi": True},
        "torch": {
            "hip_version": "6.2.0",
            "device_available": True,
            "device_count": 1,
        },
    }

    assert failure_reasons(report, require_gpu=True) == []
