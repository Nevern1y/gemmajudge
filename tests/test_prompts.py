"""Tests for prompt loading (packaged, CWD-independent)."""

import pytest

from gemmajudge.prompts import PromptNotFoundError, PromptRole, load_prompt
from gemmajudge.schemas import FailureMode


def test_attacker_hallucination_prompt_loads():
    text = load_prompt(PromptRole.ATTACKER, FailureMode.HALLUCINATION)
    assert "adversarial" in text.lower()
    assert "json" in text.lower()


def test_judge_hallucination_prompt_carries_rubric_and_pass_rule():
    text = load_prompt(PromptRole.JUDGE, FailureMode.HALLUCINATION)
    assert "passed = (score <= 2)" in text
    # rubric anchors present
    for anchor in ("1 —", "5 —"):
        assert anchor in text


def test_unbundled_mode_raises_clearly():
    with pytest.raises(PromptNotFoundError) as exc:
        load_prompt(PromptRole.ATTACKER, FailureMode.JAILBREAK)
    assert "jailbreak" in str(exc.value)
