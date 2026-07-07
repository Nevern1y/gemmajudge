"""System-prompt loading for the Attacker and Judge.

Prompts live as ``.md`` files beside this module and are loaded via
:mod:`importlib.resources`, so they resolve correctly no matter what the current
working directory is (matters once the app is deployed on Dev Cloud / in a
container). One prompt per ``(role, failure_mode)`` pair.
"""

from __future__ import annotations

from enum import StrEnum
from importlib.resources import files

from gemmajudge.schemas import FailureMode


class PromptRole(StrEnum):
    ATTACKER = "attacker"
    JUDGE = "judge"


class PromptNotFoundError(FileNotFoundError):
    """No system prompt is bundled for the requested role + failure mode."""


def load_prompt(role: PromptRole, failure_mode: FailureMode) -> str:
    """Return the system prompt text for a role + failure mode.

    Filenames follow ``{role}_{failure_mode}.md`` (e.g. ``attacker_hallucination.md``).
    Raises :class:`PromptNotFoundError` if that pair isn't shipped yet — which is the
    honest signal that a P1/P2 mode (jailbreak/bias) hasn't been authored.
    """
    filename = f"{role.value}_{failure_mode.value}.md"
    resource = files(__package__).joinpath(filename)
    if not resource.is_file():
        raise PromptNotFoundError(
            f"No system prompt bundled for role={role.value!r} "
            f"failure_mode={failure_mode.value!r} (expected {filename})."
        )
    return resource.read_text(encoding="utf-8")
