# ui/null_ui.py
"""
Non-rendering UI implementations.

- NullUI: fully silent, fails closed on interactive prompts. The default
  for tests and non-interactive execution.
- QuietUI: silent rendering, but interactive prompts are delegated to a
  parent UI. Used inside sub-agent loops: their internal tool chatter is
  hidden, yet safety gates (shell confirmation) still reach the user.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext

from ui.base import UI, PlanDecision, SessionInfo, ShellDecision


class NullUI(UI):
    """A UI that renders nothing and denies all interactive requests."""

    def session_start(self, info: SessionInfo) -> None:
        pass

    def mode_changed(self, mode: str) -> None:
        pass

    def thinking(self, text: str, duration_s: float | None = None) -> None:
        pass

    def assistant_text(self, text: str) -> None:
        pass

    def tool_status(self, summary: str) -> AbstractContextManager[None]:
        return nullcontext()

    def tool_result(self, summary: str, is_error: bool = False) -> None:
        pass

    def notice(self, text: str) -> None:
        pass

    def error(self, text: str) -> None:
        pass

    def confirm_shell(self, command: str, description: str | None = None) -> ShellDecision:
        # Fail closed: without a user to ask, the command must not run.
        return ShellDecision(approved=False)

    def approve_plan(self, plan_summary: str) -> PlanDecision:
        return PlanDecision(choice="reject", reject_reason="No interactive UI available.")

    def read_user_input(self) -> str:
        raise EOFError("No interactive UI available.")

    def for_subagent(self) -> UI:
        return self


class QuietUI(NullUI):
    """Silent renderer that forwards interactive prompts to a parent UI.

    The parent is responsible for pausing any active spinner before
    prompting (see RichUI.confirm_shell / approve_plan).
    """

    def __init__(self, parent: UI):
        self._parent = parent

    def confirm_shell(self, command: str, description: str | None = None) -> ShellDecision:
        return self._parent.confirm_shell(command, description)

    def approve_plan(self, plan_summary: str) -> PlanDecision:
        return self._parent.approve_plan(plan_summary)
