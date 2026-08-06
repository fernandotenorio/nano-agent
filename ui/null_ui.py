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

from contextlib import AbstractAsyncContextManager, nullcontext

from ui.base import (
    UI,
    PlanDecision,
    SessionInfo,
    SessionRunner,
    ShellDecision,
    ToolCallView,
    UsageInfo,
)


class NullUI(UI):
    """A UI that renders nothing and denies all interactive requests."""

    async def run(self, session: SessionRunner) -> None:
        await session()

    async def session_start(self, info: SessionInfo) -> None:
        pass

    async def mode_changed(self, mode: str) -> None:
        pass

    async def thinking(self, text: str, duration_s: float | None = None) -> None:
        pass

    async def assistant_text(self, text: str) -> None:
        pass

    def tool_status(self, summary: str) -> AbstractAsyncContextManager[None]:
        # nullcontext supports the async protocol since Python 3.10.
        return nullcontext()

    async def tool_result(self, call: ToolCallView) -> None:
        pass

    async def usage(self, info: UsageInfo) -> None:
        pass

    async def notice(self, text: str) -> None:
        pass

    async def error(self, text: str) -> None:
        pass

    async def confirm_shell(self, command: str, description: str | None = None) -> ShellDecision:
        # Fail closed: without a user to ask, the command must not run.
        return ShellDecision(approved=False)

    async def approve_plan(self, plan_summary: str) -> PlanDecision:
        return PlanDecision(choice="reject", reject_reason="No interactive UI available.")

    async def read_user_input(self) -> str:
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

    async def confirm_shell(self, command: str, description: str | None = None) -> ShellDecision:
        return await self._parent.confirm_shell(command, description)

    async def approve_plan(self, plan_summary: str) -> PlanDecision:
        return await self._parent.approve_plan(plan_summary)
