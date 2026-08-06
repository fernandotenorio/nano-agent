# ui/tui/widgets/approval.py
"""
Inline approval blocks.

A shell command or a proposed plan is a part of the conversation, so it is
asked as one: the question scrolls into the transcript like any other block,
answers with a keypress, and stays there afterwards showing what was decided.
No modal, nothing to dismiss, nothing lost from the history.

Both blocks fail closed. If the application goes away while a question is
still open, the safe answer is the one that gets recorded.
"""

from __future__ import annotations

import asyncio
from typing import Generic, Literal, TypeVar

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Markdown, Static

T = TypeVar("T")

PlanChoice = Literal["build", "plan", "reject"]


class ApprovalBlock(Vertical, Generic[T]):
    """A question in the transcript, awaited by the agent loop."""

    can_focus = True

    def __init__(self, safe_answer: T, classes: str) -> None:
        super().__init__(classes=f"block block-approval {classes}")
        self._safe_answer = safe_answer
        self._answer: asyncio.Future[T] = asyncio.get_running_loop().create_future()
        self._answered_by_user = False

    async def wait(self) -> T:
        return await self._answer

    @property
    def answered_by_user(self) -> bool:
        """False when the safe answer was recorded because nobody was there.

        Callers use this to know whether a follow-up question (why did you
        refuse?) has anyone to ask.
        """
        return self._answered_by_user

    def set_outcome(self, text: str) -> None:
        # The block may already be gone (shutdown, or a cleared transcript);
        # the decision itself is what matters, not the record of it.
        for hint in self.query(".approval-hint").results(Static):
            hint.update(text)

    def _settle(self, answer: T, outcome: str) -> None:
        if self._answer.done():
            return

        self._answered_by_user = True
        self._answer.set_result(answer)
        self.can_focus = False
        self.set_outcome(outcome)

    def on_unmount(self) -> None:
        # Nobody left to ask: record the answer that cannot do harm.
        if not self._answer.done():
            self._answer.set_result(self._safe_answer)


class ShellApprovalBlock(ApprovalBlock[bool]):
    """Asks whether one shell command may run."""

    BINDINGS = [
        Binding("y", "approve", "Allow", show=False),
        Binding("n", "deny", "Deny", show=False),
        Binding("escape", "deny", "Deny", show=False),
    ]

    def __init__(self, command: str, description: str | None = None) -> None:
        super().__init__(safe_answer=False, classes="approval-shell")
        self._command = command
        self._description = description

    def compose(self) -> ComposeResult:
        yield Static("Run this shell command?", markup=False, classes="approval-title")
        if self._description:
            yield Static(self._description, markup=False, classes="approval-description")
        yield Static(f"$ {self._command}", markup=False, classes="approval-command")
        yield Static("y  allow      n  deny", markup=False, classes="approval-hint")

    def action_approve(self) -> None:
        self._settle(True, "Allowed")

    def action_deny(self) -> None:
        self._settle(False, "Denied")


class PlanApprovalBlock(ApprovalBlock[PlanChoice]):
    """Presents a proposed plan and collects the user's decision."""

    BINDINGS = [
        Binding("1", "accept_build", "Accept and build", show=False),
        Binding("2", "accept_plan", "Accept, stay in plan mode", show=False),
        Binding("3", "reject", "Reject", show=False),
        Binding("escape", "reject", "Reject", show=False),
    ]

    def __init__(self, plan_summary: str) -> None:
        super().__init__(safe_answer="reject", classes="approval-plan")
        self._plan_summary = plan_summary

    def compose(self) -> ComposeResult:
        yield Static("Proposed plan", markup=False, classes="approval-title")
        yield Markdown(self._plan_summary, classes="approval-plan-body")
        yield Static(
            "1  accept and build      2  accept, keep planning      3  reject",
            markup=False,
            classes="approval-hint",
        )

    def action_accept_build(self) -> None:
        self._settle("build", "Accepted, switching to BUILD mode")

    def action_accept_plan(self) -> None:
        self._settle("plan", "Accepted, staying in PLAN mode")

    def action_reject(self) -> None:
        self._settle("reject", "Rejected")
