# ui/tui/app.py
"""
The Textual application.

The agent session runs as a worker on this application's event loop, which is
what lets the whole UI contract be `await`ed: `read_user_input` waits on a
queue that the prompt fills, and the approval prompts wait on a future that a
keypress resolves. One loop, no threads, no polling.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, Literal

from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.css.query import NoMatches
from textual.widget import Widget

from ui.base import PlanDecision, SessionInfo, SessionRunner, ShellDecision, ToolCallView, UsageInfo
from ui.theme import UITheme, css_variables
from ui.tui.widgets import (
    FooterBar,
    HeaderBar,
    MessageBlock,
    NoticeBlock,
    PlanApprovalBlock,
    PromptArea,
    ReasoningBlock,
    SessionBanner,
    ShellApprovalBlock,
    SpinnerLine,
    ToolBlock,
)

InputMode = Literal["locked", "prompt", "reason"]

DENY_REASON_HINT = "Reason for denying (Enter to skip)"
REJECT_REASON_HINT = "Why are you rejecting this plan? (Enter to skip)"


class PrismaApp(App[None]):
    """Full-screen front-end: header, transcript, prompt, footer."""

    CSS_PATH = "app.tcss"

    def __init__(self, theme: UITheme, session: SessionRunner) -> None:
        # Assigned before App.__init__, which reads the CSS variables while
        # building the stylesheet.
        self.ui_theme = theme

        super().__init__()
        self._session = session
        self._input_queue: asyncio.Queue[str] = asyncio.Queue()
        self._input_mode: InputMode = "locked"
        self._spinner: SpinnerLine | None = None

    def get_css_variables(self) -> dict[str, str]:
        return {**super().get_css_variables(), **css_variables(self.ui_theme)}

    def compose(self) -> ComposeResult:
        # Plain vertical layout: the transcript takes every row the fixed
        # header, prompt, and footer do not.
        yield HeaderBar()
        yield VerticalScroll(id="transcript")
        yield PromptArea(id="prompt-area")
        yield FooterBar()

    def on_mount(self) -> None:
        self._set_input_mode("locked")
        self.run_worker(self._drive_session(), name="agent-session", exclusive=True)

    async def _drive_session(self) -> None:
        try:
            await self._session()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if not self.is_running:
                return

            # The session is over, but its last words are worth reading, so
            # the application stays up until the user closes it.
            logging.exception("Agent session crashed")
            await self.add_error(f"{type(e).__name__}: {e}")
            await self.add_notice("The session ended. Press Ctrl+C to close.")
            self._set_input_mode("locked")
            return

        self.exit()

    # --- Parts ---------------------------------------------------------------

    # Shutdown races with the session worker: cancelling it unwinds through
    # `finally` blocks that want to touch widgets which the closing screen has
    # already taken away. Missing widgets are therefore normal, not an error.

    @property
    def transcript(self) -> VerticalScroll | None:
        try:
            return self.query_one("#transcript", VerticalScroll)
        except NoMatches:
            return None

    @property
    def prompt(self) -> PromptArea | None:
        try:
            return self.query_one(PromptArea)
        except NoMatches:
            return None

    async def _append(self, widget: Widget) -> None:
        transcript = self.transcript
        if transcript is None:
            return

        await transcript.mount(widget)
        transcript.scroll_end(animate=False)

    # --- Rendering -----------------------------------------------------------

    async def show_session(self, info: SessionInfo) -> None:
        self.title = info.app_name
        self.query_one(HeaderBar).show_session(info)
        self.query_one(FooterBar).show_session(info)
        await self._append(SessionBanner(info))

    def show_mode(self, mode: str) -> None:
        self.query_one(HeaderBar).show_mode(mode)

    def add_usage(self, usage: UsageInfo) -> None:
        self.query_one(FooterBar).add_usage(usage)

    async def add_user_message(self, text: str) -> None:
        await self._append(MessageBlock(self.ui_theme.user, text, kind="user"))

    async def add_assistant_message(self, text: str) -> None:
        await self._append(
            MessageBlock(self.ui_theme.assistant, text, kind="assistant", markdown=True)
        )

    async def add_reasoning(self, text: str, duration_s: float | None = None) -> None:
        await self._append(ReasoningBlock(self.ui_theme.reasoning, text, duration_s))

    async def add_tool_result(self, call: ToolCallView) -> None:
        await self._append(ToolBlock(self.ui_theme.tool, call))

    async def add_notice(self, text: str) -> None:
        await self._append(NoticeBlock(text))

    async def add_error(self, text: str) -> None:
        await self._append(NoticeBlock(text, is_error=True))

    @asynccontextmanager
    async def spinner(self, summary: str) -> AsyncIterator[None]:
        if self._spinner is not None:
            # Something slower is already spinning; one moving thing is plenty.
            yield
            return

        line = SpinnerLine(summary)
        self._spinner = line
        await self._append(line)
        try:
            yield
        finally:
            self._spinner = None
            await line.remove()

    # --- Input ---------------------------------------------------------------

    def _set_input_mode(self, mode: InputMode, hint: str = "") -> None:
        self._input_mode = mode

        prompt = self.prompt
        if prompt is not None:
            prompt.set_mode(mode, hint)

    def _focus_prompt(self) -> None:
        prompt = self.prompt
        if prompt is not None:
            prompt.focus_input()

    async def request_user_input(self) -> str:
        self._set_input_mode("prompt")
        try:
            return await self._input_queue.get()
        finally:
            self._set_input_mode("locked")

    async def _request_reason(self, hint: str) -> str:
        self._set_input_mode("reason", hint)
        try:
            return (await self._input_queue.get()).strip()
        finally:
            self._set_input_mode("locked")

    async def on_prompt_area_submitted(self, event: PromptArea.Submitted) -> None:
        event.stop()

        if self._input_mode == "locked":
            return

        if self._input_mode == "prompt":
            if not event.text.strip():
                return
            await self.add_user_message(event.text)

        self._input_queue.put_nowait(event.text)

    # --- Decisions -----------------------------------------------------------

    async def request_shell_approval(self, command: str, description: str | None) -> ShellDecision:
        block = ShellApprovalBlock(command, description)
        await self._append(block)
        block.focus()

        approved = await block.wait()
        if approved:
            self._focus_prompt()
            return ShellDecision(approved=True)

        if not block.answered_by_user:
            return ShellDecision(approved=False)

        reason = await self._request_reason(DENY_REASON_HINT)
        if reason:
            block.set_outcome(f"Denied: {reason}")

        return ShellDecision(approved=False, deny_reason=reason)

    async def request_plan_approval(self, plan_summary: str) -> PlanDecision:
        block = PlanApprovalBlock(plan_summary)
        await self._append(block)
        block.focus()

        choice = await block.wait()
        if choice != "reject":
            self._focus_prompt()
            return PlanDecision(choice=choice)

        if not block.answered_by_user:
            return PlanDecision(choice="reject")

        reason = await self._request_reason(REJECT_REASON_HINT)
        if reason:
            block.set_outcome(f"Rejected: {reason}")

        return PlanDecision(choice="reject", reject_reason=reason)
