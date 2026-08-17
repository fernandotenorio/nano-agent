# ui/tui/ui.py
"""The `UI` implementation backed by the Textual application."""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager

from ui.base import (
    UI,
    PlanDecision,
    SessionInfo,
    SessionRunner,
    ShellDecision,
    ToolCallView,
    UsageInfo,
    UsageProvider,
    UsageReport,
)
from ui.null_ui import QuietUI
from ui.theme import DEFAULT_THEME, UITheme
from ui.tui.app import PrismaApp


class TextualUI(UI):
    """Full-screen front-end.

    Every method is a thin forward to the running application: the widgets
    decide how things look, this class only decides what the agent may say.
    """

    def __init__(self, theme: UITheme | None = None):
        self._theme = theme or DEFAULT_THEME
        self._app: PrismaApp | None = None
        self._usage_provider: UsageProvider | None = None

    @property
    def app(self) -> PrismaApp:
        if self._app is None:
            raise RuntimeError("The Textual UI is not running.")
        return self._app

    # --- Lifecycle ----------------------------------------------------------

    async def run(self, session: SessionRunner) -> None:
        self._app = PrismaApp(self._theme, session, self._usage_provider)
        try:
            await self._app.run_async()
        finally:
            self._app = None

    # --- Passive rendering -------------------------------------------------

    async def session_start(self, info: SessionInfo) -> None:
        await self.app.show_session(info)

    async def mode_changed(self, mode: str) -> None:
        self.app.show_mode(mode)
        await self.app.add_notice(f"Switched to {mode} mode")

    async def thinking(self, text: str, duration_s: float | None = None) -> None:
        await self.app.add_reasoning(text, duration_s)

    async def assistant_text(self, text: str) -> None:
        await self.app.add_assistant_message(text)

    def tool_status(self, summary: str) -> AbstractAsyncContextManager[None]:
        return self.app.spinner(summary)

    async def tool_result(self, call: ToolCallView) -> None:
        await self.app.add_tool_result(call)

    async def usage(self, info: UsageInfo) -> None:
        self.app.add_usage(info)

    async def show_usage(self, report: UsageReport) -> None:
        self.app.open_usage(report)

    async def notice(self, text: str) -> None:
        await self.app.add_notice(text)

    async def error(self, text: str) -> None:
        await self.app.add_error(text)

    # --- Interactive prompts ----------------------------------------------

    async def confirm_shell(self, command: str, description: str | None = None) -> ShellDecision:
        return await self.app.request_shell_approval(command, description)

    async def approve_plan(self, plan_summary: str) -> PlanDecision:
        return await self.app.request_plan_approval(plan_summary)

    async def read_user_input(self) -> str:
        return await self.app.request_user_input()

    # --- Composition --------------------------------------------------------

    def set_usage_provider(self, provider: UsageProvider) -> None:
        # Held until `run`, which is where the application that needs it is
        # built. Wiring happens during setup, before any of this is alive.
        self._usage_provider = provider

    def for_subagent(self) -> UI:
        return QuietUI(self)
