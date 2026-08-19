# ui/rich_ui.py
"""
Rich-powered scrolling terminal UI: the fallback front-end (`--ui rich`),
useful wherever a full-screen application is a poor fit (redirected output,
CI, a terminal Textual cannot drive).

The ONLY module in the project that imports `rich`.

Rendering is synchronous and fast, so it happens inline. Prompts block on
stdin, so they are pushed onto a worker thread: blocking the event loop would
stall the agent's own subprocesses and network calls.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.prompt import Prompt
from rich.status import Status
from rich.table import Table
from rich.text import Text

from ui.base import (
    UI,
    PlanDecision,
    SessionInfo,
    SessionRunner,
    ShellDecision,
    ToolCallView,
    UsageInfo,
    UsageReport,
    UsageSection,
)
from ui.null_ui import QuietUI

USAGE_EMPTY_MESSAGE = "No model usage recorded yet."


class _ElapsedText:
    """Live-refreshing spinner label: 'summary (12s)'."""

    def __init__(self, summary: str):
        self._summary = summary
        self._start = time.monotonic()

    def __rich__(self) -> Text:
        elapsed = time.monotonic() - self._start
        return Text(f"{self._summary} ({elapsed:.0f}s)", style="dim")


class RichUI(UI):
    def __init__(self, console: Console | None = None):
        self._console = console or Console()
        # Only one rich live display may be active at a time; we track it so
        # interactive prompts can pause/resume it and nested statuses no-op.
        self._active_status: Status | None = None

        # Legacy Windows consoles / redirected streams may use encodings
        # (e.g. cp1252) that cannot represent our glyphs. Degrade to ASCII.
        self._unicode = self._can_encode("\N{CHECK MARK}\N{BALLOT X}\N{THOUGHT BALLOON}\N{BOX DRAWINGS LIGHT HORIZONTAL}")
        self._check = "\N{CHECK MARK}" if self._unicode else "+"
        self._cross = "\N{BALLOT X}" if self._unicode else "x"
        self._thought = "\N{THOUGHT BALLOON}" if self._unicode else "~"
        self._rule_char = "\N{BOX DRAWINGS LIGHT HORIZONTAL}" if self._unicode else "-"
        self._box = box.ROUNDED if self._unicode else box.ASCII
        self._spinner = "dots" if self._unicode else "line"

    def _can_encode(self, sample: str) -> bool:
        encoding = getattr(self._console.file, "encoding", None) or "ascii"
        try:
            sample.encode(encoding)
            return True
        except (UnicodeEncodeError, LookupError):
            return False

    # --- Lifecycle ----------------------------------------------------------

    async def run(self, session: SessionRunner) -> None:
        await session()

    # --- Passive rendering -------------------------------------------------

    async def session_start(self, info: SessionInfo) -> None:
        c = self._console
        c.print()
        c.print(
            f"[bold]{escape(info.app_name)}[/bold] "
            f"[bold magenta]\\[{escape(info.mode)} mode][/bold magenta] "
            f"[dim]({escape(info.model)})[/dim]"
        )
        branch = f" [dim]({escape(info.git_branch)})[/dim]" if info.git_branch else ""
        c.print(f"[dim]Workspace:[/dim] {escape(str(info.workspace))}{branch}")
        if info.cwd != info.workspace:
            c.print(f"[dim]Directory:[/dim] {escape(str(info.cwd))}")
        if info.session_id:
            c.print(f"[dim]Session:[/dim] {escape(info.session_id)}")
        c.print(f"[dim]Transcript: {escape(str(info.transcript_path))}[/dim]")
        c.print("[dim]Type '/quit' to exit, '/plan' or '/build' to switch modes.[/dim]")

        for warning in info.warnings:
            c.print(f"[yellow]![/yellow] [dim]{escape(warning)}[/dim]")

        c.rule(characters=self._rule_char, style="dim")

    async def mode_changed(self, mode: str) -> None:
        self._console.print(f"[bold magenta]Switched to {escape(mode)} mode[/bold magenta]")

    async def thinking(self, text: str, duration_s: float | None = None) -> None:
        suffix = f" for {duration_s:.1f}s" if duration_s is not None else "..."
        self._console.print(f"[dim italic]{self._thought} Thought{suffix}[/dim italic]")

    async def assistant_text(self, text: str) -> None:
        self._console.print()
        self._console.print(Markdown(text))

    @asynccontextmanager
    async def tool_status(self, summary: str) -> AsyncIterator[None]:
        if self._active_status is not None:
            # A live display is already running (e.g. a sub-agent spinner):
            # rich forbids nesting, so inner statuses become no-ops.
            yield
            return

        status = self._console.status(_ElapsedText(summary), spinner=self._spinner)
        self._active_status = status
        status.start()
        try:
            yield
        finally:
            status.stop()
            self._active_status = None

    async def tool_result(self, call: ToolCallView) -> None:
        if call.is_error:
            self._console.print(f"  [red]{self._cross}[/red] [red]{escape(call.summary)}[/red]")
        else:
            self._console.print(f"  [green]{self._check}[/green] [dim]{escape(call.summary)}[/dim]")

    async def usage(self, info: UsageInfo) -> None:
        # A scrolling log has no status bar to park a running total in, and a
        # token count after every response would drown out the conversation.
        pass

    async def show_usage(self, report: UsageReport) -> None:
        self._pause_status()
        try:
            self._console.print()

            if report.is_empty:
                self._console.print(f"[dim]{escape(USAGE_EMPTY_MESSAGE)}[/dim]")
                return

            totals = report.totals
            self._console.print(
                f"[bold]Token usage[/bold] [dim]{totals.total_tokens:,} tokens across "
                f"{totals.calls:,} model call{'' if totals.calls == 1 else 's'}[/dim]"
            )

            for section in report.sections:
                self._console.print(self._usage_table(section))
                if section.note:
                    self._console.print(f"[dim]{escape(section.note)}[/dim]")
        finally:
            self._resume_status()

    def _usage_table(self, section: UsageSection) -> Table:
        table = Table(box=self._box, expand=False, title_justify="left")
        table.add_column(section.title, style="bold")

        for name in ("Input", "Output", "Cached", "Total", "Calls"):
            table.add_column(name, justify="right")

        for row in section.rows:
            table.add_row(
                row.label,
                f"{row.input_tokens:,}",
                f"{row.output_tokens:,}",
                f"{row.cached_tokens:,}",
                f"{row.total_tokens:,}",
                f"{row.calls:,}",
            )

        return table

    async def notice(self, text: str) -> None:
        self._console.print(f"[dim]{escape(text)}[/dim]")

    async def error(self, text: str) -> None:
        self._console.print(f"[bold red]{escape(text)}[/bold red]")

    # --- Interactive prompts ----------------------------------------------

    async def confirm_shell(self, command: str, description: str | None = None) -> ShellDecision:
        return await asyncio.to_thread(self._confirm_shell_blocking, command, description)

    def _confirm_shell_blocking(self, command: str, description: str | None) -> ShellDecision:
        self._pause_status()
        try:
            body = Text()
            if description:
                body.append(f"{description}\n\n", style="italic")
            body.append(f"$ {command}", style="bold")
            self._console.print()
            self._console.print(Panel(
                body,
                title="Shell Command Confirmation",
                border_style="yellow",
                box=self._box,
                expand=False,
            ))

            try:
                answer = self._console.input("Allow this command? [bold]\\[y/N][/bold]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return ShellDecision(approved=False)  # Fail closed

            if answer in ("y", "yes"):
                return ShellDecision(approved=True)

            if answer not in ("", "n", "no"):
                # Anything else typed is treated as a denial with feedback
                return ShellDecision(approved=False, deny_reason=answer)

            try:
                reason = self._console.input("Optional reason for denying (Enter to skip): ").strip()
            except (EOFError, KeyboardInterrupt):
                reason = ""
            return ShellDecision(approved=False, deny_reason=reason)
        finally:
            self._resume_status()

    async def approve_plan(self, plan_summary: str) -> PlanDecision:
        return await asyncio.to_thread(self._approve_plan_blocking, plan_summary)

    def _approve_plan_blocking(self, plan_summary: str) -> PlanDecision:
        self._pause_status()
        try:
            self._console.print()
            self._console.print(Panel(
                Markdown(plan_summary),
                title="Proposed Plan",
                border_style="cyan",
                box=self._box,
            ))
            self._console.print("  [bold]1.[/bold] Accept plan and switch to BUILD mode")
            self._console.print("  [bold]2.[/bold] Accept plan but stay in PLAN mode (refine further)")
            self._console.print("  [bold]3.[/bold] Reject plan with a message")

            try:
                choice = Prompt.ask(
                    "Select an option",
                    choices=["1", "2", "3"],
                    console=self._console,
                )
            except (EOFError, KeyboardInterrupt):
                return PlanDecision(choice="reject")

            if choice == "1":
                return PlanDecision(choice="build")
            if choice == "2":
                return PlanDecision(choice="plan")

            try:
                reason = self._console.input("Enter rejection reason: ").strip()
            except (EOFError, KeyboardInterrupt):
                reason = ""
            return PlanDecision(choice="reject", reject_reason=reason)
        finally:
            self._resume_status()

    async def read_user_input(self) -> str:
        return await asyncio.to_thread(self._console.input, "\n[bold cyan]>[/bold cyan] ")

    # --- Composition --------------------------------------------------------

    def for_subagent(self) -> UI:
        return QuietUI(self)

    # --- Internals -----------------------------------------------------------

    def _pause_status(self) -> None:
        if self._active_status is not None:
            self._active_status.stop()

    def _resume_status(self) -> None:
        if self._active_status is not None:
            self._active_status.start()
