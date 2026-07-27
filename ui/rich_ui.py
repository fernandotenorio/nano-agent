# ui/rich_ui.py
"""
Rich-powered terminal UI. The ONLY module in the project that imports `rich`.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.prompt import Prompt
from rich.status import Status
from rich.text import Text

from ui.base import UI, PlanDecision, SessionInfo, ShellDecision
from ui.null_ui import QuietUI


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

    # --- Passive rendering -------------------------------------------------

    def session_start(self, info: SessionInfo) -> None:
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
        c.print(f"[dim]Transcript: {escape(str(info.transcript_path))}[/dim]")
        c.print("[dim]Type '/quit' to exit, '/plan' or '/build' to switch modes.[/dim]")

        for warning in info.warnings:
            c.print(f"[yellow]![/yellow] [dim]{escape(warning)}[/dim]")

        c.rule(characters=self._rule_char, style="dim")

    def mode_changed(self, mode: str) -> None:
        self._console.print(f"[bold magenta]Switched to {escape(mode)} mode[/bold magenta]")

    def thinking(self, text: str, duration_s: float | None = None) -> None:
        suffix = f" for {duration_s:.1f}s" if duration_s is not None else "..."
        self._console.print(f"[dim italic]{self._thought} Thought{suffix}[/dim italic]")

    def assistant_text(self, text: str) -> None:
        self._console.print()
        self._console.print(Markdown(text))

    @contextmanager
    def tool_status(self, summary: str) -> Iterator[None]:
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

    def tool_result(self, summary: str, is_error: bool = False) -> None:
        if is_error:
            self._console.print(f"  [red]{self._cross}[/red] [red]{escape(summary)}[/red]")
        else:
            self._console.print(f"  [green]{self._check}[/green] [dim]{escape(summary)}[/dim]")

    def notice(self, text: str) -> None:
        self._console.print(f"[dim]{escape(text)}[/dim]")

    def error(self, text: str) -> None:
        self._console.print(f"[bold red]{escape(text)}[/bold red]")

    # --- Interactive prompts ----------------------------------------------

    def confirm_shell(self, command: str, description: str | None = None) -> ShellDecision:
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

    def approve_plan(self, plan_summary: str) -> PlanDecision:
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

    def read_user_input(self) -> str:
        return self._console.input("\n[bold cyan]>[/bold cyan] ")

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
