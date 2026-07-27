# ui/base.py
"""
UI abstraction layer.

This module defines the *contract* between the agent backend and any
terminal (or other) front-end. It deliberately imports nothing from the
backend and nothing from any rendering library: backends talk to a `UI`
instance, and concrete implementations (rich, plain text, headless, ...)
live in sibling modules.

Interactive methods return small decision dataclasses; the UI never
mutates agent state itself.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class SessionInfo:
    """Everything the UI needs to render the session header."""
    app_name: str
    model: str
    mode: str
    workspace: Path
    cwd: Path
    transcript_path: Path
    git_branch: str | None = None

    # Degraded capabilities worth surfacing at startup. These are for the user,
    # not the agent: they describe things only the user can fix, such as a
    # missing binary or a repository git refuses to read.
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ShellDecision:
    """User's answer to a shell-command confirmation prompt."""
    approved: bool
    deny_reason: str = ""


@dataclass(frozen=True)
class PlanDecision:
    """User's answer to a plan-approval prompt."""
    choice: Literal["build", "plan", "reject"]
    reject_reason: str = ""


class UI(ABC):
    """Abstract terminal UI. All agent output/input flows through here."""

    # --- Passive rendering -------------------------------------------------

    @abstractmethod
    def session_start(self, info: SessionInfo) -> None:
        """Renders the startup banner (mode, model, workspace, branch...)."""

    @abstractmethod
    def mode_changed(self, mode: str) -> None:
        """Announces a PLAN/BUILD mode switch."""

    @abstractmethod
    def thinking(self, text: str, duration_s: float | None = None) -> None:
        """Renders a collapsed summary of a model reasoning block."""

    @abstractmethod
    def assistant_text(self, text: str) -> None:
        """Renders a final assistant text response (markdown-capable)."""

    @abstractmethod
    def tool_status(self, summary: str) -> AbstractContextManager[None]:
        """Context manager showing an active spinner while a long action runs."""

    @abstractmethod
    def tool_result(self, summary: str, is_error: bool = False) -> None:
        """Renders a compact one-line tool completion status."""

    @abstractmethod
    def notice(self, text: str) -> None:
        """Renders a low-priority informational message."""

    @abstractmethod
    def error(self, text: str) -> None:
        """Renders an error message."""

    # --- Interactive prompts ----------------------------------------------

    @abstractmethod
    def confirm_shell(self, command: str, description: str | None = None) -> ShellDecision:
        """Asks the user to approve a shell command. Must fail closed."""

    @abstractmethod
    def approve_plan(self, plan_summary: str) -> PlanDecision:
        """Presents a proposed plan and captures the user's decision."""

    @abstractmethod
    def read_user_input(self) -> str:
        """Reads the next user prompt (the REPL input line)."""

    # --- Composition --------------------------------------------------------

    @abstractmethod
    def for_subagent(self) -> "UI":
        """Returns the UI a sub-agent loop should use (typically quiet)."""
