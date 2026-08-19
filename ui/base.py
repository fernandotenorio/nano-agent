# ui/base.py
"""
UI abstraction layer.

This module defines the *contract* between the agent backend and any
terminal (or other) front-end. It deliberately imports nothing from the
backend and nothing from any rendering library: backends talk to a `UI`
instance, and concrete implementations (textual, rich, headless, ...)
live in sibling modules.

Every method is async. The agent loop is async end to end, and a full-screen
front-end (Textual) owns the event loop, so a blocking call from inside the
loop would freeze the interface. Interactive methods return small decision
dataclasses; the UI never mutates agent state itself.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal

# The session coroutine factory handed to `UI.run`: the whole REPL, which the
# UI is free to run directly or to drive from inside its own event loop.
SessionRunner = Callable[[], Awaitable[None]]


@dataclass(frozen=True)
class SessionInfo:
    """Everything the UI needs to render the session header."""
    app_name: str
    model: str
    mode: str
    workspace: Path
    cwd: Path
    transcript_path: Path

    # The id `--resume` takes. Shown because the transcript now lives under a
    # hashed home directory, so the path alone is not something a user can be
    # expected to read an id out of.
    session_id: str = ""

    git_branch: str | None = None

    # Which service serves `model` ("ollama", "anthropic", ...). Derived from
    # the model string by `split_model`.
    provider: str = ""

    # Degraded capabilities worth surfacing at startup. These are for the user,
    # not the agent: they describe things only the user can fix, such as a
    # missing binary or a repository git refuses to read.
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ToolCallView:
    """Everything the UI may show about one finished tool call.

    `summary` is the one-line status every front-end renders. `args` and
    `output` back the collapsed detail panes: they are the raw call and the
    raw result, so a user debugging a failed tool can see exactly what the
    model sent and what came back. Truncation is the UI's business (see
    `ui.truncate`), not the caller's.
    """
    name: str
    args: Any
    summary: str
    output: Any = None
    is_error: bool = False


@dataclass(frozen=True)
class UsageInfo:
    """Token accounting for a single model response."""
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @classmethod
    def from_dict(cls, usage: dict[str, Any] | None) -> "UsageInfo":
        """Reads LiteLLM's usage dict, tolerating absent or odd values."""
        if not isinstance(usage, dict):
            return cls()

        def _count(key: str) -> int:
            value = usage.get(key)
            return value if isinstance(value, int) and value > 0 else 0

        return cls(
            input_tokens=_count("prompt_tokens"),
            output_tokens=_count("completion_tokens"),
        )


@dataclass(frozen=True)
class UsageRow:
    """One labelled bucket of tokens in the usage view."""
    label: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class UsageSection:
    """One table in the usage view.

    `note` carries a caveat about how the rows were derived, for the sections
    whose numbers cannot be read as plain sums.
    """
    title: str
    rows: tuple[UsageRow, ...] = ()
    note: str = ""


@dataclass(frozen=True)
class UsageReport:
    """The whole usage view: already aggregated, ordered, and ready to render."""
    sections: tuple[UsageSection, ...] = ()
    totals: UsageRow = UsageRow(label="Total")

    @property
    def is_empty(self) -> bool:
        return self.totals.calls == 0


# Builds a report reflecting usage *at the moment it is called*. Front-ends
# that offer usage through their own chrome hold one of these rather than a
# snapshot, which would be stale by the time the user asks.
UsageProvider = Callable[[], UsageReport]


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


# Model strings that name no provider. LiteLLM infers the provider from the
# model family, and so do we, purely so the footer can name it.
_PROVIDER_BY_PREFIX = (
    ("claude", "anthropic"),
    ("gpt", "openai"),
    ("o1", "openai"),
    ("o3", "openai"),
    ("gemini", "google"),
)


def split_model(model: str) -> tuple[str, str]:
    """Splits 'ollama/gemma3:12b' into ('ollama', 'gemma3:12b')."""
    provider, separator, name = model.partition("/")
    if separator:
        return provider, name

    for prefix, inferred in _PROVIDER_BY_PREFIX:
        if model.startswith(prefix):
            return inferred, model

    return "", model


class UI(ABC):
    """Abstract terminal UI. All agent output/input flows through here."""

    # --- Lifecycle ----------------------------------------------------------

    @abstractmethod
    async def run(self, session: SessionRunner) -> None:
        """Runs one whole session to completion.

        A plain terminal UI just awaits `session()`. A full-screen UI starts
        its own application and drives `session()` from within it.
        """

    # --- Passive rendering -------------------------------------------------

    @abstractmethod
    async def session_start(self, info: SessionInfo) -> None:
        """Renders the startup banner (mode, model, workspace, branch...)."""

    @abstractmethod
    async def mode_changed(self, mode: str) -> None:
        """Announces a PLAN/BUILD mode switch."""

    @abstractmethod
    async def thinking(self, text: str, duration_s: float | None = None) -> None:
        """Renders a collapsed summary of a model reasoning block."""

    @abstractmethod
    async def assistant_text(self, text: str) -> None:
        """Renders a final assistant text response (markdown-capable)."""

    @abstractmethod
    def tool_status(self, summary: str) -> AbstractAsyncContextManager[None]:
        """Async context manager showing an active spinner while work runs."""

    @abstractmethod
    async def tool_result(self, call: ToolCallView) -> None:
        """Renders a finished tool call."""

    @abstractmethod
    async def usage(self, info: UsageInfo) -> None:
        """Reports the token usage of one model response."""

    @abstractmethod
    async def show_usage(self, report: UsageReport) -> None:
        """Renders the session's token usage breakdown."""

    @abstractmethod
    async def notice(self, text: str) -> None:
        """Renders a low-priority informational message."""

    @abstractmethod
    async def error(self, text: str) -> None:
        """Renders an error message."""

    # --- Interactive prompts ----------------------------------------------

    @abstractmethod
    async def confirm_shell(self, command: str, description: str | None = None) -> ShellDecision:
        """Asks the user to approve a shell command. Must fail closed."""

    @abstractmethod
    async def approve_plan(self, plan_summary: str) -> PlanDecision:
        """Presents a proposed plan and captures the user's decision."""

    @abstractmethod
    async def read_user_input(self) -> str:
        """Reads the next user prompt (the REPL input line)."""

    # --- Composition --------------------------------------------------------

    def set_usage_provider(self, provider: UsageProvider) -> None:
        """Supplies a way to build a usage report on demand.

        Only front-ends that reach for usage on their own (a key binding, a
        button on the status bar) need this; one that merely renders what the
        session hands it can ignore the provider entirely.
        """

    @abstractmethod
    def for_subagent(self) -> "UI":
        """Returns the UI a sub-agent loop should use (typically quiet)."""
