# ui/tui/widgets/chrome.py
"""The fixed header and footer strips."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.css.query import NoMatches
from textual.widgets import Static

from ui.base import SessionInfo, UsageInfo, split_model

USAGE_LABEL = "Usage ^U"


def _shorten(path, root) -> str:
    """Shows a path relative to the workspace when it sits inside it."""
    try:
        relative = path.relative_to(root)
    except ValueError:
        return str(path)
    return f"./{relative}" if relative.parts else "."


class HeaderBar(Static):
    """Where the agent is working, and under which rules."""

    def __init__(self) -> None:
        super().__init__("", id="header", markup=False)
        self._info: SessionInfo | None = None
        self._mode = ""

    def show_session(self, info: SessionInfo) -> None:
        self._info = info
        self._mode = info.mode
        self._redraw()

    def show_mode(self, mode: str) -> None:
        self._mode = mode
        self._redraw()

    def _redraw(self) -> None:
        info = self._info
        if info is None:
            return

        parts = [info.app_name, self._mode, str(info.workspace)]
        if info.cwd != info.workspace:
            parts.append(_shorten(info.cwd, info.workspace))
        if info.git_branch:
            parts.append(info.git_branch)

        self.update("  ".join(parts))


class UsageButton(Static):
    """The way into the usage breakdown for anyone not reaching for Ctrl+U."""

    def __init__(self) -> None:
        super().__init__(USAGE_LABEL, id="footer-usage", markup=False)

    def on_click(self) -> None:
        self.app.action_show_usage()


class FooterBar(Horizontal):
    """Who is answering, what it has cost so far, and a way to see the detail.

    The running total stays here because it is the number worth glancing at
    between turns; everything behind it lives one keypress away.
    """

    def __init__(self) -> None:
        super().__init__(id="footer")
        self._provider = ""
        self._model = ""
        self._input_tokens = 0
        self._output_tokens = 0

    def compose(self) -> ComposeResult:
        yield Static("", id="footer-info", markup=False)
        yield UsageButton()

    def show_session(self, info: SessionInfo) -> None:
        # The provider already names the service, so the model is shown bare:
        # "ollama  gemma3:12b" rather than "ollama  ollama/gemma3:12b".
        _, self._model = split_model(info.model)
        self._provider = info.provider
        self._redraw()

    def add_usage(self, usage: UsageInfo) -> None:
        self._input_tokens += usage.input_tokens
        self._output_tokens += usage.output_tokens
        self._redraw()

    def _redraw(self) -> None:
        # The label is a child, so it can be absent while the bar is being
        # composed or torn down. Nothing here is worth an exception.
        try:
            label = self.query_one("#footer-info", Static)
        except NoMatches:
            return

        parts = [part for part in (self._provider, self._model) if part]

        total = self._input_tokens + self._output_tokens
        if total:
            parts.append(f"{total:,} tokens ({self._input_tokens:,} in / {self._output_tokens:,} out)")

        label.update("  ".join(parts))
