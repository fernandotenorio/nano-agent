# ui/tui/widgets/chrome.py
"""The fixed header and footer strips."""

from __future__ import annotations

from textual.widgets import Static

from ui.base import SessionInfo, UsageInfo, split_model


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


class FooterBar(Static):
    """Who is answering, and what it has cost so far."""

    def __init__(self) -> None:
        super().__init__("", id="footer", markup=False)
        self._provider = ""
        self._model = ""
        self._input_tokens = 0
        self._output_tokens = 0

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
        parts = [part for part in (self._provider, self._model) if part]

        total = self._input_tokens + self._output_tokens
        if total:
            parts.append(f"{total:,} tokens ({self._input_tokens:,} in / {self._output_tokens:,} out)")

        self.update("  ".join(parts))
