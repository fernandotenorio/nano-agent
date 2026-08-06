# ui/tui/widgets/blocks.py
"""
Transcript blocks: the conversation as it scrolls past.

Every block wears its speaker's accent on both the icon/caption row and the
left border, so the eye can separate one turn from the next without reading a
word of it. Colors come from the stylesheet (fed by the theme); this module
only decides structure and content.
"""

from __future__ import annotations

from datetime import datetime

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Collapsible, Markdown, Static

from ui.base import SessionInfo, ToolCallView
from ui.theme import BlockTheme, ToolTheme
from ui.truncate import truncate_call, truncate_output


def clock(moment: datetime | None = None) -> str:
    """Renders a 12-hour timestamp such as '(5:11PM)'."""
    moment = moment or datetime.now()
    return f"({moment.strftime('%I:%M%p').lstrip('0')})"


class MessageBlock(Vertical):
    """A user or assistant turn: caption row, then the message body."""

    def __init__(
        self,
        style: BlockTheme,
        text: str,
        *,
        kind: str,
        markdown: bool = False,
        timestamp: datetime | None = None,
    ) -> None:
        super().__init__(classes=f"block block-{kind}")
        self._style = style
        self._text = text
        self._markdown = markdown
        self._timestamp = timestamp or datetime.now()

    def compose(self) -> ComposeResult:
        with Horizontal(classes="block-header"):
            yield Static(f"{self._style.icon} {self._style.caption}", markup=False, classes="block-caption")
            yield Static(clock(self._timestamp), markup=False, classes="block-time")

        if self._markdown:
            yield Markdown(self._text, classes="block-body")
        else:
            yield Static(self._text, markup=False, classes="block-body")


class ReasoningBlock(Collapsible):
    """A model reasoning block: how long it thought, and (on demand) about what."""

    def __init__(self, style: BlockTheme, text: str, duration_s: float | None = None) -> None:
        duration = f" ({duration_s:.1f}s)" if duration_s is not None else ""
        super().__init__(
            Static(text, markup=False, classes="block-body"),
            title=f"{style.icon} {style.caption}{duration}",
            collapsed=True,
            classes="block block-reasoning",
        )


class ToolBlock(Vertical):
    """One finished tool call.

    The summary line is always visible; the call and its output hide behind
    collapsed panes, because they matter only when something looks wrong.
    """

    def __init__(self, style: ToolTheme, call: ToolCallView) -> None:
        state = "failed" if call.is_error else "ok"
        super().__init__(classes=f"block block-tool tool-{state}")
        self._style = style
        self._call = call
        self._call_pane = truncate_call(call.name, call.args)
        self._output_pane = truncate_output(call.output)

    def compose(self) -> ComposeResult:
        icon = self._style.error_icon if self._call.is_error else self._style.success_icon

        with Horizontal(classes="tool-summary"):
            yield Static(icon, markup=False, classes="tool-icon")
            yield Static(f"[{self._call.name}]", markup=False, classes="tool-name")
            yield Static(self._call.summary, markup=False, classes="tool-text")

        yield Collapsible(
            Static(self._call_pane.text, markup=False, classes="tool-pane-body"),
            title=self._call_pane.label("Call"),
            collapsed=True,
            classes="tool-pane",
        )

        if self._output_pane.text:
            yield Collapsible(
                Static(self._output_pane.text, markup=False, classes="tool-pane-body"),
                title=self._output_pane.label("Output"),
                collapsed=True,
                classes="tool-pane",
            )


class NoticeBlock(Static):
    """A one-line aside: mode switches, warnings, errors."""

    def __init__(self, text: str, *, is_error: bool = False) -> None:
        classes = "notice notice-error" if is_error else "notice"
        super().__init__(text, markup=False, classes=classes)


class SessionBanner(Vertical):
    """Opening block: where the transcript lives, and anything degraded."""

    def __init__(self, info: SessionInfo) -> None:
        super().__init__(classes="session-banner")
        self._info = info

    def compose(self) -> ComposeResult:
        yield Static(f"Transcript: {self._info.transcript_path}", markup=False, classes="session-line")
        yield Static(
            "Enter sends  ·  Shift+Enter for a new line  ·  / for commands",
            markup=False,
            classes="session-line",
        )
        for warning in self._info.warnings:
            yield Static(f"! {warning}", markup=False, classes="session-line session-warning")
