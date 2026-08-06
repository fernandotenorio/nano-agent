# ui/truncate.py
"""
Shrinking tool calls and tool output to something a pane can hold.

These panes exist for debugging: when a tool misbehaves, the user wants to see
what the model actually sent and what actually came back. So truncation keeps
both ends of a value and says, in place, exactly how much it dropped. A Grep
that matched two thousand lines is far more useful showing its first and last
matches with a count between them than its first forty lines alone.

Pure text in, pure text out: no rendering library, no agent types.
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass
from typing import Any

MAX_VALUE_CHARS = 600
"""Cap on a single argument value inside a tool call."""

MAX_CALL_LINES = 60
"""Cap on the rendered call as a whole."""

MAX_LINE_CHARS = 500
"""Cap on one line of tool output (minified bundles are a single huge line)."""

HEAD_LINES = 40
TAIL_LINES = 10
"""How much of a long output survives, at each end."""

MAX_OUTPUT_CHARS = 20_000
"""Last-resort cap, after the line-based rules have had their say."""


@dataclass(frozen=True)
class TruncatedText:
    """Text ready to display, plus what it took to get there."""
    text: str
    truncated: bool = False
    detail: str = ""

    def label(self, title: str) -> str:
        """Renders a pane title such as 'Output (truncated, 1,204 lines)'."""
        if not self.truncated:
            return title
        return f"{title} (truncated, {self.detail})" if self.detail else f"{title} (truncated)"


def as_text(content: Any) -> str:
    """Flattens a tool payload into plain text.

    Accepts the shapes a tool result can take: a string, a list of content
    blocks carrying `.text`, or anything else worth a `str()`.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            item if isinstance(item, str) else str(getattr(item, "text", item))
            for item in content
        )
    return str(content)


def _elide(text: str, limit: int) -> tuple[str, bool]:
    """Keeps both ends of an over-long value, counting what went missing."""
    if len(text) <= limit:
        return text, False

    head = (limit * 2) // 3
    tail = limit - head
    omitted = len(text) - head - tail
    return f"{text[:head]}... {omitted:,} characters omitted ...{text[-tail:]}", True


def _cap_lines(text: str, head: int, tail: int) -> tuple[str, bool]:
    """Keeps the first `head` and last `tail` lines of a long block."""
    lines = text.splitlines()
    if len(lines) <= head + tail + 1:
        return text, False

    omitted = len(lines) - head - tail
    kept = [*lines[:head], f"... {omitted:,} lines omitted ...", *lines[-tail:]]
    return "\n".join(kept), True


class _Eliding:
    """Walks a JSON-ish structure, shortening every string it finds."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.truncated = False

    def walk(self, value: Any) -> Any:
        if isinstance(value, str):
            elided, was_truncated = _elide(value, self.limit)
            self.truncated = self.truncated or was_truncated
            return elided
        if isinstance(value, dict):
            return {key: self.walk(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.walk(item) for item in value]
        return value


def _render_value(value: Any, eliding: _Eliding) -> str:
    """Renders one argument, indented onto its own lines when multi-line."""
    shortened = eliding.walk(value)

    if isinstance(shortened, str):
        rendered = shortened if "\n" in shortened else json.dumps(shortened, ensure_ascii=False)
    else:
        try:
            rendered = json.dumps(shortened, indent=2, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            rendered = repr(shortened)

    if "\n" in rendered:
        return "\n" + textwrap.indent(rendered, "    ")
    return f" {rendered}"


def truncate_call(name: str, args: Any) -> TruncatedText:
    """Renders a tool invocation as readable, bounded text."""
    eliding = _Eliding(MAX_VALUE_CHARS)

    if isinstance(args, dict):
        if not args:
            body = [f"{name}()"]
        else:
            body = [f"{name}("]
            body += [f"  {key} ={_render_value(value, eliding)}" for key, value in args.items()]
            body.append(")")
        text = "\n".join(body)
    else:
        text = f"{name}({_render_value(args, eliding).strip()})"

    total_lines = len(text.splitlines())
    text, lines_dropped = _cap_lines(text, MAX_CALL_LINES - TAIL_LINES, TAIL_LINES)

    truncated = eliding.truncated or lines_dropped
    detail = f"{total_lines:,} lines" if lines_dropped else ""
    return TruncatedText(text=text, truncated=truncated, detail=detail)


def truncate_output(content: Any) -> TruncatedText:
    """Renders a tool result as readable, bounded text."""
    text = as_text(content).strip()
    if not text:
        return TruncatedText(text="")

    lines = text.splitlines()
    total_lines = len(lines)
    truncated = False

    capped_lines = []
    for line in lines:
        capped, was_capped = _elide(line, MAX_LINE_CHARS)
        truncated = truncated or was_capped
        capped_lines.append(capped)

    text, lines_dropped = _cap_lines("\n".join(capped_lines), HEAD_LINES, TAIL_LINES)
    truncated = truncated or lines_dropped

    # Whatever survived the line rules still has to fit in a pane.
    if len(text) > MAX_OUTPUT_CHARS:
        text = text[:MAX_OUTPUT_CHARS] + "\n... truncated ..."
        truncated = True

    detail = f"{total_lines:,} lines" if truncated else ""
    return TruncatedText(text=text, truncated=truncated, detail=detail)
