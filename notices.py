# notices.py
"""
One-shot, per-session notices.

Some warnings are discovered by a tool but are really about the session: "git
is broken, so these listings are wider than they should be" is true of every ls,
Glob, and Grep call that follows. Appending it to all of them would burn tokens
to repeat something the model already read, so each notice is emitted once and
then remembered here.
"""

from __future__ import annotations

from typedefs import ToolResult


def with_note(result: ToolResult, note: str) -> ToolResult:
    """Appends a caveat to a tool result, leaving the ui_summary alone.

    Structured (non-string) payloads are returned untouched: there is no
    obvious place to put prose in them.
    """
    if not isinstance(result.content, str):
        return result

    return result.model_copy(update={"content": f"{result.content}\n\n{note}"})


class NoticeLog:
    """Tracks which one-shot notices have already been delivered."""

    def __init__(self) -> None:
        self._seen: set[str] = set()

    def once(self, key: str) -> bool:
        """Claims `key`, returning True only for the first caller.

        Intended as a guard: `if ctx.notices.once("git-degraded"): ...`
        """
        if key in self._seen:
            return False

        self._seen.add(key)
        return True

    def seen(self, key: str) -> bool:
        """Reports whether `key` was already claimed, without claiming it."""
        return key in self._seen
