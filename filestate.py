# filestate.py
"""
Per-agent file freshness tracking.

Replaces the old module-level `known_content_files` / `stale_content_files`
globals in tools/filesystem.py. Each agent loop owns one FileStateTracker
(carried on its InvocationContext), so a sub-agent can never write a file
it did not Read itself.

Design: there is no background watcher. Staleness is *computed on demand*
at the only two moments it is ever consumed:

  1. The write gate (Write/Edit/MultiEdit) calls `status(path)`.
  2. The user-prompt hook (`file_changes_hook`) calls `collect_changes()`
     and injects <system-reminder> diffs for externally modified files.

Each tracked file stores its content lines (small files only) plus a stat
signature (st_mtime_ns, st_size). A checkpoint stats the file; if the
signature changed, it falls back to a content comparison to rule out
false alarms (touch, identical rewrite). Agent-initiated writes re-record
the signature immediately, so the agent's own modifications never appear
stale.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING, Literal

from typedefs import TextMessageContent

if TYPE_CHECKING:
    from hooks import UserPromptEvent
    from sessioncontext import InvocationContext

MAX_FILE_BYTES: int = 256 * 1024
"""Files at or above this size are tracked by stat signature only
(lines=None): the gate still works, but prompt-time diffs are skipped."""

MAX_DIFF_LINES: int = 200
"""Cap on the size of a single injected diff reminder."""

FreshnessStatus = Literal["unknown", "fresh", "stale"]


@dataclass
class FileRecord:
    """What we know about a file at the time the transcript last saw it."""
    lines: list[str] | None  # None => file too large to cache content
    mtime_ns: int
    size: int


@dataclass
class FileStateTracker:
    """Tracks which files the current agent has fresh knowledge of."""

    known: dict[Path, FileRecord] = field(default_factory=dict)

    def record(self, path: Path, lines: list[str] | None) -> None:
        """Records the current on-disk state of `path` as known.

        Called right after a successful Read (with the lines just read) or
        a successful Write/Edit/MultiEdit (with the lines just written).
        """
        try:
            st = path.stat()
        except OSError:
            # File vanished between the operation and this call; treat as unknown.
            self.known.pop(path, None)
            return
        self.known[path] = FileRecord(lines=lines, mtime_ns=st.st_mtime_ns, size=st.st_size)

    def forget(self, path: Path) -> None:
        self.known.pop(path, None)

    def status(self, path: Path) -> FreshnessStatus:
        """Gate check: is our knowledge of `path` current?

        - "unknown": never read in this agent session -> read-before-write error.
        - "fresh":   disk matches what the transcript knows -> writing is safe.
        - "stale":   modified externally since last seen -> must re-Read first.
        """
        rec = self.known.get(path)
        if rec is None:
            return "unknown"

        try:
            st = path.stat()
        except OSError:
            # Deleted or unreadable since we last saw it.
            return "stale"

        if st.st_mtime_ns == rec.mtime_ns and st.st_size == rec.size:
            return "fresh"

        # Signature changed. Fall back to a content comparison so that a
        # `touch` or an identical rewrite doesn't force a pointless re-Read.
        if rec.lines is None:
            return "stale"  # Too large to have cached content; can't verify.

        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            return "stale"

        if text.splitlines() == rec.lines:
            # Same content, new timestamp: refresh the signature and allow.
            self.known[path] = FileRecord(lines=rec.lines, mtime_ns=st.st_mtime_ns, size=st.st_size)
            return "fresh"

        return "stale"

    def collect_changes(self) -> list[tuple[Path, list[str], list[str]]]:
        """Finds externally modified files and refreshes their records.

        Returns (path, old_lines, new_lines) for every tracked file whose
        content actually changed. Records are refreshed *per item* (never a
        blanket clear), so nothing is lost if a file is unreadable mid-scan:

        - Deleted/unreadable files are forgotten entirely, forcing a fresh
          Read before any future write (no reminder is emitted, matching
          the original mini-agent behavior).
        - Files without cached content (too large) get their signature
          refreshed silently; there is nothing to diff.
        """
        changes: list[tuple[Path, list[str], list[str]]] = []

        for path, rec in list(self.known.items()):
            try:
                st = path.stat()
            except OSError:
                self.known.pop(path, None)
                continue

            if st.st_mtime_ns == rec.mtime_ns and st.st_size == rec.size:
                continue

            try:
                text = path.read_text(encoding="utf-8")
            except Exception:
                self.known.pop(path, None)
                continue

            new_lines = text.splitlines() if st.st_size < MAX_FILE_BYTES else None
            self.known[path] = FileRecord(lines=new_lines, mtime_ns=st.st_mtime_ns, size=st.st_size)

            if rec.lines is None or new_lines is None or rec.lines == new_lines:
                continue  # Nothing to report (no cached content, or no real change).

            changes.append((path, rec.lines, new_lines))

        return changes


def render_unified_diff(path: Path, old_lines: list[str], new_lines: list[str]) -> str:
    """Renders a real unified diff (deletions included, hunks marked by @@ headers)."""
    diff = list(difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=f"{path} (last version you saw)",
        tofile=f"{path} (current on disk)",
        lineterm="",
        n=3,
    ))
    if len(diff) > MAX_DIFF_LINES:
        diff = diff[:MAX_DIFF_LINES]
        diff.append(f"... (diff truncated at {MAX_DIFF_LINES} lines; Read the file for full contents)")
    return "\n".join(diff)


async def file_changes_hook(event: "UserPromptEvent", ctx: "InvocationContext") -> "UserPromptEvent":
    """User-prompt hook: notify the model about externally modified files.

    For each tracked file whose content changed on disk since the transcript
    last saw it, injects a <system-reminder> with a unified diff and refreshes
    the tracker so the transcript is considered up to date again.
    """
    for path, old_lines, new_lines in ctx.file_state.collect_changes():
        reminder = dedent("""\
            <system-reminder>
            Note: {path} has been modified outside of your control (by the user or
            another process) since you last saw it. You don't need to tell the user
            this; only mention it if it helps show you understood their intent.
            Here is a diff of the external changes:

            {diff}
            </system-reminder>""").format(path=path, diff=render_unified_diff(path, old_lines, new_lines))
        event.context_post.append(TextMessageContent(text=reminder))

    return event
