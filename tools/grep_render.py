# tools/grep_render.py
"""
Presentation layer shared by the two Grep backends.

Prisma has two content-search engines (ripgrep when it is installed, a built-in
Python walker otherwise) but must expose one tool. The model sees only one of
them per session and has no way to tell which, so their output has to be
identical down to the truncation wording and the shape of a file header — the
paths it reads here are fed straight back into Read.

Each backend therefore does only what makes it different: find matches. It
hands them over as `GrepRecord`s and everything downstream (grouping, caps,
summaries, notices) happens exactly once, here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, Sequence

from typedefs import ToolResult

# Output caps. Content lines and file rows are counted separately because the
# two output modes have very different information density.
MAX_GREP_LINES: int = 200
MAX_GREP_FILES: int = 100

# Longest line we are willing to show. ripgrep enforces this itself via
# --max-columns; the fallback engine calls clip_line().
MAX_LINE_COLUMNS: int = 250


class RecordKind(Enum):
    """What a single output row represents."""

    MATCH = "match"                # a line the pattern matched
    CONTEXT = "context"            # a neighbouring line shown for context
    CONTINUATION = "continuation"  # later line of a multiline match


@dataclass(frozen=True)
class GrepRecord:
    """One row of content-mode output.

    `line_no` is None when line numbers are switched off, and for continuation
    rows, which belong to the match that precedes them.

    `gap_before` marks a discontinuity: the row does not directly follow the
    previous one in its file. Backends set it rather than emitting a separator
    row of their own, so the marker is rendered in one place.
    """

    path: str  # workspace-relative, '/'-separated
    kind: RecordKind
    text: str
    line_no: int | None = None
    gap_before: bool = False


def content_limit(head_limit: int | None = None) -> int:
    """Resolves how many content rows to show."""
    return min(head_limit or MAX_GREP_LINES, MAX_GREP_LINES)


def file_limit(head_limit: int | None = None) -> int:
    """Resolves how many file rows to show."""
    return min(head_limit or MAX_GREP_FILES, MAX_GREP_FILES)


def clip_line(text: str) -> str:
    """Shortens an over-long line, mirroring ripgrep's --max-columns-preview."""
    if len(text) <= MAX_LINE_COLUMNS:
        return text

    return f"{text[:MAX_LINE_COLUMNS]} [... omitted end of long line]"


def absolute(workspace: Path, relative_path: str) -> str:
    """Turns a backend-reported path into an absolute one Read can consume."""
    return str(workspace / relative_path)


def no_matches() -> ToolResult:
    """The one empty-result shape both backends return."""
    return ToolResult(content="No matches found.", ui_summary="Found 0 matches")


def _plural(count: int, singular: str, plural: str) -> str:
    return singular if count == 1 else plural


def _row(record: GrepRecord) -> str:
    """Formats one content row.

    Continuation rows carry no separator: they are the tail of the match above,
    not a line the pattern was tested against.
    """
    if record.kind is RecordKind.CONTINUATION:
        return f"      {record.text}"

    separator = ":" if record.kind is RecordKind.MATCH else "-"
    number = "" if record.line_no is None else str(record.line_no)

    return f"{number:>5}{separator}{record.text}"


def render_content(
    records: Iterable[GrepRecord],
    workspace: Path,
    head_limit: int | None = None,
) -> ToolResult:
    """Groups records under one absolute-path header per file."""
    limit = content_limit(head_limit)

    lines: list[str] = []
    current_file: str | None = None
    matches = 0
    files = 0
    rendered = 0
    truncated = False

    for record in records:
        if rendered >= limit:
            truncated = True
            break

        if record.path != current_file:
            current_file = record.path
            files += 1
            lines.append(absolute(workspace, record.path))
        elif record.gap_before:
            lines.append("  ...")

        lines.append(_row(record))
        rendered += 1

        if record.kind is RecordKind.MATCH:
            matches += 1

    if not lines:
        return no_matches()

    if truncated:
        lines.append(
            f"\n(Results are truncated to {limit} lines. "
            "Consider a more specific pattern or path, or use "
            "output_mode='files_with_matches' to see which files match.)"
        )

    summary = (
        f"Found {matches} {_plural(matches, 'match', 'matches')} "
        f"in {files} {_plural(files, 'file', 'files')}"
    )
    if truncated:
        summary += " (truncated)"

    return ToolResult(content="\n".join(lines), ui_summary=summary)


def render_file_list(
    paths: Sequence[str],
    workspace: Path,
    head_limit: int | None = None,
) -> ToolResult:
    """Renders one absolute path per matching file."""
    if not paths:
        return no_matches()

    limit = file_limit(head_limit)
    truncated = len(paths) > limit

    lines = [absolute(workspace, path) for path in paths[:limit]]

    if truncated:
        lines.append(
            f"(Results are truncated to {limit} files. "
            "Consider a more specific path, glob, or pattern.)"
        )

    summary = (
        f"Found {len(paths)} "
        f"{_plural(len(paths), 'file', 'files')} with matches"
    )
    if truncated:
        summary += f" (showing {limit})"

    return ToolResult(content="\n".join(lines), ui_summary=summary)


def render_counts(
    counts: Sequence[tuple[str, int]],
    workspace: Path,
    head_limit: int | None = None,
) -> ToolResult:
    """Renders per-file match counts, then the total.

    The total covers every matching file, including those the cap hid: a count
    that silently excluded them would be worse than no count at all.
    """
    if not counts:
        return no_matches()

    limit = file_limit(head_limit)

    lines: list[str] = []
    total = 0
    truncated = False

    for relative_path, count in counts:
        total += count

        if len(lines) >= limit:
            truncated = True
            continue

        lines.append(f"{absolute(workspace, relative_path)}: {count}")

    if truncated:
        lines.append(
            f"(Results are truncated to {limit} files. "
            "Consider a more specific path, glob, or pattern.)"
        )

    files = len(counts)
    tally = (
        f"{total} {_plural(total, 'match', 'matches')} "
        f"in {files} {_plural(files, 'file', 'files')}"
    )

    lines.append(f"\nTotal: {tally}")

    return ToolResult(content="\n".join(lines), ui_summary=f"Found {tally}")
