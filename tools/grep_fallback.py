# tools/grep_fallback.py
"""
Content search without ripgrep.

This is the engine Prisma registers as `Grep` when no `rg` binary is on PATH.
It is deliberately less capable: a straight walk of the workspace, one file at
a time, Python's `re` on each line. On a large tree it is an order of magnitude
slower than ripgrep, so it is bounded on every axis that could hang a session —
file size, file count, and wall clock.

Two properties matter more than speed:

  1. Output is byte-identical to the ripgrep backend, because the model cannot
     tell which engine it is talking to. All formatting therefore lives in
     grep_render, and argument handling in grep_args; this module only finds
     matches.

  2. Visibility is inherited from IgnoreMatcher rather than reimplemented. The
     ripgrep backend has to export its rules to a temporary file and switch off
     rg's own .gitignore handling; here the matcher is simply consulted during
     the walk, so Grep, Glob, and ls agree by construction.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from textwrap import dedent
from typing import Any, Iterator

from capabilities import IGNORE_WIDENED_NOTE
from notices import with_note
from sessioncontext import InvocationContext
from tools import grep_render as render
from tools.args import as_count, as_flag, as_str_list
from tools.grep_args import OUTPUT_MODES, GrepRequest, parse_request
from tools.grep_render import GrepRecord, RecordKind
from tools.ignore import IgnoreMatcher
from tools.registry import ToolRegistry, ToolReturnType
from typedefs import ToolFailure
from wcmatch import glob

# Wall-clock budget for one search. Pure Python is slow enough that an
# unbounded scan of a monorepo would stall the session, so the search returns
# what it has and says it is incomplete.
SEARCH_TIMEOUT: float = 20.0

# Files above this size are skipped rather than read into memory.
MAX_FILE_BYTES: int = 5 * 1024 * 1024

# Upper bound on files considered, as a backstop for pathological trees.
MAX_SCAN_FILES: int = 20_000

# A NUL byte in the first chunk means binary, the same heuristic ripgrep uses.
BINARY_SNIFF_BYTES: int = 8_192

# Guards against a pattern that matches the empty string turning one file into
# an unbounded count.
MAX_MATCHES_PER_FILE: int = 10_000

# Slash-free patterns are rewritten to '**/...', so gitignore semantics apply:
# a bare '*.py' matches at any depth. NEGATEALL makes a lone '!...' pattern mean
# "everything except this", which is how the ripgrep backend behaves.
_GLOB_FLAGS = (
    glob.GLOBSTAR
    | glob.BRACE
    | glob.DOTGLOB
    | glob.NEGATE
    | glob.NEGATEALL
)

# MATCH outranks CONTINUATION outranks CONTEXT: a line reached by two different
# routes is labelled by the most specific one.
_KIND_PRIORITY = {
    RecordKind.CONTEXT: 0,
    RecordKind.CONTINUATION: 1,
    RecordKind.MATCH: 2,
}

_TIMEOUT_NOTE = (
    f"(Note: the search was stopped after {SEARCH_TIMEOUT:0.0f}s and is "
    "incomplete. Narrow it with the path or glob parameters, or search for a "
    "more specific pattern.)"
)

_SCAN_CAP_NOTE = (
    f"(Note: the search stopped after considering {MAX_SCAN_FILES} files and is "
    "incomplete. Narrow it with the path or glob parameters.)"
)


@dataclass
class _Outcome:
    """Whatever the scan collected, in the shape the renderers expect."""

    records: list[GrepRecord] = field(default_factory=list)
    paths: list[str] = field(default_factory=list)
    counts: list[tuple[str, int]] = field(default_factory=list)
    timed_out: bool = False
    scan_capped: bool = False

    @property
    def note(self) -> str:
        """The caveat to append, if the search gave up before finishing."""
        if self.timed_out:
            return _TIMEOUT_NOTE

        if self.scan_capped:
            return _SCAN_CAP_NOTE

        return ""


def _normalize_glob(pattern: str) -> str:
    """Rewrites a ripgrep-style glob into one wcmatch matches identically.

    ripgrep's --glob uses gitignore syntax, where a pattern without a slash
    applies at every depth and a trailing slash means a whole directory.
    wcmatch matches full paths, so both cases need spelling out.
    """
    negated = pattern.startswith("!")
    body = (pattern[1:] if negated else pattern).lstrip("/")

    if body.endswith("/"):
        body += "**"

    if "/" not in body:
        body = f"**/{body}"

    return f"!{body}" if negated else body


def _compile_regex(kwargs: dict[str, Any], pattern: str) -> re.Pattern[str] | ToolFailure:
    """Compiles the search pattern, reporting syntax errors to the model."""
    flags = 0

    if as_flag(kwargs, "-i", "case_insensitive"):
        flags |= re.IGNORECASE

    if as_flag(kwargs, "multiline"):
        # DOTALL so '.' spans lines, MULTILINE so '^' and '$' still mean
        # line boundaries rather than the boundaries of the whole file.
        flags |= re.DOTALL | re.MULTILINE

    try:
        return re.compile(pattern, flags)
    except re.error as e:
        return ToolFailure(error_message=f"Error: invalid regular expression: {e}")


def _read_lines(path: Path) -> list[str] | None:
    """Reads a searchable file as lines, or None if it should be skipped.

    Line endings are normalized so '$' anchors behave on CRLF files, matching
    ripgrep's --crlf.
    """
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return None

        data = path.read_bytes()
    except OSError:
        return None

    if b"\0" in data[:BINARY_SNIFF_BYTES]:
        return None

    lines = [line.rstrip("\r") for line in data.decode("utf-8", errors="replace").split("\n")]

    # A trailing newline leaves an empty final element that is not a real line.
    if lines and lines[-1] == "":
        lines.pop()

    return lines


def _match_spans(
    lines: list[str],
    regex: re.Pattern[str],
    multiline: bool,
    budget: int,
) -> list[tuple[int, int]]:
    """Finds matches as inclusive 0-based line spans.

    A span covers more than one line only in multiline mode, where the trailing
    lines are rendered as continuations of the match that opened them.
    """
    if not multiline:
        return [
            (index, index)
            for index, line in enumerate(lines)
            if regex.search(line)
        ][:budget]

    text = "\n".join(lines)
    spans: list[tuple[int, int]] = []

    for match in regex.finditer(text):
        start = text.count("\n", 0, match.start())
        # end - 1 so a match stopping right after a newline does not claim the
        # following line; max() keeps zero-length matches on their own line.
        end = text.count("\n", 0, max(match.start(), match.end() - 1))

        spans.append((start, max(start, end)))

        if len(spans) >= budget:
            break

    return spans


def _count_matches(lines: list[str], regex: re.Pattern[str], multiline: bool) -> int:
    """Counts occurrences, not lines, mirroring rg --count-matches."""
    if multiline:
        return _capped_count(regex.finditer("\n".join(lines)))

    return sum(_capped_count(regex.finditer(line)) for line in lines)


def _capped_count(matches: Iterator[re.Match[str]]) -> int:
    """Counts matches, giving up at MAX_MATCHES_PER_FILE.

    A pattern that can match the empty string matches at every position, so an
    uncapped count would scale with the size of the file rather than with
    anything the caller cares about.
    """
    total = 0

    for _ in matches:
        total += 1

        if total >= MAX_MATCHES_PER_FILE:
            break

    return total


def _claim(kinds: dict[int, RecordKind], index: int, kind: RecordKind) -> None:
    existing = kinds.get(index)

    if existing is None or _KIND_PRIORITY[kind] > _KIND_PRIORITY[existing]:
        kinds[index] = kind


def _records_for_file(
    relative_path: str,
    lines: list[str],
    spans: list[tuple[int, int]],
    *,
    before: int,
    after: int,
    show_numbers: bool,
    budget: int,
) -> list[GrepRecord]:
    """Turns line spans into rendered records, context lines included."""
    kinds: dict[int, RecordKind] = {}

    for start, end in spans:
        _claim(kinds, start, RecordKind.MATCH)

        for index in range(start + 1, end + 1):
            _claim(kinds, index, RecordKind.CONTINUATION)

        for index in range(max(0, start - before), start):
            _claim(kinds, index, RecordKind.CONTEXT)

        for index in range(end + 1, min(len(lines), end + 1 + after)):
            _claim(kinds, index, RecordKind.CONTEXT)

    records: list[GrepRecord] = []
    previous: int | None = None

    for index in sorted(kinds):
        records.append(
            GrepRecord(
                path=relative_path,
                kind=kinds[index],
                text=render.clip_line(lines[index]),
                line_no=(index + 1) if show_numbers else None,
                gap_before=previous is not None and index != previous + 1,
            )
        )

        previous = index

        if len(records) >= budget:
            break

    return records


def _candidates(
    request: GrepRequest,
    matcher: IgnoreMatcher,
    globs: list[str],
    outcome: _Outcome,
    deadline: float,
) -> list[tuple[float, str, Path]]:
    """Collects searchable files as (mtime, workspace-relative path, path).

    The whole tree is walked before anything is read, because results are
    ordered newest-first and that order is not known until every candidate has
    been seen. Both budgets are therefore enforced here as well as during the
    scan: on a big enough tree the walk alone could outlast the time we have.

    An explicitly named file is always searched: the caller pointed at it, so
    neither the ignore rules nor the glob filter should second-guess them. This
    matches how ripgrep treats an explicit path argument.
    """
    workspace = request.workspace

    def relative(path: str) -> str:
        return os.path.relpath(path, workspace).replace(os.sep, "/")

    if request.target.is_file():
        try:
            mtime = request.target.stat().st_mtime
        except OSError:
            mtime = 0.0

        return [(mtime, relative(str(request.target)), request.target)]

    found: list[tuple[float, str, Path]] = []
    stack: list[Path] = [request.target]

    while stack:
        if time.monotonic() > deadline:
            outcome.timed_out = True
            break

        directory = stack.pop()

        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    entry_path = Path(entry.path)

                    try:
                        if entry.is_symlink():
                            continue

                        if entry.is_dir(follow_symlinks=False):
                            if not matcher.ignores(entry_path, is_dir=True):
                                stack.append(entry_path)
                            continue

                        if not entry.is_file(follow_symlinks=False):
                            continue
                    except OSError:
                        continue

                    if matcher.ignores(entry_path, is_dir=False):
                        continue

                    relative_path = relative(entry.path)

                    if globs and not glob.globmatch(relative_path, globs, flags=_GLOB_FLAGS):
                        continue

                    if len(found) >= MAX_SCAN_FILES:
                        outcome.scan_capped = True
                        break

                    try:
                        mtime = entry.stat(follow_symlinks=False).st_mtime
                    except OSError:
                        mtime = 0.0

                    found.append((mtime, relative_path, entry_path))
        except OSError:
            continue

        if outcome.scan_capped:
            break

    # Newest first, then by path so equal timestamps stay deterministic.
    found.sort(key=lambda item: (-item[0], item[1]))

    return found


def _search(
    request: GrepRequest,
    matcher: IgnoreMatcher,
    regex: re.Pattern[str],
    globs: list[str],
    *,
    multiline: bool,
    before: int,
    after: int,
    show_numbers: bool,
) -> _Outcome:
    """Walks and scans. Synchronous by design; the caller runs it off-loop."""
    outcome = _Outcome()
    deadline = time.monotonic() + SEARCH_TIMEOUT

    content_budget = render.content_limit(request.head_limit) + 1
    file_budget = render.file_limit(request.head_limit) + 1

    for _, relative_path, path in _candidates(request, matcher, globs, outcome, deadline):
        if time.monotonic() > deadline:
            outcome.timed_out = True
            break

        lines = _read_lines(path)
        if lines is None:
            continue

        if request.output_mode == "count":
            count = _count_matches(lines, regex, multiline)
            if count:
                outcome.counts.append((relative_path, count))
            continue

        if request.output_mode == "files_with_matches":
            if _match_spans(lines, regex, multiline, budget=1):
                outcome.paths.append(relative_path)

                if len(outcome.paths) >= file_budget:
                    break
            continue

        remaining = content_budget - len(outcome.records)
        spans = _match_spans(lines, regex, multiline, budget=remaining)

        if spans:
            outcome.records.extend(
                _records_for_file(
                    relative_path,
                    lines,
                    spans,
                    before=before,
                    after=after,
                    show_numbers=show_numbers,
                    budget=remaining,
                )
            )

            if len(outcome.records) >= content_budget:
                break

    return outcome


async def _grep_impl(kwargs: dict[str, Any], ctx: InvocationContext) -> ToolReturnType:
    """Searches file contents in-process, honouring Prisma's ignore rules."""
    request = parse_request(kwargs, ctx)
    if isinstance(request, ToolFailure):
        return request

    regex = _compile_regex(kwargs, request.pattern)
    if isinstance(regex, ToolFailure):
        return regex

    matcher = IgnoreMatcher(
        workspace=request.workspace,
        extra_patterns=request.excludes,
    )

    globs = [_normalize_glob(pattern) for pattern in as_str_list(kwargs.get("glob"))]

    context = as_count(kwargs, "-C", "context")
    before = as_count(kwargs, "-B", "before_context") or context or 0
    after = as_count(kwargs, "-A", "after_context") or context or 0

    outcome = await asyncio.to_thread(
        _search,
        request,
        matcher,
        regex,
        globs,
        multiline=as_flag(kwargs, "multiline"),
        before=before,
        after=after,
        show_numbers=as_flag(kwargs, "-n", "line_numbers", default=True),
    )

    if request.output_mode == "files_with_matches":
        result = render.render_file_list(outcome.paths, request.workspace, request.head_limit)
    elif request.output_mode == "count":
        result = render.render_counts(outcome.counts, request.workspace, request.head_limit)
    else:
        result = render.render_content(outcome.records, request.workspace, request.head_limit)

    if outcome.note:
        result = with_note(result, outcome.note)

    if matcher.git_status.degraded and ctx.notices.once("git-degraded-listing"):
        result = with_note(result, IGNORE_WIDENED_NOTE)

    return result


def register_fallback_grep_tools(registry: ToolRegistry, ctx: InvocationContext):
    """Registers the in-process Grep. Used when no ripgrep binary is available."""
    registry.register(
        name="Grep",
        description=dedent("""\
            Search file contents with a regular expression.

            Prefer this over running grep through the Shell tool: it never needs
            shell quoting, and it filters out the same files that Glob and ls hide
            (.gitignore, .prismaignore, and caches such as .git, node_modules,
            __pycache__, and .venv).

            - `pattern` uses Python regex syntax. Literal braces and parens must be
              escaped (e.g. `interface\\{\\}`), and `|`, `+`, `*` work as usual.
            - Patterns match within a single line unless `multiline` is true.
            - This search reads files one by one, so scope matters: narrow it with
              `path` and `glob` (e.g. `*.py`, `*.{ts,tsx}`) rather than scanning the
              whole workspace and retrying.
            - Use `output_mode: "files_with_matches"` when you only need to know
              which files are involved; those results are absolute paths that can
              be passed straight to Read.

            Binary files and very large files are skipped. Results are ordered
            newest-modified first, long lines are clipped, and output is capped: if
            you see a truncation notice, refine the search rather than paging."""),
        input_schema={
            "type": "object",
            "properties": {
                "pattern": {
                    "description": "The regular expression to search for in file contents.",
                    "type": "string"
                },
                "path": {
                    "description": "File or directory to search in. Defaults to the current working directory.",
                    "type": "string"
                },
                "glob": {
                    "description": (
                        "Glob filter for which files to search "
                        "(e.g. '*.py', '*.{ts,tsx}'). Prefix with '!' to exclude."
                    ),
                    "type": "string"
                },
                "output_mode": {
                    "description": (
                        "'content' shows matching lines (default), "
                        "'files_with_matches' shows only file paths, "
                        "'count' shows the number of matches per file."
                    ),
                    "type": "string",
                    "enum": list(OUTPUT_MODES),
                    "default": "content"
                },
                "-i": {
                    "description": "Case-insensitive search.",
                    "type": "boolean"
                },
                "-n": {
                    "description": "Show line numbers (content mode only; on by default).",
                    "type": "boolean"
                },
                "-A": {
                    "description": "Lines of context to show after each match (content mode only).",
                    "type": "number"
                },
                "-B": {
                    "description": "Lines of context to show before each match (content mode only).",
                    "type": "number"
                },
                "-C": {
                    "description": "Lines of context to show before and after each match (content mode only).",
                    "type": "number"
                },
                "multiline": {
                    "description": "Allow the pattern to span lines ('.' then also matches newlines).",
                    "type": "boolean"
                },
                "head_limit": {
                    "description": (
                        "Keep only the first N results (content lines, or files "
                        "for the other output modes)."
                    ),
                    "type": "number"
                },
                "exclude": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional gitignore-style patterns to skip "
                        "(e.g. 'vendor/', '*.min.js'). Applied on top of the "
                        "ignore rules that are always active."
                    )
                }
            },
            "required": ["pattern"]
        },
        func=lambda kwargs: _grep_impl(kwargs, ctx),
        is_readonly = True
    )
