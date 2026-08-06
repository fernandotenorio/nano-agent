# tools/grep.py
"""
Content search backed by the ripgrep CLI.

Prisma does not implement its own regex file walker when it can avoid it:
ripgrep is faster than anything we could write in Python and is on most
developer machines. This module is the adapter around it. When rg is absent,
tools/grep_fallback.py takes over and the model is none the wiser.

Two invariants shape the implementation:

  1. ripgrep matches --ignore-file rules relative to its working directory,
     so `rg` always runs from the workspace root and the search target is
     addressed relatively. That also keeps output free of drive letters,
     whose ':' would be ambiguous when parsing 'path:line:text'.

  2. IgnoreMatcher is the single source of truth for what is invisible. Its
     rules are exported into a temporary gitignore-formatted file and handed
     to `rg --ignore-file`, while ripgrep's own VCS handling is switched off.
     Letting ripgrep read .gitignore directly looks simpler but drifts from
     Glob and ls: ripgrep hides every path matching a pattern, whereas git
     (and therefore IgnoreMatcher) keeps tracked files visible. That drift
     would make Grep silently miss deliberately committed files such as a
     `.env.example` under `.env*`, or a checked-in `dist/` bundle.

Invariant 2 holds only while git can actually be asked. When it cannot, the
export carries no git verdict at all, and insisting on --no-ignore-vcs would
turn a small inconsistency into a search that walks node_modules and .env. So
that one case hands .gitignore back to ripgrep and says so in the result.
"""

from __future__ import annotations

import asyncio
import os
import re
import tempfile
from pathlib import Path
from textwrap import dedent
from typing import Any

from capabilities import RIPGREP_GITIGNORE_NOTE
from notices import with_note
from processes import kill_quietly
from sessioncontext import InvocationContext
from tools import grep_render as render
from tools.args import as_count, as_flag, as_str_list
from tools.grep_args import OUTPUT_MODES, parse_request
from tools.grep_render import GrepRecord, RecordKind
from tools.ignore import IgnoreMatcher
from tools.registry import ToolRegistry, ToolReturnType
from typedefs import ToolFailure, ToolResult

# Hard ceiling on what we read from ripgrep. Reaching it kills the search:
# there is no point in scanning further when we already have more than we can
# show. Long lines are clipped by ripgrep itself (see MAX_LINE_COLUMNS).
MAX_OUTPUT_BYTES: int = 400_000
MAX_STDERR_BYTES: int = 8_000

RG_TIMEOUT: float = 30.0
RG_DRAIN_GRACE: float = 5.0
RG_ERROR_LINES: int = 8

# With --null, every record starts with 'path\0'. In content mode the rest is
# 'NUM:text' for a match and 'NUM-text' for a context line.
_CONTENT_PAYLOAD = re.compile(r"^(\d+)([:-])(.*)$", re.DOTALL)


def _write_ignore_file(patterns: list[str]) -> str:
    """Materializes Prisma's ignore rules where ripgrep can read them.

    Written outside the workspace so the file can never turn up in results.
    """
    fd, path = tempfile.mkstemp(prefix="prisma-ignore-", suffix=".txt", text=True)

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write("\n".join(patterns) + "\n")
    except OSError:
        _remove_quietly(path)
        raise

    return path


def _remove_quietly(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def _build_rg_args(
    rg: str,
    kwargs: dict[str, Any],
    output_mode: str,
    pattern: str,
    ignore_file: str,
    target: str,
    *,
    own_vcs_ignores: bool,
) -> list[str]:
    """Translates tool arguments into a ripgrep command line.

    `own_vcs_ignores` is True while our --ignore-file speaks for git; when it is
    False, ripgrep is left to read .gitignore itself.
    """
    args = [
        rg,
        "--color", "never",
        "--crlf",                # so '^' and '$' anchor correctly on CRLF files
        "--no-messages",         # unreadable files are not the model's problem
        "--hidden",              # dotfiles are searchable, like Glob's DOTGLOB
        "--with-filename",       # rg omits it for single-file searches
        "--no-heading",          # one self-describing record per line
        "--null",                # 'path\0...': unambiguous on every platform
        "--sortr", "modified",   # newest first (like Glob), and deterministic
        "--max-columns", str(render.MAX_LINE_COLUMNS),
        "--max-columns-preview",
        "--ignore-file", ignore_file,
    ]

    if own_vcs_ignores:
        args.append("--no-ignore-vcs")  # .gitignore reaches us through --ignore-file

    if as_flag(kwargs, "-i", "case_insensitive"):
        args.append("--ignore-case")

    if as_flag(kwargs, "multiline"):
        args.extend(["--multiline", "--multiline-dotall"])

    for pattern_glob in as_str_list(kwargs.get("glob")):
        args.extend(["--glob", pattern_glob])

    file_type = kwargs.get("type")
    if isinstance(file_type, str) and file_type.strip():
        args.extend(["--type", file_type.strip()])

    if output_mode == "files_with_matches":
        args.append("--files-with-matches")
    elif output_mode == "count":
        args.append("--count-matches")
    else:
        if as_flag(kwargs, "-n", "line_numbers", default=True):
            args.append("--line-number")

        for flag, names in (("--after-context", ("-A", "after_context")),
                            ("--before-context", ("-B", "before_context")),
                            ("--context", ("-C", "context"))):
            count = as_count(kwargs, *names)
            if count is not None:
                args.extend([flag, str(count)])

    # -e guards patterns that start with '-'; -- guards the path.
    args.extend(["--regexp", pattern, "--", target])

    return args


async def _read_capped(stream: asyncio.StreamReader, cap: int) -> tuple[bytes, bool]:
    """Reads a stream up to `cap` bytes. Returns (data, hit_cap)."""
    chunks: list[bytes] = []
    size = 0

    while size < cap:
        chunk = await stream.read(65536)
        if not chunk:
            return b"".join(chunks), False

        chunks.append(chunk)
        size += len(chunk)

    return b"".join(chunks), True


async def _run_rg(args: list[str], cwd: Path) -> tuple[str, bool] | ToolFailure:
    """Runs ripgrep and returns (stdout, found_matches).

    ripgrep's exit codes are load-bearing: 0 means matches, 1 means none, and
    anything else is a real error (bad regex, unknown --type, ...).
    """
    try:
        process = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd),
        )
    except OSError as e:
        return ToolFailure(error_message=f"Error: could not run ripgrep: {e}")

    assert process.stdout is not None and process.stderr is not None

    # Both streams are drained concurrently: reading them in sequence would
    # deadlock as soon as ripgrep filled the pipe we are not reading.
    stdout_task = asyncio.create_task(_read_capped(process.stdout, MAX_OUTPUT_BYTES))
    stderr_task = asyncio.create_task(_read_capped(process.stderr, MAX_STDERR_BYTES))

    try:
        stdout_bytes, overflowed = await asyncio.wait_for(stdout_task, RG_TIMEOUT)
    except asyncio.TimeoutError:
        stdout_task.cancel()
        stderr_task.cancel()
        kill_quietly(process)
        await process.wait()
        return ToolFailure(
            error_message=(
                f"Error: search timed out after {RG_TIMEOUT:0.0f}s. "
                "Narrow it down with the path, glob, or type parameters."
            )
        )

    if overflowed:
        # We already hold more output than we can show; stop the search.
        kill_quietly(process)

    try:
        stderr_bytes, _ = await asyncio.wait_for(stderr_task, RG_DRAIN_GRACE)
    except asyncio.TimeoutError:
        stderr_task.cancel()
        stderr_bytes = b""

    exit_code = await process.wait()
    stdout = stdout_bytes.decode("utf-8", errors="replace")

    # A killed ripgrep reports failure, but we stopped it on purpose and the
    # output we captured is valid.
    if overflowed:
        return stdout, True

    if exit_code == 0:
        return stdout, True

    if exit_code == 1:
        return stdout, False

    # Keep several lines: ripgrep spells out regex errors across a small block
    # (the offending pattern, a caret, then the reason).
    stderr = stderr_bytes.decode("utf-8", errors="replace").strip()
    detail = (
        "\n".join(stderr.splitlines()[:RG_ERROR_LINES])
        if stderr
        else f"ripgrep exited with code {exit_code}"
    )
    return ToolFailure(error_message=f"Error: {detail}")


def _parse_content(stdout: str) -> list[GrepRecord]:
    """Turns ripgrep's NUL-delimited content output into records."""
    records: list[GrepRecord] = []
    current_file: str | None = None
    pending_gap = False

    for raw in stdout.split("\n"):
        record = raw.rstrip("\r")
        if not record:
            continue

        if "\0" not in record:
            if record.strip() == "--":
                # rg's marker between non-adjacent context blocks.
                pending_gap = True
                continue

            if current_file is None:
                continue

            # Tail of a multiline match: rg does not repeat the path.
            records.append(
                GrepRecord(
                    path=current_file,
                    kind=RecordKind.CONTINUATION,
                    text=record,
                )
            )
            continue

        relative_path, payload = record.split("\0", 1)
        current_file = relative_path

        parsed = _CONTENT_PAYLOAD.match(payload)
        if parsed:
            line_no, separator, text = parsed.groups()
            records.append(
                GrepRecord(
                    path=relative_path,
                    kind=RecordKind.MATCH if separator == ":" else RecordKind.CONTEXT,
                    text=text,
                    line_no=int(line_no),
                    gap_before=pending_gap,
                )
            )
        else:
            # Line numbers were switched off; the payload is the raw line, and
            # context lines are indistinguishable from matches.
            records.append(
                GrepRecord(
                    path=relative_path,
                    kind=RecordKind.MATCH,
                    text=payload,
                    gap_before=pending_gap,
                )
            )

        pending_gap = False

    return records


def _parse_file_list(stdout: str) -> list[str]:
    """Parses --files-with-matches output (NUL-separated, no line breaks)."""
    return [
        entry.strip("\r\n")
        for entry in stdout.split("\0")
        if entry.strip("\r\n")
    ]


def _parse_counts(stdout: str) -> list[tuple[str, int]]:
    r"""Parses --count-matches output ('path\0count' per line)."""
    counts: list[tuple[str, int]] = []

    for raw in stdout.split("\n"):
        record = raw.rstrip("\r")
        if "\0" not in record:
            continue

        relative_path, payload = record.split("\0", 1)

        try:
            counts.append((relative_path, int(payload.strip())))
        except ValueError:
            continue

    return counts


async def _grep_impl(
    kwargs: dict[str, Any],
    ctx: InvocationContext,
    rg: str,
) -> ToolReturnType:
    """Searches file contents with ripgrep, honouring Prisma's ignore rules."""
    request = parse_request(kwargs, ctx)
    if isinstance(request, ToolFailure):
        return request

    # The same matcher Glob and ls build, exported wholesale so all three tools
    # search and list exactly the same set of files.
    matcher = IgnoreMatcher(
        workspace=request.workspace,
        extra_patterns=request.excludes,
    )

    try:
        ignore_file = _write_ignore_file(matcher.export_patterns())
    except OSError as e:
        return ToolFailure(error_message=f"Error: could not stage ignore rules for ripgrep: {e}")

    try:
        args = _build_rg_args(
            rg,
            kwargs,
            request.output_mode,
            request.pattern,
            ignore_file,
            request.relative_target,
            own_vcs_ignores=not matcher.git_status.degraded,
        )
        outcome = await _run_rg(args, cwd=request.workspace)
    finally:
        _remove_quietly(ignore_file)

    if isinstance(outcome, ToolFailure):
        return outcome

    stdout, found = outcome
    if not found or not stdout.strip():
        result = render.no_matches()
    elif request.output_mode == "files_with_matches":
        result = render.render_file_list(
            _parse_file_list(stdout), request.workspace, request.head_limit
        )
    elif request.output_mode == "count":
        result = render.render_counts(
            _parse_counts(stdout), request.workspace, request.head_limit
        )
    else:
        result = render.render_content(
            _parse_content(stdout), request.workspace, request.head_limit
        )

    return _annotate(result, matcher, ctx)


def _annotate(
    result: ToolResult,
    matcher: IgnoreMatcher,
    ctx: InvocationContext,
) -> ToolResult:
    """Appends the degraded-git caveat, at most once per agent."""
    if not matcher.git_status.degraded:
        return result

    if not ctx.notices.once("git-degraded-ripgrep"):
        return result

    return with_note(result, RIPGREP_GITIGNORE_NOTE)


def register_grep_tools(registry: ToolRegistry, ctx: InvocationContext, rg: str):
    """Registers the ripgrep-backed Grep. `rg` must be a runnable binary path."""
    registry.register(
        name="Grep",
        description=dedent("""\
            Search file contents with a regular expression, powered by ripgrep.

            Prefer this over running grep/rg through the Shell tool: it is faster,
            it never needs shell quoting, and it filters out the same files that
            Glob and ls hide (.gitignore, .prismaignore, and caches such as .git,
            node_modules, __pycache__, and .venv).

            - `pattern` uses Rust regex syntax. Literal braces and parens must be
              escaped (e.g. `interface\\{\\}`), and `|`, `+`, `*` work as usual.
            - Patterns match within a single line unless `multiline` is true.
            - Narrow large searches with `path`, `glob` (e.g. `*.{ts,tsx}`), or
              `type` (e.g. `py`, `js`, `rust`) instead of retrying broad ones.
            - Use `output_mode: "files_with_matches"` when you only need to know
              which files are involved; those results are absolute paths that can
              be passed straight to Read.

            Results are ordered newest-modified first, so the most relevant files
            usually come first. Long lines are clipped, and output is capped: if
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
                "type": {
                    "description": "ripgrep file type filter (e.g. 'py', 'js', 'rust', 'go'). Faster than glob for whole languages.",
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
        func=lambda kwargs: _grep_impl(kwargs, ctx, rg),
        is_readonly = True
    )
