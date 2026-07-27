# tools/grep.py
"""
Content search backed by the ripgrep CLI.

Prisma does not implement its own regex file walker: ripgrep is faster than
anything we could write in Python, already understands .gitignore, and is on
most developer machines. This module is the adapter around it.

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
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import tempfile
from pathlib import Path
from textwrap import dedent
from typing import Any

from sessioncontext import InvocationContext
from tools.ignore import IgnoreMatcher
from tools.paths import resolve_in_workspace
from tools.registry import ToolRegistry, ToolReturnType
from typedefs import ToolFailure, ToolResult

# Output caps. Content lines and file rows are counted separately because the
# two output modes have very different information density.
MAX_GREP_LINES: int = 200
MAX_GREP_FILES: int = 100

# Hard ceiling on what we read from ripgrep. Reaching it kills the search:
# there is no point in scanning further when we already have more than we can
# show. Long lines are clipped by ripgrep itself (see MAX_LINE_COLUMNS).
MAX_OUTPUT_BYTES: int = 400_000
MAX_STDERR_BYTES: int = 8_000
MAX_LINE_COLUMNS: int = 250

RG_TIMEOUT: float = 30.0
RG_DRAIN_GRACE: float = 5.0
RG_ERROR_LINES: int = 8

OUTPUT_MODES = ("content", "files_with_matches", "count")

# With --null, every record starts with 'path\0'. In content mode the rest is
# 'NUM:text' for a match and 'NUM-text' for a context line.
_CONTENT_PAYLOAD = re.compile(r"^(\d+)([:-])(.*)$", re.DOTALL)

_RG_MISSING_ERROR = dedent("""\
    Error: ripgrep (rg) was not found on PATH, so content search is unavailable.
    Install ripgrep, or fall back to the Glob tool for filename search and the
    Shell tool for one-off content searches.""")


def _as_str_list(value: Any) -> list[str]:
    """Coerces a schema 'array of string' field the LLM may send as a string."""
    if isinstance(value, str):
        return [value] if value.strip() else []

    if isinstance(value, list):
        return [item for item in value if isinstance(item, str) and item.strip()]

    return []


def _as_flag(kwargs: dict[str, Any], *names: str, default: bool = False) -> bool:
    """Reads a boolean field, accepting any of its accepted spellings."""
    for name in names:
        if name in kwargs and kwargs[name] is not None:
            return bool(kwargs[name])

    return default


def _as_count(kwargs: dict[str, Any], *names: str) -> int | None:
    """Reads a positive integer field; ignores junk instead of failing."""
    for name in names:
        raw = kwargs.get(name)
        if raw is None:
            continue

        try:
            value = int(raw)
        except (TypeError, ValueError):
            continue

        if value > 0:
            return value

    return None


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
) -> list[str]:
    """Translates tool arguments into a ripgrep command line."""
    args = [
        rg,
        "--color", "never",
        "--crlf",                # so '^' and '$' anchor correctly on CRLF files
        "--no-messages",         # unreadable files are not the model's problem
        "--hidden",              # dotfiles are searchable, like Glob's DOTGLOB
        "--no-ignore-vcs",       # .gitignore reaches us through --ignore-file
        "--with-filename",       # rg omits it for single-file searches
        "--no-heading",          # one self-describing record per line
        "--null",                # 'path\0...': unambiguous on every platform
        "--sortr", "modified",   # newest first (like Glob), and deterministic
        "--max-columns", str(MAX_LINE_COLUMNS),
        "--max-columns-preview",
        "--ignore-file", ignore_file,
    ]

    if _as_flag(kwargs, "-i", "case_insensitive"):
        args.append("--ignore-case")

    if _as_flag(kwargs, "multiline"):
        args.extend(["--multiline", "--multiline-dotall"])

    for pattern_glob in _as_str_list(kwargs.get("glob")):
        args.extend(["--glob", pattern_glob])

    file_type = kwargs.get("type")
    if isinstance(file_type, str) and file_type.strip():
        args.extend(["--type", file_type.strip()])

    if output_mode == "files_with_matches":
        args.append("--files-with-matches")
    elif output_mode == "count":
        args.append("--count-matches")
    else:
        if _as_flag(kwargs, "-n", "line_numbers", default=True):
            args.append("--line-number")

        for flag, names in (("--after-context", ("-A", "after_context")),
                            ("--before-context", ("-B", "before_context")),
                            ("--context", ("-C", "context"))):
            count = _as_count(kwargs, *names)
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
        process.kill()
        await process.wait()
        return ToolFailure(
            error_message=(
                f"Error: search timed out after {RG_TIMEOUT:0.0f}s. "
                "Narrow it down with the path, glob, or type parameters."
            )
        )

    if overflowed:
        # We already hold more output than we can show; stop the search.
        process.kill()

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


def _absolute(workspace: Path, relative_path: str) -> str:
    """Turns a ripgrep-reported path into an absolute one Read can consume."""
    return str(workspace / relative_path)


def _render_content(
    stdout: str,
    workspace: Path,
    max_lines: int,
) -> tuple[list[str], int, int, bool]:
    """Groups ripgrep's records under one header per file.

    Returns (lines, match_count, file_count, truncated).
    """
    lines: list[str] = []
    current_file: str | None = None
    matches = 0
    files = 0
    rendered = 0
    truncated = False

    for raw in stdout.split("\n"):
        record = raw.rstrip("\r")
        if not record:
            continue

        if "\0" not in record:
            # Either rg's '--' separator between context blocks, or the tail of
            # a multiline match, which rg prints without repeating the path.
            if current_file is None or rendered >= max_lines:
                continue

            lines.append("  ..." if record.strip() == "--" else f"      {record}")
            rendered += 1
            continue

        relative_path, payload = record.split("\0", 1)

        if rendered >= max_lines:
            truncated = True
            break

        if relative_path != current_file:
            current_file = relative_path
            files += 1
            lines.append(_absolute(workspace, relative_path))

        parsed = _CONTENT_PAYLOAD.match(payload)
        if parsed:
            line_no, separator, text = parsed.groups()
            lines.append(f"{line_no:>5}{separator}{text}")
            if separator == ":":
                matches += 1
        else:
            # Line numbers were switched off; the payload is the raw line.
            lines.append(f"     :{payload}")
            matches += 1

        rendered += 1

    return lines, matches, files, truncated


def _render_file_list(
    stdout: str,
    workspace: Path,
    max_files: int,
) -> tuple[list[str], int, bool]:
    """Renders --files-with-matches output (NUL-separated, no line breaks)."""
    paths = [
        entry.strip("\r\n")
        for entry in stdout.split("\0")
        if entry.strip("\r\n")
    ]

    lines = [_absolute(workspace, path) for path in paths[:max_files]]

    return lines, len(paths), len(paths) > max_files


def _render_counts(
    stdout: str,
    workspace: Path,
    max_files: int,
) -> tuple[list[str], int, int, bool]:
    r"""Renders --count-matches output ('path\0count' per line)."""
    lines: list[str] = []
    total = 0
    files = 0
    truncated = False

    for raw in stdout.split("\n"):
        record = raw.rstrip("\r")
        if "\0" not in record:
            continue

        relative_path, payload = record.split("\0", 1)

        try:
            count = int(payload.strip())
        except ValueError:
            continue

        total += count
        files += 1

        if len(lines) >= max_files:
            truncated = True
            continue

        lines.append(f"{_absolute(workspace, relative_path)}: {count}")

    return lines, total, files, truncated


async def _grep_impl(kwargs: dict[str, Any], ctx: InvocationContext) -> ToolReturnType:
    """Searches file contents with ripgrep, honouring Prisma's ignore rules."""
    pattern = kwargs.get("pattern")
    path_str = kwargs.get("path", ".")
    output_mode = kwargs.get("output_mode") or "content"

    if not isinstance(pattern, str) or not pattern:
        return ToolFailure(error_message="Error: pattern is required.")

    if not isinstance(path_str, str):
        return ToolFailure(error_message="Error: path must be a string.")

    if output_mode not in OUTPUT_MODES:
        return ToolFailure(
            error_message=f"Error: output_mode must be one of: {', '.join(OUTPUT_MODES)}."
        )

    # Workspace boundary check (resolves relative paths against ctx.cwd)
    target = resolve_in_workspace(path_str, ctx)
    if isinstance(target, ToolFailure):
        return target

    if not target.exists():
        return ToolFailure(error_message=f"Error: Path does not exist: {target}")

    rg = shutil.which("rg")
    if rg is None:
        return ToolFailure(error_message=_RG_MISSING_ERROR)

    workspace = ctx.workspace.resolve()
    relative_target = target.relative_to(workspace).as_posix()

    # The same matcher Glob and ls build, exported wholesale so all three tools
    # search and list exactly the same set of files.
    matcher = IgnoreMatcher(
        workspace=workspace,
        extra_patterns=_as_str_list(kwargs.get("exclude")),
    )

    try:
        ignore_file = _write_ignore_file(matcher.export_patterns())
    except OSError as e:
        return ToolFailure(error_message=f"Error: could not stage ignore rules for ripgrep: {e}")

    try:
        args = _build_rg_args(
            rg,
            kwargs,
            output_mode,
            pattern,
            ignore_file,
            relative_target or ".",
        )
        outcome = await _run_rg(args, cwd=workspace)
    finally:
        _remove_quietly(ignore_file)

    if isinstance(outcome, ToolFailure):
        return outcome

    stdout, found = outcome
    if not found or not stdout.strip():
        return ToolResult(content="No matches found.", ui_summary="Found 0 matches")

    head_limit = _as_count(kwargs, "head_limit")

    if output_mode == "files_with_matches":
        max_files = min(head_limit or MAX_GREP_FILES, MAX_GREP_FILES)
        lines, files, truncated = _render_file_list(stdout, workspace, max_files)

        if truncated:
            lines.append(
                f"(Results are truncated to {max_files} files. "
                "Consider a more specific path, glob, or pattern.)"
            )

        summary = f"Found {files} file{'s' if files != 1 else ''} with matches"
        if truncated:
            summary += f" (showing {max_files})"

        return ToolResult(content="\n".join(lines), ui_summary=summary)

    if output_mode == "count":
        max_files = min(head_limit or MAX_GREP_FILES, MAX_GREP_FILES)
        lines, total, files, truncated = _render_counts(stdout, workspace, max_files)

        if truncated:
            lines.append(
                f"(Results are truncated to {max_files} files. "
                "Consider a more specific path, glob, or pattern.)"
            )

        lines.append(f"\nTotal: {total} match{'es' if total != 1 else ''} in {files} file{'s' if files != 1 else ''}")

        summary = f"Found {total} match{'es' if total != 1 else ''} in {files} file{'s' if files != 1 else ''}"
        return ToolResult(content="\n".join(lines), ui_summary=summary)

    max_lines = min(head_limit or MAX_GREP_LINES, MAX_GREP_LINES)
    lines, matches, files, truncated = _render_content(stdout, workspace, max_lines)

    if truncated:
        lines.append(
            f"\n(Results are truncated to {max_lines} lines. "
            "Consider a more specific pattern or path, or use "
            "output_mode='files_with_matches' to see which files match.)"
        )

    summary = (
        f"Found {matches} match{'es' if matches != 1 else ''} "
        f"in {files} file{'s' if files != 1 else ''}"
    )
    if truncated:
        summary += " (truncated)"

    return ToolResult(content="\n".join(lines), ui_summary=summary)


def register_grep_tools(registry: ToolRegistry, ctx: InvocationContext):
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
        func=lambda kwargs: _grep_impl(kwargs, ctx),
        is_readonly = True
    )
