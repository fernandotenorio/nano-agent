# tools/grep_args.py
"""
Argument plumbing shared by the two Grep backends.

Validation lives here for the same reason rendering lives in grep_render: the
model cannot tell which engine is running, so a rejected pattern or an
out-of-workspace path must be rejected identically either way.

The coercions are deliberately forgiving. Schemas are a hint to a language
model, not a contract it can be held to: a field typed as an array of strings
regularly arrives as a bare string, and a number as "3". Bending where the
intent is obvious beats failing a search over a quoting detail.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tools.paths import resolve_in_workspace
from typedefs import ToolFailure

if TYPE_CHECKING:
    from sessioncontext import InvocationContext

OUTPUT_MODES = ("content", "files_with_matches", "count")


def as_str_list(value: Any) -> list[str]:
    """Coerces a schema 'array of string' field the LLM may send as a string."""
    if isinstance(value, str):
        return [value] if value.strip() else []

    if isinstance(value, list):
        return [item for item in value if isinstance(item, str) and item.strip()]

    return []


def as_flag(kwargs: dict[str, Any], *names: str, default: bool = False) -> bool:
    """Reads a boolean field, accepting any of its accepted spellings."""
    for name in names:
        if name in kwargs and kwargs[name] is not None:
            return bool(kwargs[name])

    return default


def as_count(kwargs: dict[str, Any], *names: str) -> int | None:
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


@dataclass(frozen=True)
class GrepRequest:
    """A validated search request, engine-agnostic."""

    pattern: str
    target: Path           # absolute, guaranteed inside the workspace
    relative_target: str   # workspace-relative and '/'-separated, '.' for the root
    workspace: Path
    output_mode: str
    head_limit: int | None
    excludes: list[str]

    @property
    def is_file_target(self) -> bool:
        return self.target.is_file()


def parse_request(
    kwargs: dict[str, Any],
    ctx: "InvocationContext",
) -> GrepRequest | ToolFailure:
    """Validates tool arguments and resolves the search target.

    The target is expressed relative to the workspace root because that is what
    both engines need: ripgrep matches --ignore-file rules against its working
    directory, and the fallback matches globs against the same base.
    """
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

    workspace = ctx.workspace.resolve()

    return GrepRequest(
        pattern=pattern,
        target=target,
        relative_target=target.relative_to(workspace).as_posix() or ".",
        workspace=workspace,
        output_mode=output_mode,
        head_limit=as_count(kwargs, "head_limit"),
        excludes=as_str_list(kwargs.get("exclude")),
    )
