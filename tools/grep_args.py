# tools/grep_args.py
"""
Argument plumbing shared by the two Grep backends.

Validation lives here for the same reason rendering lives in grep_render: the
model cannot tell which engine is running, so a rejected pattern or an
out-of-workspace path must be rejected identically either way.

The forgiving coercions this relies on (as_count, as_str_list, ...) are not
grep-specific and live in tools/args.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tools.args import as_count, as_str_list
from tools.paths import resolve_in_workspace
from typedefs import ToolFailure

if TYPE_CHECKING:
    from sessioncontext import InvocationContext

OUTPUT_MODES = ("content", "files_with_matches", "count")


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
