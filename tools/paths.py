# tools/paths.py
"""
Workspace boundary enforcement.

Every file-oriented tool must resolve model-supplied paths through
`resolve_in_workspace` before touching the disk. The helper:

  1. Resolves relative paths against `ctx.cwd` (NOT the process cwd, which
     matters for sub-agents and future headless invocations).
  2. Fully resolves the result (`Path.resolve()`), so both `..` traversal
     and symlink escapes are caught.
  3. Denies any path whose resolved form falls outside `ctx.workspace`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from typedefs import ToolFailure

if TYPE_CHECKING:
    from sessioncontext import InvocationContext


def resolve_in_workspace(path_str: str, ctx: "InvocationContext") -> Path | ToolFailure:
    """Resolves `path_str` and enforces the workspace boundary.

    Returns the fully resolved Path if it is inside the workspace root,
    otherwise a ToolFailure describing the denial.
    """
    p = Path(path_str)

    try:
        resolved = (ctx.cwd / p).resolve() if not p.is_absolute() else p.resolve()
        workspace = ctx.workspace.resolve()
    except (OSError, RuntimeError, ValueError) as e:
        return ToolFailure(error_message=f"Error: could not resolve path {path_str!r}: {e}")

    if not resolved.is_relative_to(workspace):
        return ToolFailure(
            error_message=(
                f"Error: Path '{path_str}' resolves to '{resolved}', which is outside "
                f"the workspace root ({workspace}). Access denied."
            )
        )

    return resolved
