# tools/filesearch.py
import asyncio
import re
import os
import heapq
import fnmatch
import itertools
from pathlib import Path
from textwrap import dedent
from tools.ignore import IgnoreMatcher
from tools.paths import resolve_in_workspace
from tools.registry import ToolRegistry, ToolReturnType
from typing import Any, Iterator
from typedefs import ToolFailure
from sessioncontext import InvocationContext
from wcmatch import glob

MAX_GLOB_RESULTS: int = 100
MAX_LS_ENTRIES: int = 400


async def _glob_impl(kwargs: dict[str, Any], ctx: InvocationContext) -> ToolReturnType:
    """Search for files using glob patterns, returning newest files first."""

    pattern = kwargs.get("pattern")
    path_str = kwargs.get("path", ".")

    if not isinstance(pattern, str) or not pattern:
        return ToolFailure(error_message="Error: pattern is required.")

    if not isinstance(path_str, str):
        return ToolFailure(error_message="Error: path must be a string.")

    # Workspace boundary check (resolves relative paths against ctx.cwd)
    base_path = resolve_in_workspace(path_str, ctx)
    if isinstance(base_path, ToolFailure):
        return base_path

    if not (base_path.exists() and base_path.is_dir()):
        return ToolFailure(
            error_message=f"Error: Directory does not exist or is not a valid directory: {base_path}"
        )

    raw_exclude = kwargs.get("exclude")

    # Defensively handle the case where the LLM passes a single string instead of a list
    if isinstance(raw_exclude, str):
        raw_exclude = [raw_exclude]

    user_exclude = (
        [x for x in raw_exclude if isinstance(x, str)]
        if isinstance(raw_exclude, list)
        else []
    )

    # Initialize the ignore matcher using workspace context
    ignore_matcher = IgnoreMatcher(workspace=ctx.workspace, extra_patterns=user_exclude)

    flags = (
        glob.GLOBSTAR |
        glob.BRACE |
        glob.DOTGLOB
    )

    try:
        search_matcher = glob.compile(pattern, flags=flags)
    except Exception as e:
        return ToolFailure(error_message=f"Error: invalid glob pattern: {e}")

    heap: list[tuple[float, str]] = []
    total_matches = 0

    def walk_iterative(start: Path) -> None:
        nonlocal total_matches
        stack: list[Path] = [start]

        while stack:
            directory = stack.pop()

            try:
                with os.scandir(directory) as entries:
                    for entry in entries:

                        rel_path = os.path.relpath(entry.path, base_path)
                        rel_path = rel_path.replace(os.sep, "/")
                        entry_path = Path(entry.path)

                        try:
                            is_symlink = entry.is_symlink()
                        except OSError:
                            continue

                        if is_symlink:
                            continue

                        try:
                            if entry.is_dir(follow_symlinks=False):
                                if ignore_matcher.ignores(entry_path, is_dir=True):
                                    continue
                                stack.append(entry_path)
                                continue

                            if not entry.is_file(follow_symlinks=False):
                                continue

                        except OSError:
                            continue

                        # Check if the file itself should be ignored
                        if ignore_matcher.ignores(entry_path, is_dir=False):
                            continue

                        # Check if it matches the glob pattern
                        if not search_matcher.match(rel_path):
                            continue

                        try:
                            mtime = entry.stat(follow_symlinks=False).st_mtime
                        except OSError:
                            mtime = 0.0

                        total_matches += 1
                        # Store the ABSOLUTE path: results must round-trip
                        # directly into Read/Edit (whose schemas expect
                        # absolute paths), regardless of the search base.
                        # entry.path is absolute because base_path is resolved.
                        item = (mtime, entry.path)

                        if len(heap) < MAX_GLOB_RESULTS:
                            heapq.heappush(heap, item)
                        else:
                            heapq.heappushpop(heap, item)

            except OSError:
                continue

    await asyncio.to_thread(walk_iterative, base_path)

    if not heap:
        return "No files found."

    results = sorted(heap, key=lambda x: (-x[0], x[1]))
    lines = [path for _, path in results]

    if total_matches > MAX_GLOB_RESULTS:
        lines.append(
            f"(Results are truncated to {MAX_GLOB_RESULTS}. "
            "Consider using a more specific path, pattern, or exclude list.)"
        )

    return "\n".join(lines)


async def _ls_impl(kwargs: dict[str, Any], ctx: InvocationContext) -> ToolReturnType:
    """Lists directory contents with a visual tree structure."""
    path_str = kwargs.get("path", ".")
    
    # depth:
    #   0  = root only
    #   1  = immediate children (default)
    #   2+ = recurse to that depth
    #  <0  = unlimited recursion
    try:
        depth = int(kwargs.get("depth", 1))
    except (ValueError, TypeError):
        depth = 1
        
    target_level = depth if depth >= 0 else -1
    
    raw_ignore = kwargs.get("exclude")

    # Defensively handle the case where the LLM passes a single string instead of a list
    if isinstance(raw_ignore, str):
        raw_ignore = [raw_ignore]

    # Defensively ensure all ignore patterns are actually strings
    user_ignore = [x for x in raw_ignore if isinstance(x, str)] if isinstance(raw_ignore, list) else []

    # Initialize the ignore matcher using workspace context
    matcher = IgnoreMatcher(workspace=ctx.workspace, extra_patterns=user_ignore)

    # Workspace boundary check on the fully resolved path (catches both `..`
    # traversal and symlink roots that escape the workspace).
    boundary_check = resolve_in_workspace(path_str, ctx)
    if isinstance(boundary_check, ToolFailure):
        return boundary_check

    # For traversal and display we deliberately do NOT resolve symlinks:
    # listing a symlink should show the link itself ("name -> target") to
    # preserve symlink context in the LLM's spatial map.
    p = Path(path_str)
    target = p if p.is_absolute() else (ctx.cwd / p)

    # Safely handle broken symlinks by checking both exists() and is_symlink()
    if not target.exists() and not target.is_symlink():
        return ToolFailure(error_message=f"Error: Path does not exist: {target}")

    # Root Node Symlink & File Handling
    # If it's a file, a file-symlink, or a broken symlink, return a leaf node.
    # (If it's a symlink to a valid directory, we let it fall through to be explored).
    is_sym = target.is_symlink()
    if (is_sym and not target.exists()) or target.is_file():
        if is_sym:
            try:
                display = f"{target.name} -> {target.readlink()}"
            except OSError:
                display = f"{target.name} -> [unreadable link]"
        else:
            display = target.name        
        return f"{target}\n└── {display}"

    def get_dir_count(dir_path: Path) -> str:
        """Safely count non-ignored items in a directory using os.scandir."""
        try:
            with os.scandir(dir_path) as it:
                count = 0
                for entry in it:
                    try:                       
                        entry_path = Path(entry.path)
                        is_d, is_sym = entry_type(entry_path)
                        if not matcher.ignores(entry_path, is_dir=is_d):
                            count+= 1

                    except OSError:
                        # If a single file becomes unreadable or disappears, just skip it
                        continue
            return f" ({count} items)" if count != 1 else " (1 item)"
        except OSError:
            # Triggered if the directory itself cannot be read (e.g. permission denied)
            return ""

    def entry_type(path: Path) -> tuple[bool, bool]:
        """
        Returns:
            (is_directory, is_symlink)

        Symlink targets are inspected for type detection, but symlinks are
        never followed during recursion.
        """
        return path.is_dir(), path.is_symlink()
        
    def generate_tree(current_path: Path, prefix: str = '', level: int = -1) -> Iterator[str]:
        if level == 0:
            return

        try:
            all_items = list(current_path.iterdir())
        except OSError as e:
            yield prefix + f"└── [Unable to read: {e.strerror}]"
            return

        # Pre-compute stats safely to avoid TOCTOU races and mid-sort crashes
        entries = []
        for p in all_items:           
            try:
                # Stat optimization: if it's a symlink, treat it as a file (is_d = False)
                # This saves an expensive is_dir() stat call and groups symlinks cleanly.
                is_p_sym = p.is_symlink()
                is_d, _ = entry_type(p)

                if matcher.ignores(p, is_dir=is_d):
                    continue
                
                if is_p_sym:
                    try:
                        display = f"{p.name} -> {p.readlink()}"
                    except OSError:
                        display = f"{p.name} -> [unreadable link]"
                elif is_d:
                    display = f"{p.name}/{get_dir_count(p)}"
                else:
                    display = p.name
                    
                entries.append((is_d, display, p))
            except OSError:
                # File disappeared or permissions locked during enumeration
                entries.append((False, f"{p.name} [unreadable]", p))

        # Sort: Directories first (True > False), then alphabetically
        entries.sort(key=lambda x: (not x[0], x[2].name.lower()))
        
        last = len(entries) - 1
        for i, (is_d, display, path) in enumerate(entries):
            pointer = "└── " if i == last else "├── "
            yield prefix + pointer + display

            # We safely recurse because symlinks are guaranteed is_d=False            
            if is_d and not path.is_symlink():
                extension = '│   ' if i != last else '    '
                yield from generate_tree(path, prefix=prefix + extension, level=level - 1)

    # Provide explicit root context for the LLM's filesystem map
    target_display = f"{target}/" if target.is_dir() else str(target)
    lines = [target_display]
    
    if target_level == 0:
        lines.append("└── (depth limit reached)")
        return "\n".join(lines)
        
    tree_iterator = generate_tree(target, level=target_level)
    
    for line in itertools.islice(tree_iterator, MAX_LS_ENTRIES):
        lines.append(line)
        
    # If the iterator can still yield after MAX_LS_ENTRIES, we truncated.
    if next(tree_iterator, None) is not None:
        lines.append(
            f"\n... (Results truncated to {MAX_LS_ENTRIES} items. "
            "Use a smaller `depth` or a more specific path to explore further.)"
        )
        
    if len(lines) == 1:
        lines.append("└── [Empty Directory]")
        
    return "\n".join(lines)


def register_fsearch_tools(registry: ToolRegistry, ctx: InvocationContext):
    registry.register(
        name="Glob",
        description=dedent("""\
            Search for files using glob patterns.
            Supports:
              - Recursive directory search (using `**`)
              - Single-character matching (using `?`)
              - Range/character matching (e.g. `[a-z]`, `[!0-9]`)
              - Brace expansion (e.g. `src/{a,b}/*.js`)
              - Hidden file matching (dotfiles like `.env` are matched automatically)              

            Use this when searching by file name, as it avoids reading
            file contents and is significantly more efficient.

            Results are absolute paths, sorted newest first. You can pass them
            directly to the Read/Edit tools.

            The pattern itself is matched relative to the searched directory.

            Use the `exclude` parameter to skip directories or file patterns that are
            not relevant (e.g. 'node_modules/**', 'vendor/**', '**/*.min.js')."""),
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "description": "The directory to search in. Defaults to the current working directory.",
                    "type": "string"
                },
                "pattern": {
                    "description": "The glob pattern to search with. Supports recursive (**) and brace expansion ({a,b}).",
                    "type": "string"
                },
                "exclude": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional glob patterns to exclude from the search. "
                        "Useful for skipping large or irrelevant directories "
                        "(e.g. 'node_modules/**', 'venv/**') or file patterns."
                    )
                }
            },
            "required": ["pattern"]
        },        
        func=lambda kwargs: _glob_impl(kwargs, ctx),
        is_readonly = True
    )

    registry.register(
        name="ls",
        description=dedent("""\
            List the contents of a directory in a visual tree format to understand
            the project's structure.

            Use this tool for exploration. When you already know approximately what
            file or directory you're looking for, prefer the Glob tool instead.

            Common cache and dependency directories (such as .git, node_modules,
            __pycache__, and .venv) are excluded automatically."""),
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The directory to list. Defaults to the current directory.",
                    "default": "."
                },
                "depth": {
                    "type": "integer",
                    "description": (
                        "How many levels down to search. "
                        "Use 1 to list only immediate children (no recursion, flat list). "
                        "Use 2 or more to recurse to that depth. "
                        "Use 0 to only check if the directory exists without listing any files. "
                        "Use -1 for unlimited recursion."
                    ),
                    "default": 1
                },            
                "exclude": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional glob patterns to exclude from the listing. "
                        "Useful for skipping additional directories or files. "
                        "Common cache and dependency directories are already excluded automatically."
                    )
                }
            }
        },        
        
        func=lambda kwargs: _ls_impl(kwargs, ctx),
        is_readonly = True
    )