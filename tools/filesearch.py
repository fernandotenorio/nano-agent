# tools/filesearch.py

import re
import os
import fnmatch
import glob
import itertools
from pathlib import Path
from textwrap import dedent
from tools.registry import ToolRegistry, ToolReturnType
from typing import Any, Iterator
from typedefs import ToolFailure

MAX_GLOB_RESULTS: int = 100

MAX_LS_ENTRIES: int = 400

DEFAULT_IGNORE = [
    ".git", "__pycache__", "node_modules", ".venv", "venv", 
    ".idea", ".vscode", "*.pyc", "*.pyo"
]

def _expand_braces(pattern: str) -> list[str]:
    """Expands bash-style brace patterns like 'src/{a,b}/*.py'."""
    matches = list(re.finditer(r'\{([^}]+)\}', pattern))
    parts: list[list[str]] = []
    last_end = 0
    
    for match in matches:
        parts.append([pattern[last_end:match.start()]])
        parts.append(match.group(1).split(','))    
        last_end = match.end()
        
    parts.append([pattern[last_end:]])
    return [''.join(combo) for combo in itertools.product(*parts)]


async def _glob_impl(kwargs: dict[str, Any]) -> ToolReturnType:
    """Searches for files using a glob pattern, returning recently modified files first."""
    pattern = kwargs.get("pattern")
    # Default to current working directory if path is not provided
    path_str = kwargs.get("path", ".")

    if not pattern:
        return ToolFailure(error_message="Error: pattern is required.")

    base_path = Path(path_str).resolve()
    
    # Returning ToolFailure here helps the LLM realize it provided a bad path,
    # rather than tricking it into thinking the directory is just empty.
    if not base_path.exists() or not base_path.is_dir():
        return ToolFailure(error_message=f"Error: Directory does not exist or is not a valid directory: {base_path}")

    try:
        patterns = _expand_braces(pattern)
    except Exception as e:
        return ToolFailure(error_message=f"Error expanding brace pattern: {str(e)}")

    all_matches: set[Path] = set()

    for pat in patterns:
        search_target = str(base_path / pat)
        # iglob is more memory efficient for massive directory structures
        for match_str in glob.iglob(search_target, recursive=True):
            p = Path(match_str)
            if p.is_file():
                all_matches.add(p)

    if not all_matches:
        return "No files found."

    # Safe sorting by modification time (descending)
    def get_mtime(p: Path) -> float:
        try:
            return p.stat().st_mtime
        except OSError:
            return 0.0

    # We must sort FIRST, then truncate to MAX_GLOB_RESULTS
    sorted_matches = sorted(list(all_matches), key=get_mtime, reverse=True)
    
    is_truncated = len(sorted_matches) > MAX_GLOB_RESULTS
    result_paths = [str(m) for m in sorted_matches[:MAX_GLOB_RESULTS]]

    if is_truncated:
        result_paths.append(
            f"(Results are truncated to {MAX_GLOB_RESULTS}. Consider using a more specific path or pattern.)"
        )

    return "\n".join(result_paths)


async def _ls_impl(kwargs: dict[str, Any]) -> ToolReturnType:
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
    
    raw_ignore = kwargs.get("ignore")
    # Defensively ensure all ignore patterns are actually strings
    user_ignore = [x for x in raw_ignore if isinstance(x, str)] if isinstance(raw_ignore, list) else []
    ignore_patterns = user_ignore + DEFAULT_IGNORE

    # Use .absolute() to preserve symlink context in the LLM's spatial map
    target = Path(path_str).absolute()

    # Safely handle broken symlinks by checking both exists() and is_symlink()
    if not target.exists() and not target.is_symlink():
        return ToolFailure(error_message=f"Error: Path does not exist: {target}")

    def should_ignore(name: str) -> bool:
        return any(fnmatch.fnmatch(name, pat) for pat in ignore_patterns)

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
                count = sum(1 for entry in it if not should_ignore(entry.name))
            return f" ({count} items)" if count != 1 else " (1 item)"
        except OSError:
            return ""

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
            if should_ignore(p.name):
                continue
            
            try:
                # Stat optimization: if it's a symlink, treat it as a file (is_d = False)
                # This saves an expensive is_dir() stat call and groups symlinks cleanly.
                is_p_sym = p.is_symlink()
                is_d = False if is_p_sym else p.is_dir()
                
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
            if is_d:
                extension = '│   ' if i != last else '    '
                yield from generate_tree(path, prefix=prefix+extension, level=level-1)

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


def register_fsearch_tools(registry: ToolRegistry):
    registry.register(
        name="Glob",
        description=dedent("""\
            Search for files using glob patterns (e.g., '**/*.py', 'src/{a,b}/*.js'). Sorts results by modification time.
            Use this rather than `grep` or `rg` for searching file names, because it is more efficient.
            
            If you are doing an ambiguous search that might need multiple successive attempts at Glob or Grep,
            you should do so with the Task tool."""),
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
                }
            },
            "required": ["pattern"]
        },
        func=_glob_impl
    )

    registry.register(
        name="ls",
        description=dedent("""\
            Lists the contents of a directory in a visual tree format. Helps you understand project structure.
            In larger projects, consider also using the Glob tool to locate which directories are of interest."""),
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The directory to list. Defaults to the current directory."
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
                "ignore": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of glob patterns to ignore. Common noise dirs like .git and node_modules are ignored automatically."
                }
            }
        },
        func=_ls_impl
    )