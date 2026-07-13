# tools/filesearch.py

import re
import glob
import itertools
from pathlib import Path
from textwrap import dedent
from tools.registry import ToolRegistry, ToolReturnType
from typing import Any
from typedefs import ToolFailure

MAX_GLOB_RESULTS: int = 100

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