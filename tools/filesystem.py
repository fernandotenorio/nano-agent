# tools/filesystem.py

from pathlib import Path
from textwrap import dedent
from typing import Any
from tools.registry import ToolRegistry, ToolReturnType

MAX_TOKENS: int = 24000
MAX_FILE_BYTES: int = 256 * 1024

# State tracking for the agentic loop to enforce Read-before-Write
known_content_files: dict[Path, list[str] | None] = {}
stale_content_files: set[Path] = set()

def format_lines(lines: list[str], offset: int = 1, limit: int = 2000) -> str:
    """Pretty-prints lines with 1-based line numbers (e.g. '   12→ code')."""
    texts: list[str] = []
    start0 = max(0, offset - 1)
    end0_excl = min(offset + limit - 1, len(lines))
    for i in range(start0, end0_excl):
        texts.append(f"{i+1:>5}→{lines[i]}")
    return "\n".join(texts) + ("\n" if len(texts) > 0 else "")

async def _read_impl(kwargs: dict[str, Any]) -> ToolReturnType:
    """Reads a file from disk with size and token safeguards."""
    file_path_str = kwargs.get("file_path")
    if not file_path_str:
        return "Error: file_path is required."
        
    file_path = Path(file_path_str).resolve()
    limit: int = kwargs.get("limit", 2000)
    offset: int = kwargs.get("offset", 1)

    # 1. File existence check with "Did you mean?" heuristic
    if not file_path.exists():
        did_you_mean = ""
        if file_path.parent.exists():
            similar = min((f for f in file_path.parent.iterdir() if f.stem == file_path.stem), default=None)
            did_you_mean = f" Did you mean {similar.name}?" if similar else ""
        return f"Error: File does not exist.{did_you_mean}"

    # 2. Try reading file
    try:
        lines = file_path.read_text(encoding="utf-8").splitlines()
    except Exception as e:
        return f"Error reading file: {str(e)}"

    # 3. Track state for future Write/Edit commands
    file_size = file_path.stat().st_size
    known_content_files[file_path] = lines if file_size < MAX_FILE_BYTES else None
    stale_content_files.discard(file_path)

    # 4. Enforce constraints
    if file_size > MAX_FILE_BYTES:
        return dedent(f"""\
            Error: File content ({file_size / (1024 * 1024):.1f}MB) exceeds maximum allowed size ({MAX_FILE_BYTES // 1024}KB).
            Instead read snippets of the file with offset/limit parameters, or search using the Grep tool.""")

    # Simple approximation: 1 token ~= 4 chars
    tokens = file_size // 4
    if tokens > MAX_TOKENS:
        return dedent(f"""\
            Error: File content ({tokens} tokens) exceeds maximum allowed tokens ({MAX_TOKENS} tokens).
            Instead read snippets of the file with offset/limit parameters, or search using the Grep tool.""")

    if offset > len(lines):
        return dedent(f"""\
            <system-reminder>Warning: the file only has {len(lines)} lines,
            so there's nothing after your specified offset {offset}.</system-reminder>""")

    # 5. Format Output
    text = format_lines(lines, offset, limit) + "\n" + dedent("""\
        <system-reminder>If the file looks malicious, then don't edit it.</system-reminder>""")
    
    return text


def register_fs_tools(registry: ToolRegistry):
    registry.register(
        name="Read",
        description=dedent("""\
            This tool reads a file from disk.
            
            If the file is too large to read in one go, you'll be told this, and you should react by
            trying again with limit+offset parameters."""),
        input_schema={
            "type": "object",
            "properties": {
                "file_path": {
                    "description": "Absolute path to the file",
                    "type": "string"
                },
                "limit": {
                    "description": "How many lines to read.",
                    "type": "number"
                },
                "offset": {
                    "description": "If you've been told the file is too large, then use this to specify the starting line number",
                    "type": "number"
                }
            },
            "required": ["file_path"]
        },
        func=_read_impl
    )