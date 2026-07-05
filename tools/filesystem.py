from tools.registry import ToolRegistry, ToolReturnType
from typing import Any
from pathlib import Path

async def _read_impl(kwargs: dict[str, Any]) -> ToolReturnType:
    """Reads a file from disk."""
    file_path = kwargs.get("file_path")
    if not file_path:
        return "Error: file_path is required."
        
    path = Path(file_path).resolve()
    if not path.exists():
        return f"Error: File {file_path} does not exist."
    if not path.is_file():
        return f"Error: {file_path} is a directory."
        
    try:
        content = path.read_text(encoding="utf-8")
        return content
    except Exception as e:
        return f"Error reading file: {e}"

def register_fs_tools(registry: ToolRegistry):
    registry.register(
        name="Read",
        description="Reads a file from disk. Use this to inspect code.",
        input_schema={
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Absolute or relative path"}
            },
            "required": ["file_path"]
        },
        func=_read_impl
    )
