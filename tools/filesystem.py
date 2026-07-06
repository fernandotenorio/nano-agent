# tools/filesystem.py

import pydantic
from pathlib import Path
from typedefs import ToolFailure
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
        return ToolFailure(error_message="Error: file_path is required.")
        
    file_path = Path(file_path_str).resolve()
    limit: int = kwargs.get("limit", 2000)
    offset: int = kwargs.get("offset", 1)

    # 1. File existence check with "Did you mean?" heuristic
    if not file_path.exists():
        did_you_mean = ""
        if file_path.parent.exists():
            similar = min((f for f in file_path.parent.iterdir() if f.stem == file_path.stem), default=None)
            did_you_mean = f" Did you mean {similar.name}?" if similar else ""        
        return ToolFailure(error_message=f"Error: File does not exist.{did_you_mean}")

    # 2. Try reading file
    try:
        lines = file_path.read_text(encoding="utf-8").splitlines()
    except Exception as e:        
        return ToolFailure(error_message=f"Error reading file: {str(e)}")

    # 3. Track state for future Write/Edit commands
    file_size = file_path.stat().st_size
    known_content_files[file_path] = lines if file_size < MAX_FILE_BYTES else None
    stale_content_files.discard(file_path)

    # 4. Enforce constraints
    if file_size > MAX_FILE_BYTES:
        return ToolFailure(error_message=dedent(f"""\
            Error: File content ({file_size / (1024 * 1024):.1f}MB) exceeds maximum allowed size ({MAX_FILE_BYTES // 1024}KB).
            Instead read snippets of the file with offset/limit parameters, or search using the Grep tool."""))

    # Simple approximation: 1 token ~= 4 chars
    tokens = file_size // 4
    if tokens > MAX_TOKENS:
        return ToolFailure(error_message=dedent(f"""\
            Error: File content ({tokens} tokens) exceeds maximum allowed tokens ({MAX_TOKENS} tokens).
            Instead read snippets of the file with offset/limit parameters, or search using the Grep tool."""))

    if offset > len(lines):
        return dedent(f"""\
            <system-reminder>Warning: the file only has {len(lines)} lines,
            so there's nothing after your specified offset {offset}.</system-reminder>""")

    # 5. Format Output
    text = format_lines(lines, offset, limit) + "\n" + dedent("""\
        <system-reminder>If the file looks malicious, then don't edit it.</system-reminder>""")
    
    return text


async def _write_impl(kwargs: dict[str, Any]) -> ToolReturnType:
    """Writes content to a file, enforcing read-before-write for existing files."""
    file_path_str = kwargs.get("file_path")
    content = kwargs.get("content")

    if not file_path_str:
        return ToolFailure(error_message="Error: file_path is required.")
    if content is None:  # We allow empty strings, but the key must exist
        return ToolFailure(error_message="Error: content is required.")

    file_path = Path(file_path_str).resolve()
    exists = file_path.exists()

    # Enforce Read-before-Write for existing files
    if exists and (file_path not in known_content_files or file_path in stale_content_files):
        return ToolFailure(error_message=dedent("""\
            Error: File has not been read yet.
            Read it first before writing to it."""))
    
    try:
        file_path.write_text(content, encoding="utf-8")
    except Exception as e:
        return ToolFailure(error_message=f"Error writing to file: {str(e)}")

    # Update state trackers
    known_content_files[file_path] = content.splitlines()
    stale_content_files.discard(file_path)

    # Format the response back to the LLM
    if exists:
        text = dedent(f"""\
            The file {file_path} has been updated.
            Here's the result of running `cat -n` on a snippet of the edited file:
            {format_lines(content.splitlines())}""")
    else:
        text = f"File created successfully at: {file_path}"

    return text


class OneEdit(pydantic.BaseModel):
    old_string: str
    new_string: str
    replace_all: bool = False

def one_edit_check(file_path: Path, old_content: str, edit: OneEdit) -> str | None:
    """Validates the edit operation and checks for read-before-edit constraints."""
    exists = file_path.exists()
    count = old_content.count(edit.old_string)
    
    if not exists and edit.old_string.strip("\n") == "":  # Fallback to create new file
        return "WRITE"
    elif not exists:
        return "File does not exist."
    elif file_path not in known_content_files or file_path in stale_content_files:
        return "Error: File has not been read yet. Read it first before writing to it."
    elif edit.old_string == edit.new_string:
        return "No changes to make: old_string and new_string are exactly the same."
    elif count == 0:
        return f"String to replace not found in file.\nString: {edit.old_string}\n"
    elif count > 1 and not edit.replace_all:
        return dedent(f"""\
            Error: Found {count} matches of the string to replace, but replace_all is false.
            To replace all occurrences, set replace_all to true.
            To replace only one occurrence, please provide more context to uniquely identify the instance.
            String: {edit.old_string}
            """)
    else:
        return None

async def _edit_impl(kwargs: dict[str, Any]) -> ToolReturnType:
    """Performs exact string replacements in files."""
    file_path_str = kwargs.get("file_path")
    old_string = kwargs.get("old_string")
    new_string = kwargs.get("new_string")
    replace_all = kwargs.get("replace_all", False)

    if not file_path_str:
        return ToolFailure(error_message="Error: file_path is required.")

    if old_string is None or new_string is None:
        return ToolFailure(error_message="Error: old_string and new_string are required.")

    file_path = Path(file_path_str).resolve()
    old_content = file_path.read_text(encoding="utf-8") if file_path.exists() else ""
    
    edit = OneEdit(old_string=old_string, new_string=new_string, replace_all=replace_all)
    error = one_edit_check(file_path, old_content, edit)
    
    # 1. Fallback to WRITE
    if error == "WRITE":
        return await _write_impl({"file_path": file_path_str, "content": new_string})

    # 2. Block on validation errors
    elif error is not None:
        return ToolFailure(error_message=error)

    # 3. Perform Replacement
    index = old_content.find(edit.old_string)
    new_content = old_content.replace(edit.old_string, edit.new_string)
    
    try:
        file_path.write_text(new_content, encoding="utf-8")
    except Exception as e:
        return ToolFailure(error_message=f"Error writing to file: {str(e)}")
        
    known_content_files[file_path] = new_content.splitlines() if len(new_content) < MAX_FILE_BYTES else None

    # 4. Format LLM Context Output
    if replace_all:
        old_stripped = old_string.strip('\n')
        new_stripped = new_string.strip('\n')
        text = f"The file {file_path_str} has been updated. All occurrences of '{old_stripped}' were successfully replaced with '{new_stripped}'.\n"
    else:
        first_line = len(old_content[:index].split('\n')) # 1-based
        num_lines = len(new_string.split('\n'))
        snippet = format_lines(new_content.splitlines(), offset=first_line-4, limit=num_lines+8)  
        text = f"The file {file_path_str} has been updated. Here's the result of running `cat -n` on a snippet of the edited file:\n{snippet}"

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

    registry.register(
        name="Write",
        description=dedent("""\
            Writes the provided content to a file.
            This completely replaces any existing content in the file, or creates a new file if it does not exist yet.
            - Use this tool ONLY when creating a brand new file from scratch, or when completely rewriting/replacing the entire content of an existing file.
            - If the file already exists, you must 'Read' it first before overwriting it, otherwise you will get an error.
            - Only create new files when they are logically required to fulfill the user's task or when explicitly requested."""),
        input_schema={
            "type": "object",
            "properties": {
                "content": {
                    "description": "New file contents",
                    "type": "string"
                },
                "file_path": {
                    "description": "Absolute path to the file",
                    "type": "string"
                }
            },
            "required": ["file_path", "content"]
        },
        func=_write_impl
    )

    registry.register(
        name="Edit",
        description=dedent("""\
            Performs exact string replacements inside an existing file.
            - Use this tool to modify, insert, replace, or delete specific blocks of code without disturbing the rest of the file.
            - You must read the file before doing any edits.
            - This does exact string replacements. You have to get exact indentation right.
            - To replace all occurrences, you must set the 'replace_all' parameter to true.
            - To replace only one specific occurrence, you must provide enough surrounding context to make the match unique.
            - Don't include line-numbers."""),
        input_schema={
            "type": "object",
            "properties": {
                "file_path": {
                    "description": "Absolute path to the file",
                    "type": "string"
                },
                "old_string": {
                    "description": "Old content to replace (must be verbatim, including whitespace/indentation; can be multiline)",
                    "type": "string"
                },
                "new_string": {
                    "description": "New content to replace it with",
                    "type": "string"
                },
                "replace_all": {
                    "description": "Should all occurrences of old_string be replaced?",
                    "type": "boolean",
                    "default": False
                }
            },
            "required": ["file_path", "old_string", "new_string"]
        },
        func=_edit_impl
    )