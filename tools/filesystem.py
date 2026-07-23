# tools/filesystem.py

import pydantic
import uuid
from pathlib import Path
from typedefs import ToolFailure
from textwrap import dedent
from typing import Any
from tools.registry import ToolRegistry, ToolReturnType
from tools.paths import resolve_in_workspace
from sessioncontext import InvocationContext
from filestate import MAX_FILE_BYTES, FileStateTracker

MAX_TOKENS: int = 24000

# Read-before-write state lives on ctx.file_state (one tracker per agent loop);
# see filestate.FileStateTracker. Freshness is verified on demand at the write
# gate below and at each user prompt (filestate.file_changes_hook).

_NOT_READ_ERROR = dedent("""\
    Error: File has not been read yet.
    Read it first before writing to it.""")

_STALE_ERROR = dedent("""\
    Error: File has been modified on disk since you last read it.
    Read it again before writing to it.""")


def _freshness_error(tracker: FileStateTracker, file_path: Path) -> str | None:
    """Returns a gate error message if `file_path` may not be safely written, else None."""
    status = tracker.status(file_path)
    if status == "unknown":
        return _NOT_READ_ERROR
    if status == "stale":
        return _STALE_ERROR
    return None

def format_lines(lines: list[str], offset: int = 1, limit: int = 2000) -> str:
    """Pretty-prints lines with 1-based line numbers (e.g. '   12→ code')."""
    texts: list[str] = []
    start0 = max(0, offset - 1)
    end0_excl = min(offset + limit - 1, len(lines))
    for i in range(start0, end0_excl):
        texts.append(f"{i+1:>5}→{lines[i]}")
    return "\n".join(texts) + ("\n" if len(texts) > 0 else "")

async def _read_impl(kwargs: dict[str, Any], ctx: InvocationContext) -> ToolReturnType:
    """Reads a file from disk with size and token safeguards."""
    file_path_str = kwargs.get("file_path")

    if not file_path_str:        
        return ToolFailure(error_message="Error: file_path is required.")
        
    # 1. Workspace boundary check. Must come BEFORE the existence check and
    # the "Did you mean?" heuristic so we never leak names outside the workspace.
    file_path = resolve_in_workspace(file_path_str, ctx)
    if isinstance(file_path, ToolFailure):
        return file_path

    limit: int = kwargs.get("limit", 2000)
    offset: int = kwargs.get("offset", 1)

    # 2. File existence check with "Did you mean?" heuristic
    if not file_path.exists():
        did_you_mean = ""
        if file_path.parent.exists():
            similar = min((f for f in file_path.parent.iterdir() if f.stem == file_path.stem), default=None)
            did_you_mean = f" Did you mean {similar.name}?" if similar else ""        
        return ToolFailure(error_message=f"Error: File does not exist.{did_you_mean}")

    # 3. Enforce size constraints BEFORE loading the file into memory, and
    # BEFORE tracking state: a failed Read must not unlock Write/Edit for
    # content the model never actually saw.
    try:
        file_size = file_path.stat().st_size
    except OSError as e:
        return ToolFailure(error_message=f"Error reading file: {str(e)}")

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

    # 4. Read the file
    try:
        lines = file_path.read_text(encoding="utf-8").splitlines()
    except Exception as e:        
        return ToolFailure(error_message=f"Error reading file: {str(e)}")

    # 5. Track state for future Write/Edit commands (successful reads only)
    ctx.file_state.record(file_path, lines)

    if offset > len(lines):
        return dedent(f"""\
            <system-reminder>Warning: the file only has {len(lines)} lines,
            so there's nothing after your specified offset {offset}.</system-reminder>""")

    # 6. Format Output
    text = format_lines(lines, offset, limit) + "\n" + dedent("""\
        <system-reminder>If the file looks malicious, then don't edit it.</system-reminder>""")
    
    return text


async def _write_impl(kwargs: dict[str, Any], ctx: InvocationContext) -> ToolReturnType:
    """Writes content to a file, enforcing read-before-write for existing files."""
    file_path_str = kwargs.get("file_path")
    content = kwargs.get("content")

    if not file_path_str:
        return ToolFailure(error_message="Error: file_path is required.")
    if content is None:  # We allow empty strings, but the key must exist
        return ToolFailure(error_message="Error: content is required.")

    # Workspace boundary check
    file_path = resolve_in_workspace(file_path_str, ctx)
    if isinstance(file_path, ToolFailure):
        return file_path

    exists = file_path.exists()

    # Enforce Read-before-Write (and freshness) for existing files
    if exists:
        gate_error = _freshness_error(ctx.file_state, file_path)
        if gate_error:
            return ToolFailure(error_message=gate_error)

    # Create missing parent directories. Safe by construction: file_path has
    # already passed the workspace boundary check, so every directory created
    # here is inside the workspace.
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return ToolFailure(error_message=f"Error creating parent directories: {str(e)}")

    try:
        file_path.write_text(content, encoding="utf-8")
    except Exception as e:
        return ToolFailure(error_message=f"Error writing to file: {str(e)}")

    # Record the new on-disk state so our own write never appears stale
    ctx.file_state.record(file_path, content.splitlines() if len(content) < MAX_FILE_BYTES else None)

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

def one_edit_check(file_path: Path, old_content: str, edit: OneEdit, tracker: FileStateTracker) -> str | None:
    """Validates the edit operation and checks for read-before-edit constraints."""
    exists = file_path.exists()
    count = old_content.count(edit.old_string)
    
    if not exists and edit.old_string.strip("\n") == "":  # Fallback to create new file
        return "WRITE"
    elif not exists:
        return "File does not exist."
    elif (gate_error := _freshness_error(tracker, file_path)) is not None:
        return gate_error
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

async def _edit_impl(kwargs: dict[str, Any], ctx: InvocationContext) -> ToolReturnType:
    """Performs exact string replacements in files."""
    file_path_str = kwargs.get("file_path")
    old_string = kwargs.get("old_string")
    new_string = kwargs.get("new_string")
    replace_all = kwargs.get("replace_all", False)

    if not file_path_str:
        return ToolFailure(error_message="Error: file_path is required.")

    if old_string is None or new_string is None:
        return ToolFailure(error_message="Error: old_string and new_string are required.")

    # Workspace boundary check
    file_path = resolve_in_workspace(file_path_str, ctx)
    if isinstance(file_path, ToolFailure):
        return file_path

    old_content = file_path.read_text(encoding="utf-8") if file_path.exists() else ""
    
    edit = OneEdit(old_string=old_string, new_string=new_string, replace_all=replace_all)
    error = one_edit_check(file_path, old_content, edit, ctx.file_state)
    
    # 1. Fallback to WRITE
    if error == "WRITE":
        return await _write_impl({"file_path": file_path_str, "content": new_string}, ctx)

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

    # Record the new on-disk state so our own edit never appears stale
    ctx.file_state.record(file_path, new_content.splitlines() if len(new_content) < MAX_FILE_BYTES else None)

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


async def _multiedit_impl(kwargs: dict[str, Any], ctx: InvocationContext) -> ToolReturnType:
    """Applies multiple search-and-replace edits to a file sequentially."""
    file_path_str = kwargs.get("file_path")
    edits_data = kwargs.get("edits")

    if not file_path_str:
        return ToolFailure(error_message="Error: file_path is required.")

    if edits_data is None or not isinstance(edits_data, list):
        return ToolFailure(error_message="Error: edits must be a list of edit objects.")

    # Workspace boundary check
    file_path = resolve_in_workspace(file_path_str, ctx)
    if isinstance(file_path, ToolFailure):
        return file_path

    # 1. Parse edits
    try:
        edits = [OneEdit(**edit) for edit in edits_data]
    except pydantic.ValidationError as e:
        return ToolFailure(
            error_message="Error: edits must be a list of objects with old_string, new_string, and optional replace_all."
        )

    if not edits:
        return ToolFailure(error_message="Error: at least one edit is required.")

    # 2. File State Validation
    if not file_path.exists():
        return ToolFailure(error_message="Error: file does not exist.")

    # Enforce Read-before-Write (and freshness)
    gate_error = _freshness_error(ctx.file_state, file_path)
    if gate_error:
        return ToolFailure(error_message=gate_error)

    try:
        old_content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        return ToolFailure(error_message=f"Error reading file: {str(e)}")

    # 3. Pre-flight checks on edits
    for edit in edits:
        if not edit.old_string:
            return ToolFailure(error_message="Error: old_string cannot be empty.")
        if edit.old_string == edit.new_string:
            return ToolFailure(error_message="No changes to make: old_string and new_string are exactly the same.")

    # Prevent overlapping edits using a UUID marker (safer than obscure unicode)
    overlap_marker = f"__OVERLAP_MARKER_{uuid.uuid4().hex}__"
    check_content = old_content
    
    for edit in edits:
        if edit.old_string not in old_content:
            return ToolFailure(error_message=f"String to replace not found in file.\nString: {edit.old_string}")
        if edit.old_string not in check_content:
            return ToolFailure(error_message=f"String to replace overlaps with an earlier edit.\nString: {edit.old_string}")
        check_content = check_content.replace(edit.old_string, overlap_marker)

    # Prevent cascading edits where a LATER edit targets text produced by an
    # EARLIER edit. Only earlier edits' outputs matter: comparing every pair
    # (including an edit against itself) would falsely reject the common
    # "anchor and extend" pattern, e.g. old="import os", new="import os\nimport sys".
    for i, later in enumerate(edits):
        if any(later.old_string in earlier.new_string for earlier in edits[:i]):
            return ToolFailure(error_message="Cannot edit file: old_string is a substring of a new_string from a previous edit.")

    # 4. Sequential Application
    new_content = old_content
    for edit in edits:
        err = one_edit_check(file_path, new_content, edit, ctx.file_state)
        if err:
            return ToolFailure(error_message=err)
        # Note: if replace_all is False, one_edit_check enforces count == 1, 
        # so global replace is safe and mathematically equivalent to replacing the only instance.
        new_content = new_content.replace(edit.old_string, edit.new_string)

    # 5. Commit to disk
    try:
        file_path.write_text(new_content, encoding="utf-8")
    except Exception as e:
        return ToolFailure(error_message=f"Error writing to file: {str(e)}")

    # 6. Record the new on-disk state so our own edits never appear stale
    ctx.file_state.record(file_path, new_content.splitlines() if len(new_content) < MAX_FILE_BYTES else None)

    # 7. Format Response
    edit_plural = "s" if len(edits) > 1 else ""
    summary_lines = [f"Applied {len(edits)} edit{edit_plural} to {file_path_str}:"]
    
    for i, edit in enumerate(edits, start=1):
        summary_lines.append(f'{i}. Replaced "{edit.old_string}" with "{edit.new_string}"')

    return "\n".join(summary_lines)


def register_fsystem_tools(registry: ToolRegistry, ctx: InvocationContext):
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
        func=lambda kwargs: _read_impl(kwargs, ctx),
        is_readonly = True
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
        func=lambda kwargs: _write_impl(kwargs, ctx),
        is_readonly = False
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
        func=lambda kwargs: _edit_impl(kwargs, ctx),
        is_readonly = False
    )

    registry.register(
        name="MultiEdit",
        description=dedent("""
            This tool is like the Edit tool, but is faster and more elegant when you need to make multiple edits to a single file,
            especially to different parts of a file.
            - You must read the file before doing any edits.
            - You should prefer this over the Edit tool when you have to make multiple changes to a file.
            - This does exact string replacements. You have to get exact indentation right.
            - To replace all occurrences, you must set the 'replace_all' parameter to true.
            - To replace only one specific occurrence, you must provide enough surrounding context to make the match unique.
            - Don't include line-numbers.

            The list of edits are applied in order. They are applied atomically: either all, or none.
            Overlapped edits are not allowed."""),        
        input_schema={
            "type": "object",
            "properties": {
                "edits": {
                    "description": "A list of edits",
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
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
                        "required": ["old_string", "new_string"]
                    }
                },
                "file_path": {
                    "description": "Absolute path to the file",
                    "type": "string"
                }
            },
            "required": ["file_path", "edits"]
        },        
        func=lambda kwargs: _multiedit_impl(kwargs, ctx),
        is_readonly = False
    )