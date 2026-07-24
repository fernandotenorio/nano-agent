# ui/summaries.py
"""
Pure helpers that turn a tool invocation (name + raw arguments) into a short
human-readable one-liner. UI-backend agnostic: every UI implementation gets
the same text.
"""

from __future__ import annotations

from typing import Any


def _short(value: Any, max_len: int = 60) -> str:
    text = str(value).replace("\n", " ")
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def summarize_call(name: str, args: Any) -> str:
    """Returns a concise summary of what a tool is about to do."""
    if not isinstance(args, dict):
        return f"{name}(...)"

    if name == "Shell":
        return f"$ {_short(args.get('command', ''), 80)}"

    if name == "Read":
        location = _short(args.get("file_path", "?"))
        offset, limit = args.get("offset"), args.get("limit")
        if offset is not None or limit is not None:
            start = int(offset) if offset is not None else 1
            span = f", lines {start}-{start + int(limit) - 1}" if limit is not None else f", from line {start}"
            return f"Read({location}{span})"
        return f"Read({location})"

    if name in ("Write", "Edit"):
        return f"{name}({_short(args.get('file_path', '?'))})"

    if name == "MultiEdit":
        edits = args.get("edits")
        count = len(edits) if isinstance(edits, list) else "?"
        return f"MultiEdit({_short(args.get('file_path', '?'))}, {count} edits)"

    if name == "Glob":
        pattern = _short(args.get("pattern", "?"), 40)
        path = args.get("path")
        return f"Glob({pattern}, {_short(path, 40)})" if path else f"Glob({pattern})"

    if name == "ls":
        return f"ls({_short(args.get('path', '.'), 40)})"

    if name == "Task":
        subagent = args.get("subagent_type", "default-agent")
        description = _short(args.get("description", ""), 50)
        return f"Task({subagent}: {description})" if description else f"Task({subagent})"

    if name == "SubmitPlan":
        return "SubmitPlan"

    # Generic fallback: show up to three truncated key=value pairs.
    pairs = ", ".join(f"{k}={_short(v, 30)}" for k, v in list(args.items())[:3])
    return f"{name}({pairs})" if pairs else f"{name}()"
