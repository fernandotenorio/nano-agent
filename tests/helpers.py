# tests/helpers.py
"""
Shared test utilities.

Tools now return `ToolResult(content=..., ui_summary=...)` on success so the
terminal UI can render compact summaries. These helpers unwrap that envelope,
letting tests keep asserting on the raw LLM-facing content string.
"""

from typedefs import ToolResult


def unwrap(result):
    """Returns the LLM-facing content of a ToolResult; passthrough otherwise."""
    return result.content if isinstance(result, ToolResult) else result


def unwrapped(tool_impl):
    """Wraps an async tool impl so it returns plain content (legacy test contract)."""
    async def _wrapped(*args, **kwargs):
        return unwrap(await tool_impl(*args, **kwargs))
    return _wrapped
