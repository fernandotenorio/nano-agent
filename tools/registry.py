from __future__ import annotations
import logging
import pydantic
from typing import Any, Callable, Awaitable, Union
from typedefs import ShellCallback, AgentCallback, PlanApprovalCallback, TextMessageContent, ToolFailure, ToolResult

ToolReturnType = Union[str, list[TextMessageContent], ShellCallback, AgentCallback, PlanApprovalCallback, ToolFailure, ToolResult]
ToolCallable = Callable[[dict[str, Any]], Awaitable[ToolReturnType]]

class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, dict[str, Any]] = {}
        self._callables: dict[str, ToolCallable] = {}
        self._readonly_flags: dict[str, bool] = {}

    def register(self, name: str, description: str, input_schema: dict, func: ToolCallable, is_readonly: bool = False):
        self._tools[name] = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": input_schema
            }
        }
        self._callables[name] = func
        self._readonly_flags[name] = is_readonly

    def clone_filtered(self, allowed_tools: list[str]) -> 'ToolRegistry':
        """Creates a new registry containing only the allowed tools."""
        new_reg = ToolRegistry()
        for name in allowed_tools:
            if name in self._callables:
                new_reg._tools[name] = self._tools[name]
                new_reg._callables[name] = self._callables[name]
                new_reg._readonly_flags[name] = self._readonly_flags[name]
        return new_reg

    def clone_excluding(self, excluded_tools: list[str]) -> 'ToolRegistry':
        """Creates a new registry without the excluded tools."""
        allowed = [name for name in self._callables if name not in excluded_tools]
        return self.clone_filtered(allowed)

    def clone_readonly(self) -> 'ToolRegistry':
        """Creates a new registry containing ONLY read-only tools."""
        allowed = [name for name, is_ro in self._readonly_flags.items() if is_ro]
        return self.clone_filtered(allowed)

    def get_all_schemas(self) -> list[dict]:
        return list(self._tools.values())

    async def invoke(self, name: str, kwargs: dict) -> ToolReturnType:
        # Failures are returned as ToolFailure, never as a plain string: the
        # agent loop reads a bare string as a successful result, which would
        # report a crashed tool to the model under a green check.
        if name not in self._callables:
            return ToolFailure(
                error_message=f"Error: Tool '{name}' not found.",
                ui_summary=f"Unknown tool '{name}'",
            )

        try:
            # Execute the native python function
            return await self._callables[name](kwargs)
        except Exception as e:
            logging.exception("Tool %r raised", name)
            return ToolFailure(error_message=f"Error: tool '{name}': {str(e)}")
