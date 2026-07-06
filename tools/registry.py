from __future__ import annotations
import pydantic
from typing import Any, Callable, Awaitable, Union
from typedefs import ShellCallback, AgentCallback, TextMessageContent, ToolFailure

ToolReturnType = Union[str, list[TextMessageContent], ShellCallback, AgentCallback, ToolFailure]
ToolCallable = Callable[[dict[str, Any]], Awaitable[ToolReturnType]]

class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, dict[str, Any]] = {}
        self._callables: dict[str, ToolCallable] = {}

    def register(self, name: str, description: str, input_schema: dict, func: ToolCallable):
        self._tools[name] = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": input_schema
            }
        }
        self._callables[name] = func

    def clone_filtered(self, allowed_tools: list[str]) -> 'ToolRegistry':
        """Creates a new registry containing only the allowed tools."""
        new_reg = ToolRegistry()
        for name in allowed_tools:
            if name in self._callables:
                new_reg._tools[name] = self._tools[name]
                new_reg._callables[name] = self._callables[name]
        return new_reg

    def get_all_schemas(self) -> list[dict]:
        return list(self._tools.values())

    async def invoke(self, name: str, kwargs: dict) -> ToolReturnType:
        if name not in self._callables:
            return f"Error: Tool '{name}' not found."
        
        try:
            # Execute the native python function
            return await self._callables[name](kwargs)
        except Exception as e:
            return f"Error: tool '{name}': {str(e)}"