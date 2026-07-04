from __future__ import annotations
import pydantic
from pathlib import Path
from textwrap import dedent
from typing import Any, Callable, Awaitable, Union

from prompts import SUB_AGENTS, get_subagent_system_prompt
from typedefs import BashCallback, AgentCallback

# A tool function takes a dictionary of arguments and returns either a string 
# (for direct insertion into the transcript) or a Callback object (handled by the loop).
ToolReturnType = Union[str, BashCallback, AgentCallback]
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
            return f"Error executing tool '{name}': {str(e)}"

# ---------------------------------------------------------
# Core Tool Implementations
# ---------------------------------------------------------

async def read_impl(kwargs: dict[str, Any]) -> ToolReturnType:
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

async def bash_impl(kwargs: dict[str, Any]) -> ToolReturnType:
    """Instructs the loop to run a bash command."""
    command = kwargs.get("command")
    if not command:
        return "Error: command is required."
    
    # We don't run the bash command here! We return the Callback.
    # The agent loop intercepts this and handles the streaming execution natively.
    return BashCallback(command=command)

async def task_impl(kwargs: dict[str, Any]) -> ToolReturnType:
    prompt = kwargs.get("prompt")
    description = kwargs.get("description", "Delegated sub-task")
    subagent_type = kwargs.get("subagent_type", "default-agent")
    
    if not prompt:
        return "Error: prompt is required."

    # Find the requested profile
    profile = next((sa for sa in SUB_AGENTS if sa.type == subagent_type), None)
    if not profile:
        available = ", ".join(sa.type for sa in SUB_AGENTS)
        return f"Error: subagent_type '{subagent_type}' not recognized. Available: {available}"

    # Prepare user content (including CLAUDE.md if it exists)
    # TODO: dedup CLAUDE.md injection
    claude_path = Path("CLAUDE.md")
    claude_text = ""
    if claude_path.exists():
        content = claude_path.read_text(encoding="utf-8")        
        claude_text = dedent(f'''
        <system-reminder>
        Project instructions:

        {content}
        
        </system-reminder>''')

    return AgentCallback(
        subagent_type=subagent_type,
        callback_description=description,
        tools=profile.tools,
        system_content=get_subagent_system_prompt(profile),
        user_content=f"{claude_text}Task:\n{prompt}"
    )

# ---------------------------------------------------------
# Registry Initialization
# ---------------------------------------------------------

def create_core_registry() -> ToolRegistry:
    registry = ToolRegistry()
    
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
        func=read_impl
    )
    
    registry.register(
        name="Bash",
        description="Executes a bash command. Good for running tests or grepping.",
        input_schema={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The bash command to run"}
            },
            "required": ["command"]
        },
        func=bash_impl
    )

    registry.register(
        name="Task",
        description=dedent("""\
        This tool launches a sub-agent for handling ambiguous, complex or multi-step tasks.
        These kinds of sub-agents are available. Use the subagent_type parameter to say which one you want.
        """) + \
        "\n".join([f"- {subagent.type}: {subagent.description} (Tools: {', '.join(subagent.tools) if subagent.tools else '*'})" for subagent in SUB_AGENTS]) + \
        dedent("""

        You should be eager to pro-actively use sub-agents. They are good for a wide variety of tasks,
        especially code-search and code-review. (Except, if you have just a simple job to do like Read or Grep,
        then there's no need for a sub-agent; you can do the work more efficiently yourself).

        You must take care to give the sub-agent a good prompt. The sub-agent knows nothing of your
        context. The only thing it has to go on is its prompt. You must put into the prompt
        everything that the sub-agent will need to do its job. Include guidance on whether you want it
        to research the codebase, or the internet, or write code, or review code, or read files.
        
        You will have no insight into how the sub-agent goes about its work. You will only see
        the sub-agent's final report when it's finished. You should put into the prompt any
        instructions you want to give about what that final report should include.
        
        The user will have no knowledge of how the sub-agent responded, and they cannot read its
        final report. Anything you want the user to know, you must report to the user yourself.
        """),
        input_schema={
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Comprehensive instructions for the sub-agent."},
                "subagent_type": {"type": "string", "description": "Type of sub-agent (default-agent)"}
            },
            "required": ["prompt"]
        },
        func=task_impl
    )

    return registry