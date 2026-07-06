from tools.registry import ToolRegistry, ToolReturnType
from typedefs import BashCallback
from typing import Any
from textwrap import dedent

async def _bash_impl(kwargs: dict[str, Any]) -> ToolReturnType:
    """Instructs the loop to run a bash command."""
    command = kwargs.get("command")
    if not command:
        return "Error: command is required."
    
    description = kwargs.get("description")
    # mini_agent uses timeout in ms (default 120000 ms = 120s)
    timeout_ms = kwargs.get("timeout", 120000)
    
    return BashCallback(
        command=command,
        callback_description=description,
        timeout=timeout_ms / 1000.0
    )


def register_bash_tools(registry: ToolRegistry):
    registry.register(
        name="Bash",
        description=dedent("""\
            This tool runs a bash command.
            
            Where it is good to use a bash command:
            - To invoke the project's build/typecheck/test tools
            - To run deploy commands BUT ONLY with the user's explicit instruction
            - To run git commands to learn about the repository
            - To push changes BUT ONLY with the user's explicit instruction
            - To use familiar unix helper tools like awk, sed, or jq if that's needed to learn something
            - To run a helper script that you've written
            
            BE CAREFUL! THERE ARE MANY DANGEROUS BASH COMMANDS.
            - Be very careful about any "rm" command. Only do these with the user's explicit instruction.
            - Be careful to quote arguments properly.
            - Do not use bash commands where tools like Grep, Glob, LS already exist.
            - You can use `;` and `&&` for multiple commands.
            - Any directory-changes with `cd` are ephemeral. It's best not to change directory at all.
            """),
        input_schema={
            "type": "object",
            "properties": {
                "command": {
                    "description": "The command to execute",
                    "type": "string"
                },
                "description": {
                    "description": "One-line description of what this command does",
                    "type": "string"
                },
                "timeout": {
                    "description": "Optional timeout in milliseconds (max 600000)",
                    "type": "number"
                }
            },
            "required": ["command"]
        },
        func=_bash_impl
    )