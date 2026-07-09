from tools.registry import ToolRegistry, ToolReturnType
from typedefs import ShellCallback, ToolFailure
from typing import Any
from textwrap import dedent

async def _bash_impl(kwargs: dict[str, Any]) -> ToolReturnType:
    """Instructs the loop to run a bash command."""
    command = kwargs.get("command")

    # Clean whitespace and strip command before validation
    if command is not None:
        command = command.strip()
        
    if not command:
        return ToolFailure(error_message="Error: command is required.")
    
    description = kwargs.get("description")
    # mini_agent uses timeout in ms (default 120000 ms = 120s)
    timeout_ms = kwargs.get("timeout", 120000)
    
    return ShellCallback(
        command=command,
        callback_description=description,
        timeout=timeout_ms / 1000.0
    )


def register_shell_tools(registry: ToolRegistry):
    registry.register(
        name="Shell",
        description=dedent("""\
            Executes a command in the system's default shell (bash on Unix, cmd.exe on Windows).
            - Use commands appropriate for the current OS (revealed in the system prompt).
            
            Where it is good to use a bash command:
            - To invoke the project's build/typecheck/test tools
            - To run deploy commands BUT ONLY with the user's explicit instruction
            - To run git commands to learn about the repository
            - To push changes BUT ONLY with the user's explicit instruction
            - To use familiar unix helper tools like awk, sed, or jq if that's needed to learn something
            - To run a helper script that you've written
            
            BE CAREFUL! THERE ARE MANY DANGEROUS BASH COMMANDS.
            - Be very careful about any "rm" or "del" command. Only do these with the user's explicit instruction.
            - Be careful to quote arguments properly.
            - Do not use shell commands if you have an equivalent tool at your disposal.
            - You can chain multiple commands.
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