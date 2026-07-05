from tools.registry import ToolRegistry, ToolReturnType
from typedefs import BashCallback
from typing import Any

async def _bash_impl(kwargs: dict[str, Any]) -> ToolReturnType:
    """Instructs the loop to run a bash command."""
    command = kwargs.get("command")
    if not command:
        return "Error: command is required."
    
    # We don't run the bash command here! We return the Callback.
    # The agent loop intercepts this and handles the streaming execution natively.
    return BashCallback(command=command)


def register_bash_tools(registry: ToolRegistry):
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
        func=_bash_impl
    )