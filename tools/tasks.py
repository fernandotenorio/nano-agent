from pathlib import Path
import pydantic
from environment import get_environment_details
from textwrap import dedent
from tools.registry import ToolRegistry, ToolReturnType
from typedefs import AgentCallback
from typing import Any


class SubAgentProfile(pydantic.BaseModel):
    type: str
    description: str
    core_system_prompt: str
    tools: list[str] | None

_SUB_AGENTS = [
    SubAgentProfile(
        type="default-agent",
        description="General-purpose agent for multi-step codebase search and problem solving.",
        core_system_prompt=dedent("""\
            You are a general-purpose agent who has been assigned some work to do.
            You should complete that work and finished with a detailed writeup of what you've done.            
            The user will only see that detailed writeup; they won't see your step-by-step progress.
            Therefore be sure to include everything relevant in that detailed writeup."""),
        tools=None  # Can use all tools
    ),
    SubAgentProfile(
        type="code-reviewer",
        description="Strict code reviewer. Looks for logic flaws, edge cases, and missing invariants.",
        core_system_prompt="You are a strict read-only code reviewer. Evaluate code with skepticism and rigor. Look for flaws, bugs, and loopholes.",
        tools=["Read", "Bash"]  # Can only read and run test commands!
    )
]

def get_subagent_system_prompt(profile: SubAgentProfile) -> str:
    """Builds the specific system prompt for a sub-agent."""
    return f"{profile.core_system_prompt}\n\n{get_environment_details()}"

async def _task_impl(kwargs: dict[str, Any]) -> ToolReturnType:
    prompt = kwargs.get("prompt")
    description = kwargs.get("description", "Delegated sub-task")
    subagent_type = kwargs.get("subagent_type", "default-agent")
    
    if not prompt:
        return "Error: prompt is required."

    # Find the requested profile
    profile = next((sa for sa in _SUB_AGENTS if sa.type == subagent_type), None)
    if not profile:
        available = ", ".join(sa.type for sa in _SUB_AGENTS)
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
        user_content=f"{claude_text}{prompt}"
    )

def register_tasks_tools(registry: ToolRegistry):
    registry.register(
        name="Task",
        description=dedent("""\
        This tool launches a sub-agent for handling ambiguous, complex or multi-step tasks.
        These kinds of sub-agents are available. Use the subagent_type parameter to say which one you want.
        """) + \
        "\n".join([f"- {subagent.type}: {subagent.description} (Tools: {', '.join(subagent.tools) if subagent.tools else '*'})" for subagent in _SUB_AGENTS]) + \
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
        func=_task_impl
    )