import pydantic
from pathlib import Path
from textwrap import dedent
from typedefs import SystemMessage
from environment import get_environment_details

MAIN_AGENT_PROMPT = """\
You are an interactive coding agent designed to help users with their software engineering tasks.
You can read files, execute bash commands, and orchestrate sub-agents.
- Be concise and direct in your responses.
- Always use tools when you need to gather information or make changes.
"""

# --- Sub-Agent Definitions ---
class SubAgentProfile(pydantic.BaseModel):
    type: str
    description: str
    core_system_prompt: str
    tools: list[str] | None

SUB_AGENTS = [
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

def get_system_prompt() -> SystemMessage:
    """Builds the complete system prompt."""
    
    # Base identity
    base_prompt = MAIN_AGENT_PROMPT
    
    # Look for user-defined overrides in .agent/system.md
    custom_sys_path = Path.cwd() / ".agent" / "SYSTEM.md"
    if custom_sys_path.exists():
        custom_content = custom_sys_path.read_text(encoding="utf-8")
        base_prompt += f"\n\n<custom-instructions>\n{custom_content}\n</custom-instructions>"

    # Inject dynamic environment variables
    base_prompt += f"\n\n{get_environment_details()}"
    return SystemMessage(content=base_prompt)

def get_subagent_system_prompt(profile: SubAgentProfile) -> str:
    """Builds the specific system prompt for a sub-agent."""
    return f"{profile.core_system_prompt}\n\n{get_environment_details()}"