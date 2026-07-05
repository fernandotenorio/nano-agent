import pydantic
from pathlib import Path
from textwrap import dedent
from typedefs import SystemMessage
from environment import get_environment_details

_MAIN_AGENT_PROMPT = """\
You are an interactive coding agent designed to help users with their software engineering tasks.
You can read files, execute bash commands, and orchestrate sub-agents.
- Be concise and direct in your responses.
- Always use tools when you need to gather information or make changes.
"""

def get_system_prompt() -> SystemMessage:
    """Builds the complete system prompt."""
    
    # Base identity
    base_prompt = _MAIN_AGENT_PROMPT
    
    # Look for user-defined overrides in .agent/system.md
    custom_sys_path = Path.cwd() / ".agent" / "SYSTEM.md"
    if custom_sys_path.exists():
        custom_content = custom_sys_path.read_text(encoding="utf-8")
        base_prompt += f"\n\n<custom-instructions>\n{custom_content}\n</custom-instructions>"

    base_prompt += f"\n\n{get_environment_details()}"
    return SystemMessage(content=base_prompt)