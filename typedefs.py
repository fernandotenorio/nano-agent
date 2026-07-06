from __future__ import annotations
import pydantic
from typing import Literal, Any, Union

# ---------------------------------------------------------
# Content Blocks
# ---------------------------------------------------------

class TextMessageContent(pydantic.BaseModel):
    type: Literal["text"] = "text"
    text: str

class ToolUseMessageContent(pydantic.BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any]

class ToolResultMessageContent(pydantic.BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str | list[TextMessageContent]
    is_error: bool = False

# ---------------------------------------------------------
# Messages
# ---------------------------------------------------------

class SystemMessage(pydantic.BaseModel):
    role: Literal["system"] = "system"
    content: str | list[TextMessageContent]

class UserMessage(pydantic.BaseModel):
    role: Literal["user"] = "user"
    content: str | list[Union[TextMessageContent, ToolResultMessageContent]]

class AssistantMessage(pydantic.BaseModel):
    role: Literal["assistant"] = "assistant"
    content: list[Union[TextMessageContent, ToolUseMessageContent]]

# A helpful alias for the Transcript
Message = Union[SystemMessage, UserMessage, AssistantMessage]

# ---------------------------------------------------------
# Tool Callbacks (We will expand these later)
# ---------------------------------------------------------

class BashCallback(pydantic.BaseModel):
    """Returned by the Bash tool to instruct the loop to run a subprocess."""
    kind: Literal["bash_callback"] = "bash_callback"
    command: str
    callback_description: str | None = None
    timeout: float = 120.0  # seconds

class AgentCallback(pydantic.BaseModel):
    kind: Literal["agent_callback"] = "agent_callback"
    subagent_type: str
    callback_description: str
    tools: list[str] | None  # If None, subagent can use all tools.
    system_content: str
    user_content: str