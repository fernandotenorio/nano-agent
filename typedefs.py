from __future__ import annotations
import pydantic
import uuid
from typing import Literal, Any, Union

# ---------------------------------------------------------
# Content Blocks
# ---------------------------------------------------------

class TextMessageContent(pydantic.BaseModel):
    type: Literal["text"] = "text"
    text: str

class ThinkingMessageContent(pydantic.BaseModel):
    type: Literal["thinking"] = "thinking"
    thinking: str
    signature: str = ""

class ToolUseMessageContent(pydantic.BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: Any

class ToolResultMessageContent(pydantic.BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str | list[TextMessageContent]
    is_error: bool = False

    # Accounting for an LLM the tool ran inside its own implementation. Carried
    # on the transcript rather than only in memory so that a resumed session
    # can rebuild the same usage ledger. All optional: transcripts written
    # before this existed must still load.
    tool_name: str | None = None
    usage: dict[str, Any] | None = None
    internal_model: str | None = None

class ToolFailure(pydantic.BaseModel):
    """Returned by a tool to explicitly signal an error state to the agent loop."""
    error_message: str
    ui_summary: str | None = None

class ToolResult(pydantic.BaseModel):
    """Wraps a tool's raw payload for the LLM together with a short, human-friendly
    summary for the terminal UI (e.g. 'Read src/agent.py (487 lines)')."""
    content: str | list[TextMessageContent]
    ui_summary: str | None = None

    # Set by a tool that called an LLM of its own, so the loop can bill those
    # tokens to the tool rather than losing them.
    usage: dict[str, Any] | None = None
    internal_model: str | None = None

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
    id: str = pydantic.Field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:8]}")
    content: list[Union[TextMessageContent, ThinkingMessageContent, ToolUseMessageContent]]
    type: Literal["message"] = "message"
    model: str | None = None
    stop_reason: str | None = None
    stop_sequence: str | None = None
    usage: dict[str, Any] | None = None

    # What the session asked for ("ollama/gemma3:12b"), as opposed to `model`,
    # which is whatever the provider echoed back ("gemma3:12b"). Usage is keyed
    # on this so a resumed transcript groups with a live one, and so the
    # provider can still be read off the string.
    request_model: str | None = None


# A helpful alias for the Transcript
Message = Union[SystemMessage, UserMessage, AssistantMessage]

# ---------------------------------------------------------
# Tool Callbacks (We will expand these later)
# ---------------------------------------------------------

class ShellCallback(pydantic.BaseModel):
    """Returned by the Shell tool to instruct the loop to run a subprocess."""
    kind: Literal["shell_callback"] = "shell_callback"
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

class PlanApprovalCallback(pydantic.BaseModel):
    kind: Literal["plan_approval"] = "plan_approval"
    plan_summary: str