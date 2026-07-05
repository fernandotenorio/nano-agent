from __future__ import annotations
import pydantic
from pathlib import Path
from textwrap import dedent
from typing import Literal, Callable, Awaitable
from typedefs import TextMessageContent

# ---------------------------------------------------------
# Event Context Models
# ---------------------------------------------------------

class UserPromptEvent(pydantic.BaseModel):
    prompt: str
    is_first_prompt: bool  # The orchestrator tells us if this is the start
    context_pre: list[TextMessageContent] = []
    context_post: list[TextMessageContent] = []
    block: bool = False
    block_reason: str = ""

class PreToolUseEvent(pydantic.BaseModel):
    tool_name: str
    tool_input: dict
    decision: Literal["allow", "deny"] = "allow"
    deny_reason: str = ""

class PostToolUseEvent(pydantic.BaseModel):
    tool_name: str
    tool_input: dict
    tool_output: str | list[TextMessageContent]
    additional_context: list[TextMessageContent] = []

# Type aliases for Async Hook Callbacks
UserPromptHook = Callable[[UserPromptEvent], Awaitable[UserPromptEvent]]
PreToolHook = Callable[[PreToolUseEvent], Awaitable[PreToolUseEvent]]
PostToolHook = Callable[[PostToolUseEvent], Awaitable[PostToolUseEvent]]

# ---------------------------------------------------------
# Hook Manager
# ---------------------------------------------------------

class HookManager:
    def __init__(self):
        self._user_prompt_hooks: list[UserPromptHook] = []
        self._pre_tool_hooks: list[PreToolHook] = []
        self._post_tool_hooks: list[PostToolHook] = []

    def register_user_prompt(self, hook: UserPromptHook):
        self._user_prompt_hooks.append(hook)

    def register_pre_tool(self, hook: PreToolHook):
        self._pre_tool_hooks.append(hook)

    def register_post_tool(self, hook: PostToolHook):
        self._post_tool_hooks.append(hook)

    # --- Trigger Methods ---

    async def trigger_user_prompt(self, prompt: str, is_first_prompt: bool) -> UserPromptEvent:
        event = UserPromptEvent(prompt=prompt, is_first_prompt=is_first_prompt)
        for hook in self._user_prompt_hooks:
            event = await hook(event)
            if event.block:
                break
        return event
    
    async def trigger_pre_tool(self, name: str, inputs: dict) -> PreToolUseEvent:
        event = PreToolUseEvent(tool_name=name, tool_input=inputs)
        for hook in self._pre_tool_hooks:
            event = await hook(event)
            if event.decision == "deny":
                break
        return event

    async def trigger_post_tool(self, name: str, inputs: dict, output: str) -> PostToolUseEvent:
        event = PostToolUseEvent(tool_name=name, tool_input=inputs, tool_output=output)
        for hook in self._post_tool_hooks:
            event = await hook(event)
        return event

# ---------------------------------------------------------
# Sample Built-in Hook: CLAUDE.md Injector
# ---------------------------------------------------------

async def initial_setup_hook(event: UserPromptEvent) -> UserPromptEvent:
    """Injects CLAUDE.md on the very first user prompt."""

    # Fast exit: No disk I/O if we are already mid-conversation!
    if not event.is_first_prompt:
        return event

    claude_path = Path("CLAUDE.md")
    if claude_path.exists():
        content = claude_path.read_text(encoding="utf-8")

        reminder = dedent(f'''
        <system-reminder>
        You can use the following context as you answer the user's questions:

        {content}
        
        </system-reminder>''')
        event.context_pre.append(TextMessageContent(text=reminder))
        
    return event