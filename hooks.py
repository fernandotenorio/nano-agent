from __future__ import annotations
import pydantic
from pathlib import Path
from textwrap import dedent
from typing import Literal, Callable, Awaitable, Sequence
from typedefs import (
    Message,
    TextMessageContent,
    ToolResultMessageContent,
    UserMessage,
)
from capabilities import Capabilities, model_warnings
from config import AppConfig
from context import gather_context_files
from sessioncontext import AgentPolicy, AgentMode
from ui.base import UI
from ui.null_ui import NullUI

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

    async def trigger_post_tool(self, name: str, inputs: dict, output: str | list[TextMessageContent]) -> PostToolUseEvent:
        event = PostToolUseEvent(tool_name=name, tool_input=inputs, tool_output=output)
        for hook in self._post_tool_hooks:
            event = await hook(event)
        return event

# ---------------------------------------------------------
# Built-in Hook: Shell Command Confirmation Gate
# ---------------------------------------------------------

async def shell_confirmation_hook(event: PreToolUseEvent, ui: UI | None = None) -> PreToolUseEvent:
    """Requires explicit user confirmation before any Shell command runs.

    Shell is the one tool the workspace boundary cannot inspect: a command
    can read or write anywhere on disk. This gate puts the user in the loop
    for every command. Fails closed: the default NullUI (and any UI on
    EOF/interrupt) denies the command.
    """
    if event.tool_name != "Shell":
        return event

    command = str(event.tool_input.get("command", "")).strip()
    description = event.tool_input.get("description")

    decision = await (ui or NullUI()).confirm_shell(command, description)

    if decision.approved:
        return event

    event.decision = "deny"
    event.deny_reason = "User denied permission to run this shell command."
    if decision.deny_reason:
        event.deny_reason += f" Reason: {decision.deny_reason}"

    return event


# ---------------------------------------------------------
# Built-in Hook: Agends.md Context Injector
# ---------------------------------------------------------

async def initial_setup_hook(
    event: UserPromptEvent, 
    app_config: AppConfig, 
    root: Path, 
    cwd: Path
) -> UserPromptEvent:
    """Injects AGENTS.md context on the very first user prompt."""

    # Fast exit: No disk I/O if we are already mid-conversation!
    if not event.is_first_prompt:
        return event

    context_text = gather_context_files(app_config, root, cwd)
    
    if context_text:
        reminder = dedent(f'''
        <system-reminder>
        You can use the following project context as you answer the user's questions:

        {context_text}
        
        </system-reminder>''')
        event.context_pre.append(TextMessageContent(text=reminder))
        
    return event


# ---------------------------------------------------------
# Built-in Hook: Degraded Capability Notice
# ---------------------------------------------------------

async def capabilities_hook(
    event: UserPromptEvent,
    capabilities: Capabilities,
) -> UserPromptEvent:
    """Warns about degraded capabilities on the very first user prompt.

    The system prompt carries the same warning, but only for a fresh session:
    `build_system_prompt` runs once, when the transcript is empty. A resumed
    transcript therefore describes the machine as it was, which may no longer be
    true — so the warning is repeated here, where it is recomputed every run.
    """
    if not event.is_first_prompt:
        return event

    warnings = model_warnings(capabilities)
    if not warnings:
        return event

    reminder = dedent("""
    <system-reminder>
    {warnings}
    </system-reminder>""").format(warnings="\n\n".join(warnings))

    event.context_pre.append(TextMessageContent(text=reminder))

    return event


# ---------------------------------------------------------
# Built-in Hook: Plan mode Injector
# ---------------------------------------------------------

MODE_ANNOUNCEMENTS: dict[AgentMode, str] = {
    AgentMode.PLAN: "You are now in PLAN MODE.",
    AgentMode.BUILD: "You are now in BUILD MODE.",
}
"""The sentence that opens each mode reminder.

It doubles as the marker `last_notified_mode` looks for when rebuilding the
policy of a resumed session, so the wording exists in exactly one place: a
reminder the transcript cannot be read back out of would leave a resumed
session unable to tell what the model has already been told.
"""

_MODE_INSTRUCTIONS: dict[AgentMode, str] = {
    AgentMode.PLAN: (
        "You only have access to read-only tools.\n"
        "Investigate the codebase as needed. When you are ready, you MUST use "
        "the `SubmitPlan` tool to propose your plan to the user."
    ),
    AgentMode.BUILD: "You have full access to write and shell tools.",
}

PLAN_ACCEPTED_TO_BUILD = (
    "SUCCESS: User accepted the plan and switched to BUILD mode. "
    "You now have access to write tools. Proceed with execution."
)
"""Reported by the loop when an approved plan moves the session into BUILD.

Kept with the mode wording because it is the one mode transition announced by
a tool result rather than by a reminder, and `last_notified_mode` has to
recognise it as such.
"""


def mode_reminder(mode: AgentMode) -> str:
    """Builds the reminder that announces `mode` to the model."""
    return dedent("""
    <system-reminder>
    {announcement} {instructions}
    </system-reminder>""").format(
        announcement=MODE_ANNOUNCEMENTS[mode],
        instructions=_MODE_INSTRUCTIONS[mode],
    )


async def agent_mode_hook(
    event: UserPromptEvent, 
    policy: AgentPolicy
) -> UserPromptEvent:
    """Injects a system reminder only when the agent transitions between modes."""

    if policy.mode == policy.notified_mode:
        return event

    event.context_pre.append(TextMessageContent(text=mode_reminder(policy.mode)))

    # Mark as notified so we don't spam the LLM on subsequent messages
    policy.notified_mode = policy.mode

    return event


# ---------------------------------------------------------
# Recovering the mode of a resumed session
# ---------------------------------------------------------

def _announced_mode(
    block: TextMessageContent | ToolResultMessageContent,
) -> AgentMode | None:
    """The mode this content block told the model it was in, if any."""
    if isinstance(block, TextMessageContent):
        # The reminder wrapper is what separates our announcement from a user
        # who happened to type the same sentence: injected context and typed
        # text arrive as the same kind of block on the same UserMessage.
        if "<system-reminder>" not in block.text:
            return None

        for mode, announcement in MODE_ANNOUNCEMENTS.items():
            if announcement in block.text:
                return mode

        return None

    if isinstance(block, ToolResultMessageContent) and not block.is_error:
        text = (
            block.content
            if isinstance(block.content, str)
            else "\n".join(item.text for item in block.content)
        )

        if PLAN_ACCEPTED_TO_BUILD in text:
            return AgentMode.BUILD

    return None


def last_notified_mode(messages: Sequence[Message]) -> AgentMode | None:
    """The mode the transcript most recently told the model about.

    `notified_mode` is a fact about the conversation, not about the process
    that happened to host it, so a resumed session reads it back out of the
    transcript instead of starting from None. Starting from None is what used
    to repeat the mode reminder on the first prompt after a resume.

    Only announcements count, because only they leave a trace. A mode switch
    the model was never told about — `/plan` typed with the session closed
    before the next prompt — is not recoverable, and such a session resumes in
    BUILD.
    """
    for message in reversed(messages):
        if not isinstance(message, UserMessage) or not isinstance(message.content, list):
            continue

        for block in reversed(message.content):
            mode = _announced_mode(block)
            if mode is not None:
                return mode

    return None


def restore_policy(messages: Sequence[Message]) -> AgentPolicy:
    """Rebuilds the policy of a resumed session from its transcript.

    The mode is restored alongside the notification state: a user who left a
    session in PLAN mode deliberately withheld write and shell access, and
    resuming is not the moment to hand it back.
    """
    notified = last_notified_mode(messages)

    return AgentPolicy(mode=notified or AgentMode.BUILD, notified_mode=notified)