# adapter.py

from __future__ import annotations
import os
import sys
import time
import json
import uuid
import asyncio
import hashlib
import itertools
from pathlib import Path
from typing import Any, Awaitable, TypeVar, Union

from typedefs import (
    AssistantMessage, SystemMessage, UserMessage, Message,
    TextMessageContent, ThinkingMessageContent,
    ToolUseMessageContent, ToolResultMessageContent
)

T = TypeVar("T")

async def spinner(awaitable: Awaitable[T]) -> T:
    """Displays a terminal spinner during API calls."""
    if not sys.stdout.isatty():
        return await awaitable
    t0, spin, task = time.time(), itertools.cycle("🌑🌒🌓🌔🌕🌖🌗🌘"), asyncio.ensure_future(awaitable)
    try:
        while True:
            sys.stdout.write(f"\r{next(spin)} {time.time() - t0:.1f}s")
            sys.stdout.flush()
            try:
                return await asyncio.wait_for(asyncio.shield(task), timeout=0.2)
            except asyncio.TimeoutError:
                continue
    finally:
        sys.stdout.write("\r" + " " * 30 + "\r")


def format_tool_desc(tool: Any) -> dict[str, Any]:
    """Ensures tool definitions are formatted as OpenAI function specifications."""
    if isinstance(tool, dict):
        if "type" in tool and "function" in tool:
            return tool
        elif "name" in tool and "parameters" in tool:
            return {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {})
                }
            }
        return tool
    elif hasattr(tool, "name") and hasattr(tool, "inputSchema"):  # MCP Tool object
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": getattr(tool, "description", ""),
                "parameters": tool.inputSchema
            }
        }
    return tool


def to_openai_message(message: Message, add_cache_control: bool) -> list[dict[str, Any]]:
    """
    Translates internal Pydantic message structures into standard OpenAI/LiteLLM format.
    """
    if isinstance(message, SystemMessage):
        content_str = message.content if isinstance(message.content, str) else "\n".join(
            c.text for c in message.content if isinstance(c, TextMessageContent)
        )
        sys_msg: dict[str, Any] = {"role": "system", "content": content_str}
        if add_cache_control:
            sys_msg["content"] = [{"type": "text", "text": content_str, "cache_control": {"type": "ephemeral"}}]
        return [sys_msg]

    elif isinstance(message, AssistantMessage):
        tool_calls = []
        text_parts = []
        for item in message.content:
            if isinstance(item, ToolUseMessageContent):
                tool_calls.append({
                    "id": item.id,
                    "type": "function",
                    "function": {
                        "name": item.name,
                        "arguments": item.input if isinstance(item.input, str) else json.dumps(item.input)
                    }
                })
            elif isinstance(item, TextMessageContent):
                text_parts.append(item.text)

        res: dict[str, Any] = {"role": "assistant"}
        combined_text = "\n".join(text_parts) if text_parts else None
        if combined_text:
            res["content"] = combined_text
        if tool_calls:
            res["tool_calls"] = tool_calls
        return [res]

    elif isinstance(message, UserMessage):
        message_content = [TextMessageContent(text=message.content)] if isinstance(message.content, str) else message.content
        tool_results = [item for item in message_content if isinstance(item, ToolResultMessageContent)]
        text_blocks = [item for item in message_content if isinstance(item, TextMessageContent)]

        out_messages: list[dict[str, Any]] = []

        # 1. Convert tool results into OpenAI 'tool' role messages
        for tr in tool_results:
            tr_content = tr.content if isinstance(tr.content, str) else "\n".join(
                c.text for c in tr.content if isinstance(c, TextMessageContent)
            )
            out_messages.append({
                "role": "tool",
                "tool_call_id": tr.tool_use_id,
                "content": tr_content
            })

        # 2. Convert user text blocks into 'user' role message
        if text_blocks:
            combined_user_text = "\n".join(b.text for b in text_blocks)
            user_msg: dict[str, Any] = {"role": "user", "content": combined_user_text}
            if add_cache_control:
                user_msg["content"] = [{
                    "type": "text",
                    "text": combined_user_text,
                    "cache_control": {"type": "ephemeral"}
                }]
            out_messages.append(user_msg)

        if not out_messages:
            out_messages.append({"role": "user", "content": ""})

        return out_messages

    return []


def parse_assistant_response(response: Any) -> AssistantMessage:
    """Parses a LiteLLM ModelResponse object back into an AssistantMessage."""
    import litellm.types.utils

    if not isinstance(response, litellm.types.utils.ModelResponse):
        raise TypeError(f"Expected ModelResponse, got {type(response)}")
    
    choice = response.choices[0]
    message = choice.message
    usage = getattr(response, 'usage', None)
    
    content: list[TextMessageContent | ThinkingMessageContent | ToolUseMessageContent] = []

    if message.content:
        content.append(TextMessageContent(type="text", text=message.content))

    # Thinking content: prefer structured thinking_blocks (they carry
    # signatures); only fall back to reasoning_content when no blocks exist.
    # Appending both would duplicate the same thinking text, since providers
    # like Anthropic populate both fields with identical content.
    thinking_blocks = [
        t for t in (getattr(message, 'thinking_blocks', None) or [])
        if isinstance(t, dict) and 'thinking' in t
    ]
    if thinking_blocks:
        for thinking in thinking_blocks:
            content.append(ThinkingMessageContent(
                type="thinking",
                thinking=thinking['thinking'],
                signature=thinking.get('signature', '')
            ))
    elif getattr(message, 'reasoning_content', None):
        content.append(ThinkingMessageContent(type="thinking", thinking=message.reasoning_content, signature=""))

    for tool_call in (getattr(message, 'tool_calls', None) or []):
        tool_args = tool_call.function.arguments
        if isinstance(tool_args, str):
            try:
                tool_args = json.loads(tool_args)
            except Exception:
                tool_args = {"raw": tool_args}
        
        content.append(ToolUseMessageContent(
            type="tool_use",
            id=tool_call.id,
            name=str(tool_call.function.name),
            input=tool_args
        ))

    usage_dict = None
    if isinstance(usage, litellm.types.utils.Usage):
        usage_dict = usage.model_dump()

    return AssistantMessage(
        role="assistant",
        id=getattr(response, 'id', f"msg_{uuid.uuid4().hex[:8]}"),
        content=content,
        type="message",
        model=str(getattr(response, 'model', 'unknown')),
        stop_reason="tool_use" if (hasattr(message, 'tool_calls') and message.tool_calls) else getattr(choice, 'finish_reason', 'end_turn'),
        usage=usage_dict,
    )


async def acompletion(
    model: str, 
    tools: list[Any], 
    messages: list[Message]
) -> AssistantMessage:
    """
    Executes an async completion call via LiteLLM supporting Cloud APIs and local Ollama.
    """
    import litellm  # Lazy import for startup performance

    is_anthropic = model.startswith("anthropic/") or model.startswith("claude")
    is_ollama = model.startswith("ollama/") or model.startswith("ollama_chat/")

    # Stable per-directory cache key. Python's built-in hash() is randomized
    # per process, which would defeat session affinity across restarts.
    openai_prompt_cache_key = hashlib.sha256(str(Path.cwd()).encode("utf-8")).hexdigest()

    # Build formatted OpenAI messages list
    formatted_messages: list[dict[str, Any]] = []
    for i, msg in enumerate(messages):
        add_cache = is_anthropic and (i <= 1 or i == len(messages) - 1)
        formatted_messages.extend(to_openai_message(msg, add_cache_control=add_cache))

    formatted_tools = [format_tool_desc(t) for t in tools] if tools else None

    # Base LiteLLM call arguments
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": formatted_messages,
    }

    # Only pass tools if available (prevents Ollama empty tools validation errors)
    if formatted_tools:
        kwargs["tools"] = formatted_tools

    # Handle Ollama host configuration
    if is_ollama:
        kwargs["api_base"] = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
    else:
        kwargs["user"] = openai_prompt_cache_key

    # Call LLM through spinner
    response = await spinner(litellm.acompletion(**kwargs))
    return parse_assistant_response(response)