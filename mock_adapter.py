# mock_adapter.py

import uuid
from typing import Any
from typedefs import (
    AssistantMessage,
    TextMessageContent,
    ToolUseMessageContent,
    ToolResultMessageContent,
    UserMessage,
    Message
)

async def acompletion(model: str, tools: list[dict[str, Any]], messages: list[Message]) -> AssistantMessage:
    
    # 1. Get the last user message
    last_user_msg = None
    for msg in reversed(messages):
        if isinstance(msg, UserMessage):
            last_user_msg = msg
            break
            
    if not last_user_msg:
        return AssistantMessage(content=[TextMessageContent(text="Hello!")])

    # 2. Check if the last message contains Tool Results
    # This means the loop just executed a tool and is asking us to summarize it.
    if isinstance(last_user_msg.content, list):
        tool_results = [c for c in last_user_msg.content if isinstance(c, ToolResultMessageContent)]
        if tool_results:
            result = tool_results[0]
            return AssistantMessage(content=[
                TextMessageContent(text=f"[Mock LLM] I successfully used the tool {result.tool_use_id}'. The result was: {result.content[:50]}...")
            ])

    # 3. Otherwise, it's a standard user text prompt
    last_user_text = ""
    if isinstance(last_user_msg.content, str):
        last_user_text = last_user_msg.content
    else:
        texts = [c.text for c in last_user_msg.content if getattr(c, "type", None) == "text"]
        last_user_text = "\n".join(texts)
            
    last_user_text = last_user_text.lower().strip()
    call_id = f"call_{uuid.uuid4().hex[:8]}"
    
    # 4. Heuristics for triggering tools
    if last_user_text.startswith("shell "):
        command = last_user_text[5:].strip()
        return AssistantMessage(content=[
            TextMessageContent(text="I will run that shell command for you."),
            ToolUseMessageContent(id=call_id, name="Shell", input={"command": command})
        ])
        
    elif last_user_text.startswith("read "):
        file_path = last_user_text[5:].strip()
        return AssistantMessage(content=[
            TextMessageContent(text=f"Let me read the contents of {file_path}."),
            ToolUseMessageContent(id=call_id, name="Read", input={"file_path": file_path})
        ])
    elif last_user_text.startswith("write "):
        file_path = last_user_text[6:].strip()
        content = "Some random text."
        return AssistantMessage(content=[
            TextMessageContent(text=f"Writing to {file_path}."),
            ToolUseMessageContent(id=call_id, name="Write", input={"file_path": file_path, "content": content})
        ])
    elif last_user_text.startswith("task "):
        task_prompt = last_user_text[5:].strip()
        return AssistantMessage(content=[
            TextMessageContent(text="I'll create a sub-agent to handle this complex task."),
            ToolUseMessageContent(id=call_id, name="Task", input={
                "description": "Delegated sub-task",
                "prompt": task_prompt,
                "subagent_type": "default-agent"
            })
        ])
        
    # 5. Conversational text response
    return AssistantMessage(content=[
        TextMessageContent(text=f"[Mock LLM] You said:\n'{last_user_text}'.\nI don't see a tool command, so I'm just chatting!")
    ])