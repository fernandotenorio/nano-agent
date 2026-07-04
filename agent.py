import asyncio
import sys
import argparse
import uuid
from pathlib import Path
from datetime import datetime

from prompts import get_system_prompt
from typedefs import TextMessageContent, ToolResultMessageContent, UserMessage, SystemMessage, BashCallback, AgentCallback
from mock_adapter import acompletion
from transcript import Transcript
from hooks import HookManager, initial_setup_hook
from tools import create_core_registry, ToolRegistry


async def handle_bash(callback: BashCallback) -> tuple[str, bool]:
    """Executes a bash command natively and captures stdout/stderr."""
    print(f"  $ {callback.command}")
    process = await asyncio.create_subprocess_shell(
        callback.command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()
    
    exit_code = process.returncode
    output = ""
    if stdout:
        output += stdout.decode('utf-8', errors='replace')
    if stderr:
        output += "\n" + stderr.decode('utf-8', errors='replace')
        
    is_error = exit_code != 0
    return output.strip() or "Command completed with no output.", is_error


async def handle_subagent(callback: AgentCallback, parent_registry: ToolRegistry, hooks: HookManager, parent_transcript_path: Path) -> tuple[str, bool]:
    """Spins up a recursive sub-agent loop."""
    print(f"  >> [Sub-Agent '{callback.subagent_type}' started] Task: {callback.callback_description}")
    
    # Put the sub-agent transcript in the exact same directory as the parent transcript
    # Naming convention: {parent_name}_{subagent_type}_{short_uuid}.jsonl
    parent_dir = parent_transcript_path.parent
    sub_id = uuid.uuid4().hex[:6]
    sub_transcript_path = parent_dir / f"{parent_transcript_path.stem}_{callback.subagent_type}_{sub_id}.jsonl"
    
    sub_transcript = Transcript(sub_transcript_path)
    print(f"  >> [Sub-Agent log: {sub_transcript_path}]")
    
    # 1. Inject the Sub-Agent's specific System Prompt
    sub_transcript.append(SystemMessage(content=callback.system_content))
    
    # 2. Inject the Task instructions as the first User Message
    sub_transcript.append(UserMessage(content=[TextMessageContent(text=callback.user_content)]))
    
    # 3. Filter tools if the profile restricts them
    sub_registry = parent_registry
    if callback.tools is not None:
        sub_registry = parent_registry.clone_filtered(callback.tools)
    
    await run_agentic_loop(sub_transcript, sub_registry, hooks)
    
    print(f"  >> [Sub-Agent '{callback.subagent_type}' finished]")
    return "Sub-agent completed its task. See its final output in the transcript.", False

async def run_agentic_loop(transcript: Transcript, registry: ToolRegistry, hooks: HookManager):
    """
    The core driver. Repeatedly calls the LLM. If the LLM uses a tool, it executes it,
    appends the result, and loops again. Stops when the LLM replies with pure text.
    """
    while True:
        schemas = registry.get_all_schemas()
        response = await acompletion("mock-model", schemas, transcript.messages)
        transcript.append(response)

        texts = [c for c in response.content if getattr(c, "type", None) == "text"]
        tool_uses = [c for c in response.content if getattr(c, "type", None) == "tool_use"]

        for text_block in texts:
            print(f"< {text_block.text}")

        if not tool_uses:
            break

        tool_results_content = []
        for tu in tool_uses:
            print(f"  >> Calling {tu.name}(...)")
            
            # --- Pre-Tool Hook ---
            pre_event = await hooks.trigger_pre_tool(tu.name, tu.input)
            if pre_event.decision == "deny":
                print(f"  >> [BLOCKED by Hook]: {pre_event.deny_reason}")
                result_str, is_error = f"Tool blocked: {pre_event.deny_reason}", True
            else:
                # --- Invoke Native Tool ---
                raw_result = await registry.invoke(tu.name, tu.input)
                is_error = False

                if isinstance(raw_result, BashCallback):
                    result_str, is_error = await handle_bash(raw_result)
                elif isinstance(raw_result, AgentCallback):
                    # We now pass the current transcript's file path down to the sub-agent handler
                    result_str, is_error = await handle_subagent(raw_result, registry, hooks, transcript.file_path)
                else:
                    result_str = str(raw_result)
                    if result_str.startswith("Error:"):
                        is_error = True

            # --- Post-Tool Hook ---
            post_event = await hooks.trigger_post_tool(tu.name, tu.input, result_str)
            
            tool_results_content.append(ToolResultMessageContent(
                tool_use_id=tu.id,
                content=post_event.tool_output,
                is_error=is_error
            ))
            tool_results_content.extend(post_event.additional_context)

        transcript.append(UserMessage(content=tool_results_content))


def get_transcript_path(resume_arg: str | None) -> Path:
    """Determines where to load/save the transcript file."""
    if resume_arg:
        path = Path(resume_arg).expanduser().resolve()
        if not path.exists():
            print(f"Warning: Provided resume path '{path}' does not exist. It will be created.")
        return path
    
    # Default behavior: create a hidden `.agent/transcripts/` folder in the current directory
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    default_dir = Path.cwd() / ".agent" / "transcripts"
    default_dir.mkdir(parents=True, exist_ok=True)
    return default_dir / f"{timestamp}.jsonl"


async def main():
    # Parse Command Line Arguments
    parser = argparse.ArgumentParser(description="Native Code Agent")
    parser.add_argument("--resume", type=str, help="Path to an existing .jsonl transcript to resume")
    args = parser.parse_args()

    # Determine transcript file path
    transcript_file = get_transcript_path(args.resume)
    print(f"[TRANSCRIPT: {transcript_file}]")
    
    # Initialize State
    registry = create_core_registry()
    hooks = HookManager()
    hooks.register_user_prompt(initial_setup_hook)
    
    # Load (or create) the main transcript
    transcript = Transcript(transcript_file)

    # System Prompt injection (only if transcript is brand new)
    if len(transcript.messages) == 0:
        transcript.append(get_system_prompt())
    
    print("Welcome to Native Code Agent (Type '/quit' to exit)")
    
    while True:
        try:
            user_input = input("\n> ")
            if user_input.strip().lower() in ["/quit", "/exit"]:
                break
            if not user_input.strip():
                continue

            # Fire User Hooks
            is_first_prompt = not any(isinstance(m, UserMessage) for m in transcript.messages)
            event = await hooks.trigger_user_prompt(user_input, is_first_prompt)
            
            if event.block:
                print(f"[BLOCKED]: {event.block_reason}")
                continue

            # 3. Assemble the payload: [ PRE, PROMPT, POST ]
            message_content = [
                *event.context_pre,
                TextMessageContent(text=event.prompt),
                *event.context_post
            ]
            
            transcript.append(UserMessage(content=message_content))
            await run_agentic_loop(transcript, registry, hooks)
            
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

if __name__ == "__main__":
    asyncio.run(main())