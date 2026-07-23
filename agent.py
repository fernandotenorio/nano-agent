import asyncio
import os
import sys
import argparse
import uuid
from functools import partial
from pathlib import Path
from datetime import datetime
import logging

from config import AppConfig, load_app_config
from prompts import build_system_prompt
from sessioncontext import InvocationContext, AgentPolicy, AgentMode

from typing import Literal
from typedefs import (
    TextMessageContent, ToolResultMessageContent, ToolUseMessageContent,
    ToolFailure, UserMessage, SystemMessage, ShellCallback, AgentCallback,
    PlanApprovalCallback
)
from adapter import acompletion
from dotenv import load_dotenv
from transcript import Transcript
from hooks import HookManager, initial_setup_hook, agent_mode_hook
from filestate import file_changes_hook
from tools.registry import ToolRegistry
from tools.core import create_core_registry


load_dotenv(".env.development")
logging.basicConfig(level=logging.WARNING)

async def handle_shell(callback: ShellCallback) -> tuple[str, bool]:
    """
    Executes a shell command natively with timeouts and streaming partial output.
    Returns (output_text, is_error).
    """
    MAX_OUTPUT = 30000
    print(f"  $ {callback.command}")
    
    process = await asyncio.create_subprocess_shell(
        callback.command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    # Concurrently read stdout and stderr up to MAX_OUTPUT bytes
    assert process.stdout is not None and process.stderr is not None
    stdout_parts: list[bytes] = []
    stderr_parts: list[bytes] = []
    
    async def read_stream(stream: asyncio.StreamReader, parts: list[bytes]) -> str:
        while (chunk := await stream.read(8192)) and sum(len(part) for part in parts) < MAX_OUTPUT:
            parts.append(chunk)                
        return b''.join(parts).decode('utf-8', errors='replace')[:MAX_OUTPUT]

    stdout_task = asyncio.create_task(read_stream(process.stdout, stdout_parts))
    stderr_task = asyncio.create_task(read_stream(process.stderr, stderr_parts))

    # Wait for completion or timeout
    exit_code: int | Literal["timeout"]
    try:
        exit_code = await asyncio.wait_for(process.wait(), callback.timeout)
    except asyncio.TimeoutError:
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
        exit_code = "timeout"
    
    stdout, stderr = await asyncio.gather(stdout_task, stderr_task)
    
    # Format the result just like mini_agent
    if exit_code == "timeout":
        is_error = True
        text = f"Command timed out after {callback.timeout:0.1f}s\n{stderr}\n{stdout}"
    elif exit_code == 0:
        is_error = False
        text = f"{stdout}\n{stderr}"
    else:
        is_error = True
        text = f"{stderr}\n{stdout}"
        
    return text.strip() or "Command completed with no output.", is_error


async def handle_subagent(
    callback: AgentCallback,
    ctx: InvocationContext,
    parent_transcript_path: Path,
    model: str
) -> tuple[list[TextMessageContent], bool]:
    """Spins up a recursive sub-agent loop with isolated file-state, registry, and hooks."""
    print(f"  >> [Sub-Agent '{callback.subagent_type}' started] Task: {callback.callback_description}")
    
    parent_dir = parent_transcript_path.parent
    sub_id = uuid.uuid4().hex[:6]
    sub_transcript_path = parent_dir / f"{parent_transcript_path.stem}_{callback.subagent_type}_{sub_id}.jsonl"
    
    sub_transcript = Transcript(sub_transcript_path)
    print(f"  >> [Sub-Agent log: {sub_transcript_path}]")

    # 1. Isolated context: a fresh, empty file-state tracker. The sub-agent
    # must Read files itself before writing them; it never inherits the
    # parent's read history (and vice versa).
    sub_ctx = ctx.clone_for_subagent()

    # 2. Isolated registry whose tool closures are bound to the sub-agent's context
    sub_registry = create_core_registry(sub_ctx)
    if callback.tools is not None:
        sub_registry = sub_registry.clone_filtered(callback.tools)

    # 3. Isolated hooks: AGENTS.md setup + file-change reminders bound to sub_ctx.
    # Deliberately NOT the parent's HookManager: sub-agents always run in BUILD
    # mode (no mode hook), and sharing the parent's hooks would leak pending
    # parent notifications (mode changes, file diffs) into this transcript.
    sub_hooks = HookManager()
    app_config = load_app_config()
    sub_hooks.register_user_prompt(
        partial(initial_setup_hook, app_config=app_config, root=sub_ctx.workspace, cwd=sub_ctx.cwd)
    )
    sub_hooks.register_user_prompt(partial(file_changes_hook, ctx=sub_ctx))

    # 4. Inject the Sub-Agent's specific System Prompt
    sub_transcript.append(SystemMessage(content=callback.system_content))
    
    # 5. Fire the user hooks (This automatically injects AGENTS.md via initial_setup_hook!)
    event = await sub_hooks.trigger_user_prompt(callback.user_content, is_first_prompt=True)
    
    # Handle hook blocks (e.g. if a future hook denies the sub-agent prompt)
    if event.block:
        print(f"  >> [Sub-Agent BLOCKED]: {event.block_reason}")
        return [TextMessageContent(text=f"Sub-agent blocked before starting: {event.block_reason}")], True

    # Assemble the payload just like the main loop
    message_content = [
        *event.context_pre,
        TextMessageContent(text=event.prompt),
        *event.context_post
    ]
    
    # 6. Inject the Task instructions as the first User Message
    sub_transcript.append(UserMessage(content=message_content))

    # Create an isolated policy for the sub-agent and pass it
    sub_policy = AgentPolicy(mode=AgentMode.BUILD)

    # --- Capture the pristine list of blocks ---
    final_blocks = await run_agentic_loop(
        sub_transcript, sub_registry, sub_hooks, model=model, policy=sub_policy, ctx=sub_ctx
    )
    
    print(f"  >> [Sub-Agent '{callback.subagent_type}' finished]")
    return final_blocks, False


async def execute_tool(
    tu: ToolUseMessageContent, 
    registry: ToolRegistry, 
    hooks: HookManager, 
    transcript_path: Path,
    model: str,
    policy: AgentPolicy | None = None,
    ctx: InvocationContext | None = None
) -> list[TextMessageContent | ToolResultMessageContent]:
    """
    Invokes a tool, handles pre/post hooks, and catches execution exceptions.
    Modeled after mini_agent's invoke_tool.
    """
    print(f"  >> Calling {tu.name}(...)")
    
    # 1. Pre-Hook
    pre_event = await hooks.trigger_pre_tool(tu.name, tu.input)
    if pre_event.decision == "deny":
        print(f"  >> [BLOCKED by Hook]: {pre_event.deny_reason}")
        return [ToolResultMessageContent(
            tool_use_id=tu.id,
            content=f"Tool blocked: {pre_event.deny_reason}",
            is_error=True
        )]

    # 2. Invoke Tool with Error Boundaries
    try:
        raw_result = await registry.invoke(tu.name, tu.input)
        
        # Route Native Callbacks
        if isinstance(raw_result, ShellCallback):
            result_output, is_error = await handle_shell(raw_result)
        elif isinstance(raw_result, AgentCallback):
            result_output, is_error = await handle_subagent(raw_result, ctx, transcript_path, model=model)
        elif isinstance(raw_result, PlanApprovalCallback):
            print("\n" + "="*40)
            print(" AI PROPOSED PLAN:")
            print("=" * 40)
            print(raw_result.plan_summary)
            print("=" * 40)
            print("1. Accept plan and switch to BUILD mode")
            print("2. Accept plan but keep in PLAN mode (to refine further)")
            print("3. Reject plan with message")
            
            # Note: For simple CLI, synchronous input() here is fine
            choice = input("\nSelect an option (1/2/3): ").strip()
            
            if choice == "1":
                policy.mode = AgentMode.BUILD
                policy.notified_mode = AgentMode.BUILD # Prevent the hook from double-firing
                result_output = "SUCCESS: User accepted the plan and switched to BUILD mode. You now have access to write tools. Proceed with execution."
                is_error = False

            elif choice == "2":
                result_output = "User accepted the plan, but chose to remain in PLAN mode. Await further user instructions."
                is_error = False

            else:
                reason = input("Enter rejection reason: ").strip()
                result_output = f"REJECTED: User rejected the plan. Reason: {reason}"
                is_error = True
        elif isinstance(raw_result, ToolFailure):
            # EXPLICIT FAILURE
            result_output = raw_result.error_message
            is_error = True
        else:
            # Standard tool output
            result_output = raw_result
            is_error = False                
    except Exception as e:
        # Catch Python exceptions (FileNotFound, JSON decoding, missing keys, etc.)
        result_output = f"Error during tool execution: {str(e)}"
        is_error = True

    # 3. Format Base Result
    content: list[TextMessageContent | ToolResultMessageContent] = [
        ToolResultMessageContent(
            tool_use_id=tu.id,                
            content=result_output,
            is_error=is_error
        )
    ]

    # 4. Post-Hook (Only on success!)
    if not is_error:
        post_event = await hooks.trigger_post_tool(tu.name, tu.input, result_output)
        
        # If the post-hook adds extra context (e.g. file watchers, reminders, or block warnings),
        # they are appended as TextMessageContent next to the ToolResultMessageContent.
        if post_event.additional_context:
            content.extend(post_event.additional_context)

    return content

async def run_agentic_loop(
    transcript: Transcript,
    base_registry: ToolRegistry,
    hooks: HookManager,
    model: str,
    policy: AgentPolicy,
    ctx: InvocationContext
) -> list[TextMessageContent]:
    """
    Returns the pristine list of text blocks from the LLM when no more tools are requested.
    """
    while True:
        # Dynamically evaluate tools on every loop iteration
        current_registry = base_registry.clone_readonly() if policy.mode == AgentMode.PLAN else base_registry
        schemas = current_registry.get_all_schemas()

        response = await acompletion(model, schemas, transcript.messages)
        transcript.append(response)

        texts = [c for c in response.content if getattr(c, "type", None) == "text"]
        tool_uses = [c for c in response.content if getattr(c, "type", None) == "tool_use"]

        for text_block in texts:
            print(f"< {text_block.text}")

        # If LLM doesn't want to use any more tools, break the loop and return texts
        if not tool_uses:
            return texts

        # Execute all tools requested by the LLM
        tool_results_content = []
        for tu in tool_uses:
            # Pass current_registry, policy and ctx down
            result_blocks = await execute_tool(tu, current_registry, hooks, transcript.file_path, model=model, policy=policy, ctx=ctx)
            tool_results_content.extend(result_blocks)

        transcript.append(UserMessage(content=tool_results_content))

def get_transcript_path(app_config: AppConfig, cwd: Path, resume_arg: str | None) -> Path:
    """Determines where to load/save the transcript file."""
    if resume_arg:
        path = Path(resume_arg).expanduser().resolve()
        if not path.exists():
            print(f"Warning: Provided resume path '{path}' does not exist. It will be created.")
        return path
    
    # Default behavior: create a hidden `.agent/transcripts/` folder in the current directory
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    default_dir = app_config.project_transcripts_dir(cwd)
    default_dir.mkdir(parents=True, exist_ok=True)
    return default_dir / f"{timestamp}.jsonl"


async def main():
    # Get app config
    app_config = load_app_config()
    cwd = Path.cwd().resolve()
    
    # Parse Command Line Arguments
    parser = argparse.ArgumentParser(description=f"{app_config.app_name.capitalize()} Code Agent")
    parser.add_argument("--resume", type=str, help="Path to an existing .jsonl transcript to resume")
    parser.add_argument(
        "--model", 
        type=str, 
        default="ollama/gemma3:12b",
        help="LLM model (e.g. anthropic/claude-3-5-sonnet-20241022, ollama/qwen2.5-coder:14b, gpt-4o)"
    )

    # project workspace
    parser.add_argument(
        "--workspace-root",
        type=str,
        default=None,
        help="The workspace root directory (defaults to current working directory)"
    )

    # System prompt customization flags
    parser.add_argument(
        "--system-prompt-file",
        type=str,
        default=None,
        help="Path to a file whose contents replace the default user-customizable system instructions"
    )
    parser.add_argument(
        "--no-global-system-prompt-file",
        action="store_true",
        help=f"Skip loading {app_config.global_system_prompt_file()}"
    )
    parser.add_argument(
        "--no-proj-system-prompt-file",
        action="store_true",
        help=f"Skip loading {app_config.project_system_prompt_file(cwd)}"
    )

    args = parser.parse_args()

    # Workspace root directory resolution and validation
    root_dir = Path(args.workspace_root).expanduser().resolve() if args.workspace_root else cwd

    # Exit with error if cwd it no within workspace dir
    if not cwd.is_relative_to(root_dir):
        print(f"Error: Current directory ({cwd}) is not within the specified --workspace-root ({root_dir}).")
        sys.exit(1)

    # Creates transcripts folder if it does not exists
    transcript_file = get_transcript_path(app_config, cwd, args.resume)
    print(f"[TRANSCRIPT: {transcript_file}]")
    print(f"[MODEL: {args.model}]")

    if root_dir != cwd:
        print(f"[ROOT: {root_dir}]")

    # 1. Create the context
    ctx = InvocationContext(
        workspace=root_dir,
        cwd=cwd,
        workspace_is_git_repo = (root_dir / ".git").exists(),
        resume_file=Path(args.resume) if args.resume else None
    )
    
    # Initialize State
    registry = create_core_registry(ctx)
    hooks = HookManager()

    # Agent policy
    policy = AgentPolicy()
    policy.mode = AgentMode.BUILD
    
    # Bind and register the built-in hooks
    bound_setup_hook = partial(initial_setup_hook, app_config=app_config, root=root_dir, cwd=cwd)
    bound_mode_hook = partial(agent_mode_hook, policy=policy)
    bound_file_changes_hook = partial(file_changes_hook, ctx=ctx)

    hooks.register_user_prompt(bound_setup_hook)
    hooks.register_user_prompt(bound_mode_hook)
    hooks.register_user_prompt(bound_file_changes_hook)
    
    # Load (or create) the main transcript
    transcript = Transcript(transcript_file)

    # System Prompt injection (only if transcript is brand new)
    if len(transcript.messages) == 0:
        transcript.append(build_system_prompt(app_config, cwd, ctx, args))
    
    print(f"Welcome to {app_config.app_name.capitalize()} Code Agent (Type '/quit' to exit, '/plan' or '/build' to switch modes)")
    
    while True:
        try:
            user_input = input("\n> ")
            if user_input.strip().lower() in ["/quit", "/exit"]:
                break

            # Intercept Mode Commands
            user_input_lower = user_input.strip().lower()

            if user_input_lower.startswith("/plan"):
                policy.mode = AgentMode.PLAN
                print("[Switched to PLAN Mode]")                
                user_input = user_input[len("/plan"):].strip()

                if not user_input:
                    continue
            elif user_input_lower.startswith("/build"):
                policy.mode = AgentMode.BUILD
                print("[Switched to BUILD Mode]")
                user_input = user_input[len("/build"):].strip()

                if not user_input:
                    continue

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
            await run_agentic_loop(transcript, registry, hooks, model=args.model, policy=policy, ctx=ctx)
            
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

if __name__ == "__main__":
    asyncio.run(main())