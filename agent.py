import asyncio
import os
import sys
import time
import argparse
import uuid
from functools import partial
from pathlib import Path
from datetime import datetime
import logging

from capabilities import probe_capabilities, user_warnings
from config import AppConfig, load_app_config
from prompts import build_system_prompt
from sessioncontext import InvocationContext, AgentPolicy, AgentMode

from typing import Literal
from typedefs import (
    TextMessageContent, ToolResultMessageContent, ToolUseMessageContent,
    ToolFailure, ToolResult, UserMessage, SystemMessage, ShellCallback,
    AgentCallback, PlanApprovalCallback
)
from adapter import acompletion
from dotenv import load_dotenv
from transcript import Transcript
from hooks import (
    HookManager,
    agent_mode_hook,
    capabilities_hook,
    initial_setup_hook,
    shell_confirmation_hook,
)
from filestate import file_changes_hook
from processes import kill_quietly, terminate_quietly
from tools.registry import ToolRegistry
from tools.core import create_core_registry
from ui.base import UI, SessionInfo
from ui.null_ui import NullUI
from ui.summaries import summarize_call


load_dotenv(".env.development")
logging.basicConfig(level=logging.WARNING)

# Grace period for draining stdout/stderr after the process exits. A shell
# that backgrounds a child hands it the pipe write-ends, so EOF may never
# arrive; we take whatever was captured and move on.
SHELL_DRAIN_GRACE = 5.0

# How much of each stream reaches the model. Applied per stream, not to the
# combined output.
MAX_SHELL_OUTPUT = 30000


class _StreamCapture:
    """Accumulates one stream up to a cap, remembering what it had to drop.

    The dropped-output flag has to live on an object rather than be returned by
    the reader: on the timeout path the reader tasks are cancelled, so the
    caller reads the buffers directly and never sees a return value.
    """

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.truncated = False
        self._parts: list[bytes] = []
        self._size = 0

    def add(self, chunk: bytes) -> None:
        if self._size >= self.limit:
            self.truncated = True
            return
        self._parts.append(chunk)
        self._size += len(chunk)

    def text(self) -> str:
        decoded = b''.join(self._parts).decode('utf-8', errors='replace')
        if len(decoded) > self.limit:
            self.truncated = True
        return decoded[:self.limit]


async def handle_shell(callback: ShellCallback, ctx: InvocationContext) -> tuple[str, bool, str]:
    """
    Executes a shell command natively with timeouts and streaming partial output.
    Returns (output_text, is_error, ui_summary).
    """
    started_at = time.monotonic()
    
    # Pin the working directory to the agent's cwd (always inside the workspace).
    process = await asyncio.create_subprocess_shell(
        callback.command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(ctx.cwd)
    )

    # Concurrently read stdout and stderr, accumulating up to MAX_SHELL_OUTPUT bytes
    assert process.stdout is not None and process.stderr is not None
    stdout_capture = _StreamCapture(MAX_SHELL_OUTPUT)
    stderr_capture = _StreamCapture(MAX_SHELL_OUTPUT)
    
    async def read_stream(stream: asyncio.StreamReader, capture: _StreamCapture) -> str:
        # Always drain to EOF: stopping reads early would fill the OS pipe
        # buffer and block the child process until the timeout kills it.
        # We just stop *accumulating* once the output cap is reached.
        while chunk := await stream.read(8192):
            capture.add(chunk)
        return capture.text()

    stdout_task = asyncio.create_task(read_stream(process.stdout, stdout_capture))
    stderr_task = asyncio.create_task(read_stream(process.stderr, stderr_capture))

    # Wait for completion or timeout
    exit_code: int | Literal["timeout"]
    try:
        exit_code = await asyncio.wait_for(process.wait(), callback.timeout)
    except asyncio.TimeoutError:
        terminate_quietly(process)
        try:
            await asyncio.wait_for(process.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            kill_quietly(process)
            await process.wait()
        exit_code = "timeout"
    
    # The process being gone doesn't guarantee EOF: a backgrounded grandchild
    # inherits the pipe write-ends and can hold them open forever. Bound the
    # drain and keep whatever was captured so far.
    try:
        stdout, stderr = await asyncio.wait_for(
            asyncio.gather(stdout_task, stderr_task), SHELL_DRAIN_GRACE
        )
        drained = True
    except asyncio.TimeoutError:
        stdout_task.cancel()
        stderr_task.cancel()
        stdout = stdout_capture.text()
        stderr = stderr_capture.text()
        drained = False

    elapsed = time.monotonic() - started_at
    
    # Format the result just like mini_agent
    if exit_code == "timeout":
        is_error = True
        text = f"Command timed out after {callback.timeout:0.1f}s\n{stderr}\n{stdout}"
        ui_summary = f"Shell command timed out after {callback.timeout:0.1f}s"
    elif exit_code == 0:
        is_error = False
        text = f"{stdout}\n{stderr}"
        ui_summary = f"Shell exited with code 0 ({elapsed:.1f}s)"
    else:
        is_error = True
        text = f"{stderr}\n{stdout}"
        ui_summary = f"Shell exited with code {exit_code} ({elapsed:.1f}s)"

    text = text.strip() or "Command completed with no output."
    if not drained:
        text += "\n(Output streams stayed open, likely a backgrounded process; showing partial output.)"

    # Silently dropping the tail of a build log would let the model conclude
    # things about output it never saw.
    if stdout_capture.truncated or stderr_capture.truncated:
        text += (
            f"\n(Output truncated: only the first {MAX_SHELL_OUTPUT} characters of each "
            "stream are shown. Narrow or filter the command to see the rest.)"
        )
        ui_summary += " - output truncated"

    return text, is_error, ui_summary


async def handle_subagent(
    callback: AgentCallback,
    ctx: InvocationContext,
    parent_transcript_path: Path,
    model: str,
    ui: UI = NullUI()
) -> tuple[list[TextMessageContent], bool]:
    """Spins up a recursive sub-agent loop with isolated file-state, registry, and hooks."""
    parent_dir = parent_transcript_path.parent
    sub_id = uuid.uuid4().hex[:6]
    sub_transcript_path = parent_dir / f"{parent_transcript_path.stem}_{callback.subagent_type}_{sub_id}.jsonl"
    
    sub_transcript = Transcript(sub_transcript_path)

    # Sub-agent internals are hidden behind a single spinner: the quiet UI
    # suppresses nested rendering but still routes safety prompts (shell
    # confirmation) back to the real UI.
    sub_ui = ui.for_subagent()

    # 1. Isolated context: a fresh, empty file-state tracker. The sub-agent
    # must Read files itself before writing them; it never inherits the
    # parent's read history (and vice versa).
    sub_ctx = ctx.clone_for_subagent()

    # 2. Isolated registry whose tool closures are bound to the sub-agent's context.
    # Sub-agents never get Task (no unbounded recursion) or SubmitPlan (no
    # interactive plan-approval prompts fired from inside a sub-agent).
    sub_registry = create_core_registry(sub_ctx).clone_excluding(["Task", "SubmitPlan"])
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

    # Sub-agents must not bypass the Shell confirmation gate
    sub_hooks.register_pre_tool(partial(shell_confirmation_hook, ui=sub_ui))

    # 4. Inject the Sub-Agent's specific System Prompt
    sub_transcript.append(SystemMessage(content=callback.system_content))
    
    # 5. Fire the user hooks (This automatically injects AGENTS.md via initial_setup_hook!)
    event = await sub_hooks.trigger_user_prompt(callback.user_content, is_first_prompt=True)
    
    # Handle hook blocks (e.g. if a future hook denies the sub-agent prompt)
    if event.block:
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
    with ui.tool_status(f"Running sub-agent '{callback.subagent_type}' ({callback.callback_description})"):
        final_blocks = await run_agentic_loop(
            sub_transcript, sub_registry, sub_hooks, model=model, policy=sub_policy, ctx=sub_ctx, ui=sub_ui
        )

    # A sub-agent that stops without producing any final text gives the parent
    # nothing to act on; report that as an error rather than a silent success.
    if not any(block.text.strip() for block in final_blocks):
        return [TextMessageContent(
            text="Sub-agent finished without producing a final report. Treat the task as not completed."
        )], True

    return final_blocks, False


def _error_headline(output: str | list[TextMessageContent], max_len: int = 100) -> str:
    """Extracts a compact first line from an error payload for UI summaries."""
    text = output if isinstance(output, str) else " ".join(b.text for b in output)
    first_line = text.strip().splitlines()[0] if text.strip() else "unknown error"
    return first_line if len(first_line) <= max_len else first_line[:max_len - 3] + "..."


async def execute_tool(
    tu: ToolUseMessageContent, 
    registry: ToolRegistry, 
    hooks: HookManager, 
    transcript_path: Path,
    model: str,
    policy: AgentPolicy,
    ctx: InvocationContext,
    ui: UI = NullUI()
) -> list[TextMessageContent | ToolResultMessageContent]:
    """
    Invokes a tool, handles pre/post hooks, and catches execution exceptions.
    Modeled after mini_agent's invoke_tool.
    """
    call_summary = summarize_call(tu.name, tu.input)
    
    # 1. Pre-Hook
    pre_event = await hooks.trigger_pre_tool(tu.name, tu.input)
    if pre_event.decision == "deny":
        ui.tool_result(f"{call_summary} - blocked: {pre_event.deny_reason}", is_error=True)
        return [ToolResultMessageContent(
            tool_use_id=tu.id,
            content=f"Tool blocked: {pre_event.deny_reason}",
            is_error=True
        )]

    # 2. Invoke Tool with Error Boundaries
    ui_summary: str | None = None
    try:
        with ui.tool_status(call_summary):
            raw_result = await registry.invoke(tu.name, tu.input)
        
        # Route Native Callbacks
        if isinstance(raw_result, ShellCallback):
            with ui.tool_status(call_summary):
                result_output, is_error, ui_summary = await handle_shell(raw_result, ctx)
        elif isinstance(raw_result, AgentCallback):
            result_output, is_error = await handle_subagent(raw_result, ctx, transcript_path, model=model, ui=ui)
            ui_summary = (
                f"Sub-agent '{raw_result.subagent_type}' failed" if is_error
                else f"Sub-agent '{raw_result.subagent_type}' completed"
            )
        elif isinstance(raw_result, PlanApprovalCallback):
            decision = ui.approve_plan(raw_result.plan_summary)
            
            if decision.choice == "build":
                policy.mode = AgentMode.BUILD
                policy.notified_mode = AgentMode.BUILD # Prevent the hook from double-firing
                result_output = "SUCCESS: User accepted the plan and switched to BUILD mode. You now have access to write tools. Proceed with execution."
                is_error = False
                ui_summary = "Plan accepted - switched to BUILD mode"

            elif decision.choice == "plan":
                result_output = "User accepted the plan, but chose to remain in PLAN mode. Await further user instructions."
                is_error = False
                ui_summary = "Plan accepted - staying in PLAN mode"

            else:
                result_output = f"REJECTED: User rejected the plan. Reason: {decision.reject_reason}"
                is_error = True
                ui_summary = "Plan rejected"
        elif isinstance(raw_result, ToolFailure):
            # EXPLICIT FAILURE
            result_output = raw_result.error_message
            is_error = True
            ui_summary = raw_result.ui_summary
        elif isinstance(raw_result, ToolResult):
            # Standard tool output with a human-facing summary attached
            result_output = raw_result.content
            is_error = False
            ui_summary = raw_result.ui_summary
        else:
            # Standard tool output
            result_output = raw_result
            is_error = False                
    except Exception as e:
        # Catch Python exceptions (FileNotFound, JSON decoding, missing keys, etc.)
        result_output = f"Error during tool execution: {str(e)}"
        is_error = True

    if is_error:
        ui.tool_result(ui_summary or f"{call_summary} - {_error_headline(result_output)}", is_error=True)
    else:
        ui.tool_result(ui_summary or call_summary, is_error=False)

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

# Hard ceiling on tool-calling turns per user prompt. Prevents a confused
# model from looping forever; the user can always prompt again to continue.
MAX_AGENT_TURNS = 50

async def run_agentic_loop(
    transcript: Transcript,
    base_registry: ToolRegistry,
    hooks: HookManager,
    model: str,
    policy: AgentPolicy,
    ctx: InvocationContext,
    ui: UI = NullUI()
) -> list[TextMessageContent]:
    """
    Returns the pristine list of text blocks from the LLM when no more tools are requested.
    """
    for _ in range(MAX_AGENT_TURNS):
        # Dynamically evaluate tools on every loop iteration
        current_registry = base_registry.clone_readonly() if policy.mode == AgentMode.PLAN else base_registry
        schemas = current_registry.get_all_schemas()

        llm_started_at = time.monotonic()
        with ui.tool_status(f"Waiting for {model}"):
            response = await acompletion(model, schemas, transcript.messages)
        llm_duration = time.monotonic() - llm_started_at
        transcript.append(response)

        texts = [c for c in response.content if getattr(c, "type", None) == "text"]
        tool_uses = [c for c in response.content if getattr(c, "type", None) == "tool_use"]
        thinkings = [c for c in response.content if getattr(c, "type", None) == "thinking"]

        # Reasoning blocks are collapsed into a single subtle line
        if thinkings:
            ui.thinking("\n\n".join(t.thinking for t in thinkings), duration_s=llm_duration)

        for text_block in texts:
            ui.assistant_text(text_block.text)

        # If LLM doesn't want to use any more tools, break the loop and return texts
        if not tool_uses:
            return texts

        # Execute all tools requested by the LLM
        tool_results_content = []
        for tu in tool_uses:
            # Pass current_registry, policy and ctx down
            result_blocks = await execute_tool(tu, current_registry, hooks, transcript.file_path, model=model, policy=policy, ctx=ctx, ui=ui)
            tool_results_content.extend(result_blocks)

        transcript.append(UserMessage(content=tool_results_content))

    # Turn ceiling reached: stop the loop instead of running away.
    warning = f"Stopped after reaching the maximum of {MAX_AGENT_TURNS} tool-calling turns for a single prompt."
    ui.notice(warning)
    return [TextMessageContent(text=warning)]

def get_transcript_path(app_config: AppConfig, cwd: Path, resume_arg: str | None, ui: UI = NullUI()) -> Path:
    """Determines where to load/save the transcript file."""
    if resume_arg:
        path = Path(resume_arg).expanduser().resolve()
        if not path.exists():
            ui.notice(f"Warning: Provided resume path '{path}' does not exist. It will be created.")
        return path
    
    # Default behavior: create a hidden `.agent/transcripts/` folder in the current directory
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    default_dir = app_config.project_transcripts_dir(cwd)
    default_dir.mkdir(parents=True, exist_ok=True)
    return default_dir / f"{timestamp}.jsonl"


def read_git_branch(root: Path) -> str | None:
    """Reads the current branch name (or short detached hash) from .git/HEAD."""
    head_file = root / ".git" / "HEAD"
    try:
        content = head_file.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if content.startswith("ref: "):
        return content.removeprefix("ref: ").rsplit("/", 1)[-1]
    return content[:8] or None


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

    # The single concrete UI for this session. Imported lazily so that every
    # other module (and the test suite) never pulls in `rich` transitively.
    from ui.rich_ui import RichUI
    ui: UI = RichUI()

    # Workspace root directory resolution and validation
    root_dir = Path(args.workspace_root).expanduser().resolve() if args.workspace_root else cwd

    # Exit with error if cwd it no within workspace dir
    if not cwd.is_relative_to(root_dir):
        ui.error(f"Current directory ({cwd}) is not within the specified --workspace-root ({root_dir}).")
        sys.exit(1)

    # Creates transcripts folder if it does not exists
    transcript_file = get_transcript_path(app_config, cwd, args.resume, ui=ui)

    # Probe the environment once. This decides which Grep engine gets
    # registered below, and what the startup banner and system prompt warn about.
    capabilities = probe_capabilities(root_dir)

    # 1. Create the context
    ctx = InvocationContext(
        workspace=root_dir,
        cwd=cwd,
        workspace_is_git_repo = (root_dir / ".git").exists(),
        resume_file=Path(args.resume) if args.resume else None,
        capabilities=capabilities,
    )
    
    # Initialize State
    registry = create_core_registry(ctx)
    hooks = HookManager()

    # Agent policy
    policy = AgentPolicy()
    policy.mode = AgentMode.BUILD
    
    # Bind and register the built-in hooks
    bound_setup_hook = partial(initial_setup_hook, app_config=app_config, root=root_dir, cwd=cwd)
    bound_capabilities_hook = partial(capabilities_hook, capabilities=capabilities)
    bound_mode_hook = partial(agent_mode_hook, policy=policy)
    bound_file_changes_hook = partial(file_changes_hook, ctx=ctx)

    hooks.register_user_prompt(bound_setup_hook)
    hooks.register_user_prompt(bound_capabilities_hook)
    hooks.register_user_prompt(bound_mode_hook)
    hooks.register_user_prompt(bound_file_changes_hook)

    # Safety gate: every Shell command requires explicit user confirmation
    hooks.register_pre_tool(partial(shell_confirmation_hook, ui=ui))
    
    # Load (or create) the main transcript
    transcript = Transcript(transcript_file)

    # System Prompt injection (only if transcript is brand new)
    if len(transcript.messages) == 0:
        transcript.append(build_system_prompt(app_config, cwd, ctx, args))
    
    ui.session_start(SessionInfo(
        app_name=app_config.app_name.capitalize(),
        model=args.model,
        mode=policy.mode.name,
        workspace=root_dir,
        cwd=cwd,
        transcript_path=transcript_file,
        git_branch=read_git_branch(root_dir) if ctx.workspace_is_git_repo else None,
        warnings=tuple(user_warnings(capabilities)),
    ))
    
    while True:
        try:
            user_input = ui.read_user_input()
            if user_input.strip().lower() in ["/quit", "/exit"]:
                break

            # Intercept Mode Commands
            user_input_lower = user_input.strip().lower()

            if user_input_lower.startswith("/plan"):
                policy.mode = AgentMode.PLAN
                ui.mode_changed("PLAN")
                user_input = user_input[len("/plan"):].strip()

                if not user_input:
                    continue
            elif user_input_lower.startswith("/build"):
                policy.mode = AgentMode.BUILD
                ui.mode_changed("BUILD")
                user_input = user_input[len("/build"):].strip()

                if not user_input:
                    continue

            if not user_input.strip():
                continue

            # Fire User Hooks
            is_first_prompt = not any(isinstance(m, UserMessage) for m in transcript.messages)
            event = await hooks.trigger_user_prompt(user_input, is_first_prompt)
            
            if event.block:
                ui.error(f"Blocked: {event.block_reason}")
                continue

            # 3. Assemble the payload: [ PRE, PROMPT, POST ]
            message_content = [
                *event.context_pre,
                TextMessageContent(text=event.prompt),
                *event.context_post
            ]
            
            transcript.append(UserMessage(content=message_content))
            await run_agentic_loop(transcript, registry, hooks, model=args.model, policy=policy, ctx=ctx, ui=ui)
            
        except (KeyboardInterrupt, EOFError):
            ui.notice("Exiting...")
            break
        except Exception as e:
            # An API hiccup (rate limit, network blip) or a bug in a hook must
            # not kill the session: the transcript persists incrementally, so
            # the conversation can simply continue on the next prompt.
            logging.exception("Error during agent turn")
            ui.error(f"{type(e).__name__}: {e}")
            ui.notice("The session is still alive. You can try again or type '/quit' to exit.")

if __name__ == "__main__":
    asyncio.run(main())