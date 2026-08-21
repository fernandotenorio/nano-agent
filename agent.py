import asyncio
import os
import sys
import time
import argparse
from functools import partial
from pathlib import Path
import logging

import sessions
from capabilities import probe_capabilities, user_warnings
from config import AppConfig, load_app_config
from prompts import build_system_prompt
from sessioncontext import InvocationContext, AgentPolicy, AgentMode

from typing import Literal, NoReturn
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
    PLAN_ACCEPTED_TO_BUILD,
    agent_mode_hook,
    capabilities_hook,
    initial_setup_hook,
    restore_policy,
    shell_confirmation_hook,
)
from filestate import file_changes_hook
from processes import kill_quietly, terminate_quietly
from tools.registry import ToolRegistry
from tools.core import create_core_registry
from ui.base import UI, SessionInfo, ToolCallView, UsageInfo, split_model
from ui.null_ui import NullUI
from ui.summaries import summarize_call
from usagetracker import (
    SessionUsageTracker,
    build_report,
    rehydrate_session_usage,
    subagent_name,
)


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
    sub_transcript_path = sessions.subagent_transcript_path(
        parent_transcript_path, callback.subagent_type
    )

    sub_transcript = Transcript(sub_transcript_path)

    # Sub-agent internals are hidden behind a single spinner: the quiet UI
    # suppresses nested rendering but still routes safety prompts (shell
    # confirmation) back to the real UI.
    sub_ui = ui.for_subagent()

    # 1. Isolated context: a fresh, empty file-state tracker. The sub-agent
    # must Read files itself before writing them; it never inherits the
    # parent's read history (and vice versa).
    sub_ctx = ctx.clone_for_subagent(subagent_name(callback.subagent_type))

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
    async with ui.tool_status(f"Running sub-agent '{callback.subagent_type}' ({callback.callback_description})"):
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
        await ui.tool_result(ToolCallView(
            name=tu.name,
            args=tu.input,
            summary=f"{call_summary} - blocked: {pre_event.deny_reason}",
            is_error=True,
        ))
        return [ToolResultMessageContent(
            tool_use_id=tu.id,
            content=f"Tool blocked: {pre_event.deny_reason}",
            is_error=True,
            tool_name=tu.name,
        )]

    # 2. Invoke Tool with Error Boundaries
    ui_summary: str | None = None
    try:
        async with ui.tool_status(call_summary):
            raw_result = await registry.invoke(tu.name, tu.input)
        
        # Route Native Callbacks
        if isinstance(raw_result, ShellCallback):
            async with ui.tool_status(call_summary):
                result_output, is_error, ui_summary = await handle_shell(raw_result, ctx)
        elif isinstance(raw_result, AgentCallback):
            result_output, is_error = await handle_subagent(raw_result, ctx, transcript_path, model=model, ui=ui)
            ui_summary = (
                f"Sub-agent '{raw_result.subagent_type}' failed" if is_error
                else f"Sub-agent '{raw_result.subagent_type}' completed"
            )
        elif isinstance(raw_result, PlanApprovalCallback):
            decision = await ui.approve_plan(raw_result.plan_summary)
            
            if decision.choice == "build":
                policy.mode = AgentMode.BUILD
                policy.notified_mode = AgentMode.BUILD # Prevent the hook from double-firing
                result_output = PLAN_ACCEPTED_TO_BUILD
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
        summary = ui_summary or f"{call_summary} - {_error_headline(result_output)}"
    else:
        summary = ui_summary or call_summary

    await ui.tool_result(ToolCallView(
        name=tu.name,
        args=tu.input,
        summary=summary,
        output=result_output,
        is_error=is_error,
    ))

    # 3. Format Base Result
    #
    # A tool that ran an LLM of its own reports what that cost. It is recorded
    # against the tool here and also written onto the transcript block, so a
    # resumed session can rebuild the same ledger.
    reports_usage = not is_error and isinstance(raw_result, ToolResult)
    internal_usage = raw_result.usage if reports_usage else None
    internal_model = raw_result.internal_model if reports_usage else None

    if internal_usage:
        ctx.usage_tracker.record_tool_internal(
            agent=ctx.agent_name,
            model=internal_model or model,
            tool=tu.name,
            usage=internal_usage,
        )
        # Anything the ledger counts, the running total has to count too, or
        # the two only agree again after a --resume.
        await ui.usage(UsageInfo.from_dict(internal_usage))

    content: list[TextMessageContent | ToolResultMessageContent] = [
        ToolResultMessageContent(
            tool_use_id=tu.id,                
            content=result_output,
            is_error=is_error,
            tool_name=tu.name,
            usage=internal_usage,
            internal_model=internal_model or (model if internal_usage else None),
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
        async with ui.tool_status(f"Waiting for {model}"):
            response = await acompletion(model, schemas, transcript.messages)
        llm_duration = time.monotonic() - llm_started_at
        transcript.append(response)
        await ui.usage(UsageInfo.from_dict(response.usage))

        texts = [c for c in response.content if getattr(c, "type", None) == "text"]
        tool_uses = [c for c in response.content if getattr(c, "type", None) == "tool_use"]
        thinkings = [c for c in response.content if getattr(c, "type", None) == "thinking"]

        # One response, one ledger entry, however many tools it asked for: the
        # tokens bought the whole turn. The tool names travel with the entry so
        # the per-tool view can apportion them later.
        ctx.usage_tracker.record_turn(
            agent=ctx.agent_name,
            model=model,
            tools=[tu.name for tu in tool_uses],
            usage=response.usage,
        )

        # Reasoning blocks are collapsed into a single subtle line
        if thinkings:
            await ui.thinking("\n\n".join(t.thinking for t in thinkings), duration_s=llm_duration)

        for text_block in texts:
            await ui.assistant_text(text_block.text)

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
    await ui.notice(warning)
    return [TextMessageContent(text=warning)]

RECENT_SESSIONS_SHOWN = 5
"""How many ids to offer when a requested session cannot be found."""


def _exit_with_session_help(message: str, known_ids: list[str]) -> NoReturn:
    """Reports a session that cannot be resumed, and stops.

    Resolution happens before the UI exists, so this reports itself the only
    way it can, like the workspace check does. The recent ids are listed
    because the alternative is asking the user to go read a directory name
    built from a hash.
    """
    print(message, file=sys.stderr)

    if known_ids:
        print("\nMost recent sessions for this workspace:", file=sys.stderr)
        for session_id in reversed(known_ids[-RECENT_SESSIONS_SHOWN:]):
            print(f"  {session_id}", file=sys.stderr)

    sys.exit(1)


def resolve_session(
    app_config: AppConfig,
    workspace: Path,
    resume_id: str | None,
    continue_latest: bool,
) -> tuple[sessions.SessionPaths, bool]:
    """Decides which session this run belongs to.

    Returns the session's paths and whether it already has a conversation to
    pick up, which is what tells the caller to rebuild the token ledger instead
    of starting a fresh one.

    Nothing is created here; asking for a session that does not exist is an
    error rather than an empty new one, because a mistyped id silently starting
    over is how a conversation gets lost.
    """
    if resume_id:
        paths = sessions.session_for(app_config, workspace, resume_id)
        if not paths.exists:
            _exit_with_session_help(
                f"No session '{resume_id}' found for workspace {workspace}.",
                sessions.list_session_ids(app_config, workspace),
            )
        return paths, True

    if continue_latest:
        latest = sessions.latest_session(app_config, workspace)
        if latest is None:
            _exit_with_session_help(
                f"No previous session to continue for workspace {workspace}.", []
            )
        return latest, True

    return sessions.new_session(app_config, workspace), False


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


def create_ui(kind: str, app_config: AppConfig) -> UI:
    """Builds the single concrete UI for this session.

    Rendering libraries are imported here and nowhere else, so that every
    other module (and the test suite) never pulls in `rich` or `textual`
    transitively.
    """
    if kind == "rich":
        from ui.rich_ui import RichUI
        return RichUI()

    from ui.theme import load_ui_theme
    from ui.tui import TextualUI
    return TextualUI(load_ui_theme(app_config))


async def run_repl(
    ui: UI,
    info: SessionInfo,
    transcript: Transcript,
    registry: ToolRegistry,
    hooks: HookManager,
    policy: AgentPolicy,
    ctx: InvocationContext,
    model: str,
) -> None:
    """The interactive session: banner, then one agentic loop per prompt.

    Runs *inside* whatever lifecycle the UI provides (see `UI.run`), which is
    why the banner is rendered here rather than during setup: a full-screen
    front-end has nothing to draw on until its application is up.
    """
    await ui.session_start(info)

    # A resumed session starts with tokens already spent. Handing them to the
    # UI once keeps the running total in the status bar agreeing with the
    # usage view, which counts the whole conversation.
    resumed = ctx.usage_tracker.total()
    if resumed.calls:
        await ui.usage(UsageInfo(
            input_tokens=resumed.input_tokens,
            output_tokens=resumed.output_tokens,
        ))

    while True:
        try:
            user_input = await ui.read_user_input()
            if user_input.strip().lower() in ["/quit", "/exit"]:
                break

            # Intercept Mode Commands
            user_input_lower = user_input.strip().lower()

            if user_input_lower.startswith("/usage"):
                await ui.show_usage(build_report(ctx.usage_tracker))
                user_input = user_input[len("/usage"):].strip()

                if not user_input:
                    continue
            elif user_input_lower.startswith("/plan"):
                policy.mode = AgentMode.PLAN
                await ui.mode_changed("PLAN")
                user_input = user_input[len("/plan"):].strip()

                if not user_input:
                    continue
            elif user_input_lower.startswith("/build"):
                policy.mode = AgentMode.BUILD
                await ui.mode_changed("BUILD")
                user_input = user_input[len("/build"):].strip()

                if not user_input:
                    continue

            if not user_input.strip():
                continue

            # Fire User Hooks
            is_first_prompt = not any(isinstance(m, UserMessage) for m in transcript.messages)
            event = await hooks.trigger_user_prompt(user_input, is_first_prompt)

            if event.block:
                await ui.error(f"Blocked: {event.block_reason}")
                continue

            # 3. Assemble the payload: [ PRE, PROMPT, POST ]
            message_content = [
                *event.context_pre,
                TextMessageContent(text=event.prompt),
                *event.context_post
            ]

            transcript.append(UserMessage(content=message_content))
            await run_agentic_loop(transcript, registry, hooks, model=model, policy=policy, ctx=ctx, ui=ui)

        except (KeyboardInterrupt, EOFError):
            await ui.notice("Exiting...")
            break
        except Exception as e:
            # An API hiccup (rate limit, network blip) or a bug in a hook must
            # not kill the session: the transcript persists incrementally, so
            # the conversation can simply continue on the next prompt.
            logging.exception("Error during agent turn")
            await ui.error(f"{type(e).__name__}: {e}")
            await ui.notice("The session is still alive. You can try again or type '/quit' to exit.")


async def main():
    # Get app config
    app_config = load_app_config()
    cwd = Path.cwd().resolve()
    
    # Parse Command Line Arguments
    parser = argparse.ArgumentParser(description=f"{app_config.app_name.capitalize()} Code Agent")
    # A session is named by its id, not by a path: the transcript now lives in
    # a home directory nobody types by hand.
    session_group = parser.add_mutually_exclusive_group()
    session_group.add_argument(
        "--resume",
        type=str,
        metavar="SESSION_ID",
        help="Id of a session to resume for this workspace",
    )
    session_group.add_argument(
        "--continue",
        dest="continue_latest",
        action="store_true",
        help="Resume the most recently used session for this workspace",
    )

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
        "--ui",
        choices=["textual", "rich"],
        default="textual",
        help="Front-end to use: the full-screen Textual interface, or plain scrolling output"
    )

    args = parser.parse_args()

    # Workspace root directory resolution and validation
    root_dir = Path(args.workspace_root).expanduser().resolve() if args.workspace_root else cwd

    # Exit with error if cwd it no within workspace dir. This happens before
    # any UI exists, so it reports itself the only way it can.
    if not cwd.is_relative_to(root_dir):
        print(
            f"Current directory ({cwd}) is not within the specified --workspace-root ({root_dir}).",
            file=sys.stderr,
        )
        sys.exit(1)

    # Anything discovered during setup that the user should see, held until
    # the UI is alive and can render the session banner.
    startup_warnings: list[str] = []

    # Which conversation this run belongs to. Keyed by the workspace root, so
    # the same project answers the same way from any subdirectory.
    session, resuming = resolve_session(
        app_config, root_dir, args.resume, args.continue_latest
    )
    transcript_file = session.transcript

    # Record what this session is while it is being created. A failure here
    # costs a future rename its title, not the conversation.
    meta_warning = sessions.ensure_meta(session, root_dir)
    if meta_warning:
        startup_warnings.append(meta_warning)

    # Probe the environment once. This decides which Grep engine gets
    # registered below, and what the startup banner and system prompt warn about.
    capabilities = probe_capabilities(root_dir)

    # Resuming continues the previous session's accounting rather than
    # restarting it, so the totals on screen cover the whole conversation and
    # not just the part of it that happened since the last launch.
    usage_tracker = (
        rehydrate_session_usage(transcript_file) if resuming else SessionUsageTracker()
    )

    # 1. Create the context
    ctx = InvocationContext(
        workspace=root_dir,
        cwd=cwd,
        workspace_is_git_repo = (root_dir / ".git").exists(),
        resume_file=transcript_file if resuming else None,
        capabilities=capabilities,
        usage_tracker=usage_tracker,
    )
    
    # Initialize State
    registry = create_core_registry(ctx)
    hooks = HookManager()

    # Load (or create) the main transcript
    transcript = Transcript(transcript_file)

    # Agent policy. A resumed conversation has already been told which mode it
    # is in, so the policy is read back out of the transcript rather than
    # started over: a blank one repeats the mode reminder on the first prompt,
    # and drops a session the user left in PLAN mode back into BUILD.
    policy = (
        restore_policy(transcript.messages) if resuming
        else AgentPolicy(mode=AgentMode.BUILD)
    )
    
    # Bind and register the built-in hooks
    bound_setup_hook = partial(initial_setup_hook, app_config=app_config, root=root_dir, cwd=cwd)
    bound_capabilities_hook = partial(capabilities_hook, capabilities=capabilities)
    bound_mode_hook = partial(agent_mode_hook, policy=policy)
    bound_file_changes_hook = partial(file_changes_hook, ctx=ctx)

    hooks.register_user_prompt(bound_setup_hook)
    hooks.register_user_prompt(bound_capabilities_hook)
    hooks.register_user_prompt(bound_mode_hook)
    hooks.register_user_prompt(bound_file_changes_hook)

    ui = create_ui(args.ui, app_config)

    # A front-end with its own way in to the usage view (a key binding, a
    # button) needs to build the report at the moment it is asked for.
    ui.set_usage_provider(lambda: build_report(ctx.usage_tracker))

    # Safety gate: every Shell command requires explicit user confirmation
    hooks.register_pre_tool(partial(shell_confirmation_hook, ui=ui))

    # System Prompt injection (only if transcript is brand new)
    if len(transcript.messages) == 0:
        transcript.append(build_system_prompt(app_config, ctx, args))

    provider, _ = split_model(args.model)

    info = SessionInfo(
        app_name=app_config.app_name.capitalize(),
        model=args.model,
        mode=policy.mode.name,
        session_id=session.session_id,
        workspace=root_dir,
        cwd=cwd,
        transcript_path=transcript_file,
        git_branch=read_git_branch(root_dir) if ctx.workspace_is_git_repo else None,
        provider=provider,
        warnings=tuple(startup_warnings + user_warnings(capabilities)),
    )

    await ui.run(partial(
        run_repl,
        ui=ui,
        info=info,
        transcript=transcript,
        registry=registry,
        hooks=hooks,
        policy=policy,
        ctx=ctx,
        model=args.model,
    ))

if __name__ == "__main__":
    asyncio.run(main())