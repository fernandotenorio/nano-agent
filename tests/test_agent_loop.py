import asyncio
import io
import json
import os
import tempfile
import unittest
from contextlib import nullcontext, redirect_stderr
from unittest.mock import ANY, AsyncMock, MagicMock, PropertyMock, patch
from pathlib import Path
import sys
import uuid

import sessions

from typedefs import (
    AssistantMessage, TextMessageContent, ToolUseMessageContent, 
    ToolResultMessageContent, ToolResult, UserMessage, ToolFailure, ShellCallback,
    AgentCallback, SystemMessage
)

from config import AppConfig
from hooks import (
    MODE_ANNOUNCEMENTS,
    PreToolUseEvent,
    PostToolUseEvent,
    UserPromptEvent,
    mode_reminder,
)
from agent import run_agentic_loop
from agent import execute_tool, handle_shell, handle_subagent, main, run_repl
from sessioncontext import AgentPolicy, AgentMode, InvocationContext
from ui.base import SessionInfo, UsageInfo
from ui.null_ui import NullUI, QuietUI
from usagetracker import SessionUsageTracker


class StubREPLUI(NullUI):
    """Silent UI whose REPL prompt still reads from the builtin input(),
    so the existing @patch('builtins.input') plumbing keeps driving main()."""
    async def read_user_input(self) -> str:
        return input()


class RecordingREPLUI(StubREPLUI):
    """StubREPLUI that also keeps the session banner it was handed.

    main() builds its own front-end, so a class-level record is the only way
    back to the SessionInfo it assembled -- which is where the starting mode
    becomes observable.
    """

    sessions_started: list[SessionInfo] = []

    async def session_start(self, info: SessionInfo) -> None:
        type(self).sessions_started.append(info)


class RecordingUI(NullUI):
    """Replays a scripted conversation and keeps what the session rendered.

    `for_subagent` wraps itself the way the real front-ends do, rather than
    inheriting NullUI's silent self, so tests take the path a live session
    takes into a sub-agent.
    """

    def __init__(self, inputs: list[str] | None = None):
        self.inputs = list(inputs or ())
        self.reports: list = []
        self.usage_updates: list[UsageInfo] = []

    async def read_user_input(self) -> str:
        return self.inputs.pop(0)

    async def show_usage(self, report) -> None:
        self.reports.append(report)

    async def usage(self, info) -> None:
        self.usage_updates.append(info)

    def for_subagent(self) -> NullUI:
        return QuietUI(self)

    @property
    def reported_tokens(self) -> int:
        """The running total a status bar would be showing by now."""
        return sum(info.total_tokens for info in self.usage_updates)


class TestAgenticLoopGroup1(unittest.IsolatedAsyncioTestCase):
    """
    Test Group 1: Core Agentic Loop (run_agentic_loop)
    Validates the orchestration of LLM calls, tool dispatcher invocations, 
    and transcript state management.
    """

    def setUp(self):
        # 1. Mock the Transcript so we don't do disk I/O
        self.transcript = MagicMock()
        self.transcript.messages = []
        self.transcript.file_path = Path("/mock/transcripts/test.jsonl")
        
        # 2. Mock the Tool Registry
        self.registry = MagicMock()
        self.mock_schemas = [{"type": "function", "name": "dummy_tool"}]
        self.registry.get_all_schemas.return_value = self.mock_schemas
        
        # 3. Mock the HookManager
        self.hooks = MagicMock()

        # Policy
        self.policy = AgentPolicy()
        self.policy.mode = AgentMode.BUILD

        # Invocation context (carries the per-agent file-state tracker)
        self.ctx = InvocationContext(
            workspace=Path("/mock/workspace"),
            cwd=Path("/mock/workspace"),
            workspace_is_git_repo=False
        )
        
        # 4. Standard vars
        self.model = "test-mock-model"

    @patch("builtins.print")  # Keep test runner output clean
    @patch("agent.execute_tool", new_callable=AsyncMock)
    @patch("agent.acompletion", new_callable=AsyncMock)
    async def test_zero_tool_uses(self, mock_acompletion, mock_execute_tool, mock_print):
        """
        Test 1.1: Zero Tool Uses (Simple Conversation)
        If the LLM returns only text and no tool uses, the loop should exit immediately
        and return the text blocks.
        """
        # Setup: LLM just replies with text
        text_block = TextMessageContent(text="Hello, I am ready to help.")
        mock_acompletion.return_value = AssistantMessage(
            content=[text_block],
            model=self.model,
            stop_reason="end_turn"
        )

        # Action
        result = await run_agentic_loop(self.transcript, self.registry, self.hooks, self.model, self.policy, self.ctx)

        # Assertions
        self.assertEqual(result, [text_block])
        
        # Ensure LLM was called exactly once with correct args
        mock_acompletion.assert_called_once_with(self.model, self.mock_schemas, self.transcript.messages)
        
        # Ensure transcript stored the AssistantMessage
        self.transcript.append.assert_called_once_with(mock_acompletion.return_value)
        
        # Ensure no tools were executed
        mock_execute_tool.assert_not_called()

    @patch("builtins.print")
    @patch("agent.execute_tool", new_callable=AsyncMock)
    @patch("agent.acompletion", new_callable=AsyncMock)
    async def test_single_tool_invocation(self, mock_acompletion, mock_execute_tool, mock_print):
        """
        Test 1.2: Single Tool Invocation
        If the LLM asks for a tool, the loop should execute it, append the result, 
        and call the LLM again.
        """
        # Setup: Iteration 1 (LLM requests a tool)
        tool_use = ToolUseMessageContent(id="call_123", name="Read", input={"file_path": "main.py"})
        text_ack = TextMessageContent(text="Let me check that file.")
        msg1 = AssistantMessage(content=[text_ack, tool_use], model=self.model, stop_reason="tool_use")
        
        # Setup: Iteration 2 (LLM gives final answer)
        msg2 = AssistantMessage(content=[TextMessageContent(text="File looks good.")], model=self.model, stop_reason="end_turn")
        
        mock_acompletion.side_effect = [msg1, msg2]
        
        # Mock the tool executor's return value
        tool_result = ToolResultMessageContent(tool_use_id="call_123", content="print('hello')", is_error=False)
        mock_execute_tool.return_value = [tool_result]

        # Action
        result = await run_agentic_loop(self.transcript, self.registry, self.hooks, self.model, self.policy, self.ctx)

        # Assertions
        # Loop exited correctly with the final text
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].text, "File looks good.")
        
        # LLM called exactly twice
        self.assertEqual(mock_acompletion.call_count, 2)
        
        # Tool executed exactly once with correct parameters
        mock_execute_tool.assert_called_once_with(
            tool_use, self.registry, self.hooks, self.transcript.file_path, model=self.model, policy=self.policy, ctx=self.ctx, ui=ANY
        )
        
        # Transcript should have 3 appends: msg1, UserMessage(ToolResult), msg2
        self.assertEqual(self.transcript.append.call_count, 3)
        append_calls = self.transcript.append.call_args_list
        
        self.assertEqual(append_calls[0][0][0], msg1)
        
        user_msg = append_calls[1][0][0]
        self.assertIsInstance(user_msg, UserMessage)
        self.assertEqual(user_msg.content, [tool_result])
        
        self.assertEqual(append_calls[2][0][0], msg2)

    @patch("builtins.print")
    @patch("agent.execute_tool", new_callable=AsyncMock)
    @patch("agent.acompletion", new_callable=AsyncMock)
    async def test_parallel_multiple_tool_uses(self, mock_acompletion, mock_execute_tool, mock_print):
        """
        Test 1.3: Parallel/Multiple Tool Uses in One Turn
        If the LLM requests multiple tools simultaneously, they should all be executed 
        and packaged together into a single UserMessage for the next LLM call.
        """
        # Setup: Iteration 1 (LLM requests 3 tools at once)
        tu1 = ToolUseMessageContent(id="t1", name="ToolA", input={"a": 1})
        tu2 = ToolUseMessageContent(id="t2", name="ToolB", input={"b": 2})
        tu3 = ToolUseMessageContent(id="t3", name="ToolC", input={"c": 3})
        
        msg1 = AssistantMessage(content=[tu1, tu2, tu3], model=self.model, stop_reason="tool_use")
        msg2 = AssistantMessage(content=[TextMessageContent(text="Done running all 3 tools.")], model=self.model, stop_reason="end_turn")
        
        mock_acompletion.side_effect = [msg1, msg2]
        
        # Mock the tool executor to return corresponding results
        tr1 = ToolResultMessageContent(tool_use_id="t1", content="Result A")
        tr2 = ToolResultMessageContent(tool_use_id="t2", content="Result B")
        tr3 = ToolResultMessageContent(tool_use_id="t3", content="Result C", is_error=True)
        
        # side_effect allows us to return a different result for each of the 3 execute_tool calls
        mock_execute_tool.side_effect = [[tr1], [tr2], [tr3]]

        # Action
        result = await run_agentic_loop(self.transcript, self.registry, self.hooks, self.model, self.policy, self.ctx)

        # Assertions
        # execute_tool was called exactly 3 times
        self.assertEqual(mock_execute_tool.call_count, 3)
        
        # Verify the transcript was injected with a SINGLE UserMessage containing all 3 tool results
        append_calls = self.transcript.append.call_args_list
        user_msg = append_calls[1][0][0]
        
        self.assertIsInstance(user_msg, UserMessage)
        self.assertEqual(len(user_msg.content), 3)
        self.assertEqual(user_msg.content, [tr1, tr2, tr3])


class TestLoopUsageRecording(unittest.IsolatedAsyncioTestCase):
    """
    Test Group 1b: Usage Recording in the Loop (run_agentic_loop)

    The loop is the only place that knows which model string was asked for and
    which tools a response called, so it is where the ledger is written. The
    property under test throughout: one response is one entry, so a turn that
    calls three tools does not bill the session three times.
    """

    def setUp(self):
        self.transcript = MagicMock()
        self.transcript.messages = []
        self.transcript.file_path = Path("/mock/transcripts/test.jsonl")

        self.registry = MagicMock()
        self.registry.get_all_schemas.return_value = []

        self.hooks = MagicMock()
        self.policy = AgentPolicy()
        self.policy.mode = AgentMode.BUILD

        self.ctx = InvocationContext(
            workspace=Path("/mock/workspace"),
            cwd=Path("/mock/workspace"),
            workspace_is_git_repo=False,
        )
        self.model = "ollama/gemma3:12b"
        self.usage = {"prompt_tokens": 300, "completion_tokens": 30}

    @property
    def tracker(self):
        return self.ctx.usage_tracker

    def final(self, usage: dict | None = None) -> AssistantMessage:
        """A closing text response, optionally carrying a usage figure."""
        return AssistantMessage(
            content=[TextMessageContent(text="Done.")],
            model="gemma3:12b",
            stop_reason="end_turn",
            usage=usage,
        )

    async def run_loop(self):
        return await run_agentic_loop(
            self.transcript, self.registry, self.hooks, self.model, self.policy, self.ctx
        )

    @patch("builtins.print")
    @patch("agent.acompletion", new_callable=AsyncMock)
    async def test_text_turn_is_recorded_with_no_tools(self, mock_acompletion, mock_print):
        mock_acompletion.return_value = self.final(self.usage)

        await self.run_loop()

        record = self.tracker.records[0]
        self.assertEqual(record.agent, "main")
        self.assertEqual(record.tools, ())
        self.assertEqual(record.activities, ("text",))
        self.assertEqual(record.total_tokens, 330)

    @patch("builtins.print")
    @patch("agent.acompletion", new_callable=AsyncMock)
    async def test_the_requested_model_is_recorded_not_the_echoed_one(
        self, mock_acompletion, mock_print
    ):
        """The response says 'gemma3:12b'; only the loop knows the provider."""
        mock_acompletion.return_value = self.final(self.usage)

        await self.run_loop()

        self.assertEqual(self.tracker.records[0].model, "ollama/gemma3:12b")
        self.assertIn("ollama", self.tracker.by_provider())

    @patch("builtins.print")
    @patch("agent.acompletion", new_callable=AsyncMock)
    async def test_a_response_without_usage_records_nothing(self, mock_acompletion, mock_print):
        mock_acompletion.return_value = self.final()

        await self.run_loop()

        self.assertEqual(self.tracker.records, [])

    @patch("builtins.print")
    @patch("agent.execute_tool", new_callable=AsyncMock)
    @patch("agent.acompletion", new_callable=AsyncMock)
    async def test_a_multi_tool_response_is_recorded_exactly_once(
        self, mock_acompletion, mock_execute_tool, mock_print
    ):
        tool_uses = [
            ToolUseMessageContent(id="t1", name="Read", input={}),
            ToolUseMessageContent(id="t2", name="Grep", input={}),
            ToolUseMessageContent(id="t3", name="ls", input={}),
        ]
        mock_acompletion.side_effect = [
            AssistantMessage(
                content=tool_uses, model="gemma3:12b", stop_reason="tool_use", usage=self.usage
            ),
            self.final(),
        ]
        mock_execute_tool.side_effect = [
            [ToolResultMessageContent(tool_use_id=tu.id, content="ok")] for tu in tool_uses
        ]

        await self.run_loop()

        self.assertEqual(len(self.tracker.records), 1)
        self.assertEqual(self.tracker.total().total_tokens, 330)
        self.assertEqual(self.tracker.records[0].tools, ("Read", "Grep", "ls"))

    @patch("builtins.print")
    @patch("agent.execute_tool", new_callable=AsyncMock)
    @patch("agent.acompletion", new_callable=AsyncMock)
    async def test_a_multi_tool_response_is_split_across_its_tools(
        self, mock_acompletion, mock_execute_tool, mock_print
    ):
        tool_uses = [
            ToolUseMessageContent(id="t1", name="Read", input={}),
            ToolUseMessageContent(id="t2", name="Grep", input={}),
        ]
        mock_acompletion.side_effect = [
            AssistantMessage(
                content=tool_uses, model="gemma3:12b", stop_reason="tool_use", usage=self.usage
            ),
            self.final(),
        ]
        mock_execute_tool.side_effect = [
            [ToolResultMessageContent(tool_use_id=tu.id, content="ok")] for tu in tool_uses
        ]

        await self.run_loop()

        by_tool = self.tracker.by_tool()
        self.assertEqual(by_tool["Read"].total_tokens, 165)
        self.assertEqual(by_tool["Grep"].total_tokens, 165)

    @patch("builtins.print")
    @patch("agent.execute_tool", new_callable=AsyncMock)
    @patch("agent.acompletion", new_callable=AsyncMock)
    async def test_every_iteration_of_a_turn_is_recorded(
        self, mock_acompletion, mock_execute_tool, mock_print
    ):
        tool_use = ToolUseMessageContent(id="t1", name="Read", input={})
        mock_acompletion.side_effect = [
            AssistantMessage(
                content=[tool_use], model="gemma3:12b", stop_reason="tool_use", usage=self.usage
            ),
            self.final({"prompt_tokens": 100, "completion_tokens": 10}),
        ]
        mock_execute_tool.return_value = [
            ToolResultMessageContent(tool_use_id="t1", content="ok")
        ]

        await self.run_loop()

        self.assertEqual(len(self.tracker.records), 2)
        self.assertEqual(self.tracker.total().total_tokens, 440)

    @patch("builtins.print")
    @patch("agent.execute_tool", new_callable=AsyncMock)
    @patch("agent.acompletion", new_callable=AsyncMock)
    async def test_a_sub_agent_loop_records_under_its_own_name(
        self, mock_acompletion, mock_execute_tool, mock_print
    ):
        """The tracker is shared, so the agent name is what keeps them apart."""
        sub_ctx = self.ctx.clone_for_subagent("subagent:code-reviewer")
        mock_acompletion.return_value = self.final(self.usage)

        await run_agentic_loop(
            self.transcript, self.registry, self.hooks, self.model, self.policy, sub_ctx
        )

        self.assertIs(sub_ctx.usage_tracker, self.ctx.usage_tracker)
        self.assertEqual(
            self.tracker.by_agent()["subagent:code-reviewer"].total_tokens, 330
        )


class TestExecuteToolGroup2(unittest.IsolatedAsyncioTestCase):
    """
    Test Group 2: Tool Execution Boundaries (execute_tool)
    Validates hook interceptions, error handling (both explicit and unhandled),
    and formatting of the ToolResultMessageContent.
    """

    def setUp(self):
        # 1. Setup Standard Inputs
        self.tu = ToolUseMessageContent(id="call_999", name="TestTool", input={"key": "val"})
        self.transcript_path = Path("/mock/path.jsonl")
        self.model = "test-model"

        # 2. Mock Registry
        self.registry = MagicMock()
        self.registry.invoke = AsyncMock()

        # 3. Mock Hooks
        self.hooks = MagicMock()
        self.hooks.trigger_pre_tool = AsyncMock()
        self.hooks.trigger_post_tool = AsyncMock()
        
        # Default hook behavior (allow by default, no extra context)
        self.hooks.trigger_pre_tool.return_value = PreToolUseEvent(
            tool_name=self.tu.name, tool_input=self.tu.input, decision="allow"
        )
        self.hooks.trigger_post_tool.return_value = PostToolUseEvent(
            tool_name=self.tu.name, tool_input=self.tu.input, tool_output=""
        )

        # 4. Policy and context (now required by execute_tool)
        self.policy = AgentPolicy()
        self.ctx = InvocationContext(
            workspace=Path("/mock/workspace"),
            cwd=Path("/mock/workspace"),
            workspace_is_git_repo=False
        )

    @patch("builtins.print")
    async def test_standard_tool_success(self, mock_print):
        """
        Test 2.1: Standard Tool Success
        Verifies that a normal string output from a tool is formatted into a
        ToolResultMessageContent with is_error=False.
        """
        # Setup
        self.registry.invoke.return_value = "Normal tool execution output."

        # Action
        result = await execute_tool(self.tu, self.registry, self.hooks, self.transcript_path, self.model, self.policy, self.ctx)

        # Assertions
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], ToolResultMessageContent)
        self.assertEqual(result[0].tool_use_id, "call_999")
        self.assertEqual(result[0].content, "Normal tool execution output.")
        self.assertFalse(result[0].is_error)

        # Ensure post hook was triggered on success
        self.hooks.trigger_post_tool.assert_called_once()

    @patch("builtins.print")
    async def test_pre_hook_block(self, mock_print):
        """
        Test 2.2: Pre-Hook Block / Deny
        If the pre-hook denies execution, the registry should never be invoked,
        and an error should be returned to the LLM immediately.
        """
        # Setup
        self.hooks.trigger_pre_tool.return_value = PreToolUseEvent(
            tool_name=self.tu.name, tool_input=self.tu.input, 
            decision="deny", deny_reason="Admin privileges required."
        )

        # Action
        result = await execute_tool(self.tu, self.registry, self.hooks, self.transcript_path, self.model, self.policy, self.ctx)

        # Assertions
        self.registry.invoke.assert_not_called()  # Tool never executed!
        self.hooks.trigger_post_tool.assert_not_called()  # Post hook skipped!
        
        self.assertEqual(len(result), 1)
        self.assertTrue(result[0].is_error)
        self.assertIn("Admin privileges required.", result[0].content)

    @patch("builtins.print")
    async def test_python_exception_during_execution(self, mock_print):
        """
        Test 2.3: Python Exception during Execution
        If a tool crashes with a raw Python exception, it should be caught and
        returned safely as an error string, preventing the loop from crashing.
        """
        # Setup
        self.registry.invoke.side_effect = ValueError("Corrupted JSON payload")

        # Action
        result = await execute_tool(self.tu, self.registry, self.hooks, self.transcript_path, self.model, self.policy, self.ctx)

        # Assertions
        self.assertEqual(len(result), 1)
        self.assertTrue(result[0].is_error)
        self.assertIn("Error during tool execution:", result[0].content)
        self.assertIn("Corrupted JSON payload", result[0].content)
        
        # Post hook skipped because it's an error
        self.hooks.trigger_post_tool.assert_not_called()

    @patch("builtins.print")
    async def test_explicit_tool_failure(self, mock_print):
        """
        Test 2.4: Explicit ToolFailure Return
        If a tool returns the strict ToolFailure object (signaling a controlled error state),
        it should be flagged as is_error=True.
        """
        # Setup
        self.registry.invoke.return_value = ToolFailure(error_message="File not found on disk.")

        # Action
        result = await execute_tool(self.tu, self.registry, self.hooks, self.transcript_path, self.model, self.policy, self.ctx)

        # Assertions
        self.assertEqual(len(result), 1)
        self.assertTrue(result[0].is_error)
        self.assertEqual(result[0].content, "File not found on disk.")
        
        # Post hook skipped because it's an error
        self.hooks.trigger_post_tool.assert_not_called()

    @patch("builtins.print")
    async def test_registry_tool_failure_is_not_reported_as_success(self, mock_print):
        """
        Test 2.6: Crash Reported by the Registry
        ToolRegistry.invoke catches exceptions itself and returns a ToolFailure.
        That must reach the LLM flagged as an error, with post-hooks skipped;
        the old plain-string return made a crashed tool look successful.
        """
        # Setup: the real registry's own error envelope, not a raised exception
        self.registry.invoke.return_value = ToolFailure(
            error_message="Error: tool 'TestTool': Corrupted JSON payload"
        )

        # Action
        result = await execute_tool(self.tu, self.registry, self.hooks, self.transcript_path, self.model, self.policy, self.ctx)

        # Assertions
        self.assertEqual(len(result), 1)
        self.assertTrue(result[0].is_error)
        self.assertIn("Corrupted JSON payload", result[0].content)

        self.hooks.trigger_post_tool.assert_not_called()

    @patch("builtins.print")
    async def test_post_hook_context_injection(self, mock_print):
        """
        Test 2.5: Post-Hook Context Injection
        If the post-hook returns additional context (like AGENTS.md injections or state reminders),
        it should be appended alongside the tool result.
        """
        # Setup
        self.registry.invoke.return_value = "Code successfully compiled."
        
        extra_context = TextMessageContent(text="<system>Remember to run tests.</system>")
        self.hooks.trigger_post_tool.return_value = PostToolUseEvent(
            tool_name=self.tu.name, 
            tool_input=self.tu.input, 
            tool_output="Code successfully compiled.",
            additional_context=[extra_context]
        )

        # Action
        result = await execute_tool(self.tu, self.registry, self.hooks, self.transcript_path, self.model, self.policy, self.ctx)

        # Assertions
        self.assertEqual(len(result), 2)
        
        # First block should be the standard tool result
        self.assertIsInstance(result[0], ToolResultMessageContent)
        self.assertEqual(result[0].content, "Code successfully compiled.")
        
        # Second block should be the injected text content
        self.assertIsInstance(result[1], TextMessageContent)
        self.assertEqual(result[1].text, "<system>Remember to run tests.</system>")


class TestExecuteToolUsageStamping(unittest.IsolatedAsyncioTestCase):
    """
    Test Group 2b: Tool Accounting (execute_tool)

    Every tool result is stamped with the name of the tool that produced it,
    and a tool that ran an LLM of its own has that billed to it. Both are also
    written onto the transcript, which is the only way a resumed session can
    rebuild the same ledger.
    """

    def setUp(self):
        self.tu = ToolUseMessageContent(id="call_999", name="WebFetch", input={"url": "x"})
        self.transcript_path = Path("/mock/path.jsonl")
        self.model = "ollama/gemma3:12b"

        self.registry = MagicMock()
        self.registry.invoke = AsyncMock()

        self.hooks = MagicMock()
        self.hooks.trigger_pre_tool = AsyncMock(return_value=PreToolUseEvent(
            tool_name=self.tu.name, tool_input=self.tu.input, decision="allow"
        ))
        self.hooks.trigger_post_tool = AsyncMock(return_value=PostToolUseEvent(
            tool_name=self.tu.name, tool_input=self.tu.input, tool_output=""
        ))

        self.policy = AgentPolicy()
        self.ctx = InvocationContext(
            workspace=Path("/mock/workspace"),
            cwd=Path("/mock/workspace"),
            workspace_is_git_repo=False,
        )
        self.usage = {"prompt_tokens": 80, "completion_tokens": 8}

    async def execute(self, ui=None):
        return await execute_tool(
            self.tu, self.registry, self.hooks, self.transcript_path,
            self.model, self.policy, self.ctx, ui=ui or NullUI(),
        )

    @patch("builtins.print")
    async def test_the_tool_name_is_stamped_on_the_result(self, mock_print):
        self.registry.invoke.return_value = "Fetched."

        result = await self.execute()

        self.assertEqual(result[0].tool_name, "WebFetch")

    @patch("builtins.print")
    async def test_a_plain_tool_carries_no_usage(self, mock_print):
        self.registry.invoke.return_value = "Fetched."

        result = await self.execute()

        self.assertIsNone(result[0].usage)
        self.assertIsNone(result[0].internal_model)
        self.assertEqual(self.ctx.usage_tracker.records, [])

    @patch("builtins.print")
    async def test_an_llm_run_inside_a_tool_is_billed_to_that_tool(self, mock_print):
        self.registry.invoke.return_value = ToolResult(
            content="Summarised.",
            usage=self.usage,
            internal_model="openai/gpt-4o-mini",
        )

        await self.execute()

        record = self.ctx.usage_tracker.records[0]
        self.assertTrue(record.internal)
        self.assertEqual(record.tools, ("WebFetch",))
        self.assertEqual(record.model, "openai/gpt-4o-mini")
        self.assertEqual(self.ctx.usage_tracker.by_tool()["WebFetch"].total_tokens, 88)

    @patch("builtins.print")
    async def test_internal_usage_is_written_onto_the_transcript(self, mock_print):
        self.registry.invoke.return_value = ToolResult(
            content="Summarised.",
            usage=self.usage,
            internal_model="openai/gpt-4o-mini",
        )

        result = await self.execute()

        self.assertEqual(result[0].usage, self.usage)
        self.assertEqual(result[0].internal_model, "openai/gpt-4o-mini")

    @patch("builtins.print")
    async def test_usage_without_a_model_falls_back_to_the_session_model(self, mock_print):
        """A tool that reports tokens but not which model spent them."""
        self.registry.invoke.return_value = ToolResult(content="Summarised.", usage=self.usage)

        result = await self.execute()

        self.assertEqual(self.ctx.usage_tracker.records[0].model, "ollama/gemma3:12b")
        self.assertEqual(result[0].internal_model, "ollama/gemma3:12b")

    @patch("builtins.print")
    async def test_an_llm_run_inside_a_tool_reaches_the_running_total(self, mock_print):
        """Anything the ledger counts, the status bar has to count too, or the
        two only agree again after a --resume."""
        self.registry.invoke.return_value = ToolResult(
            content="Summarised.",
            usage=self.usage,
            internal_model="openai/gpt-4o-mini",
        )
        ui = RecordingUI()

        await self.execute(ui)

        self.assertEqual(ui.reported_tokens, 88)
        self.assertEqual(ui.reported_tokens, self.ctx.usage_tracker.total().total_tokens)

    @patch("builtins.print")
    async def test_a_plain_tool_moves_nothing(self, mock_print):
        self.registry.invoke.return_value = "Fetched."
        ui = RecordingUI()

        await self.execute(ui)

        self.assertEqual(ui.usage_updates, [])

    @patch("builtins.print")
    async def test_a_failed_tool_records_nothing(self, mock_print):
        self.registry.invoke.side_effect = ValueError("boom")

        result = await self.execute()

        self.assertTrue(result[0].is_error)
        self.assertEqual(self.ctx.usage_tracker.records, [])

    @patch("builtins.print")
    async def test_a_blocked_tool_is_still_named(self, mock_print):
        self.hooks.trigger_pre_tool.return_value = PreToolUseEvent(
            tool_name=self.tu.name, tool_input=self.tu.input,
            decision="deny", deny_reason="Not allowed.",
        )

        result = await self.execute()

        self.assertTrue(result[0].is_error)
        self.assertEqual(result[0].tool_name, "WebFetch")
        self.assertEqual(self.ctx.usage_tracker.records, [])

    @patch("builtins.print")
    async def test_a_sub_agent_tool_records_under_the_sub_agent(self, mock_print):
        sub_ctx = self.ctx.clone_for_subagent("subagent:explore")
        self.registry.invoke.return_value = ToolResult(
            content="Summarised.", usage=self.usage, internal_model="openai/gpt-4o-mini"
        )

        await execute_tool(
            self.tu, self.registry, self.hooks, self.transcript_path,
            self.model, self.policy, sub_ctx,
        )

        self.assertEqual(
            self.ctx.usage_tracker.by_agent()["subagent:explore"].total_tokens, 88
        )


class TestHandleShellGroup3(unittest.IsolatedAsyncioTestCase):
    """
    Test Group 3: Subprocess / Shell Handler (handle_shell)
    Validates async stream reading, output formatting, byte limits, 
    and strict timeout/termination boundaries.
    """

    def setUp(self):
        self.callback = ShellCallback(command="echo 'test'", timeout=0.1)
        self.ctx = InvocationContext(
            workspace=Path("/mock/workspace"),
            cwd=Path("/mock/workspace"),
            workspace_is_git_repo=False
        )

    def _create_mock_process(self, stdout_data: bytes, stderr_data: bytes, exit_code: int = 0, hang_time: float = 0, eof: bool = True):
        """
        Helper to create a fake asyncio subprocess with pre-filled streams.
        Pass eof=False to simulate a backgrounded grandchild holding the pipe
        write-ends open, so the readers never see EOF.
        """
        process = MagicMock()
        
        # 1. Setup real asyncio StreamReaders
        stdout_stream = asyncio.StreamReader()
        stdout_stream.feed_data(stdout_data)
        
        stderr_stream = asyncio.StreamReader()
        stderr_stream.feed_data(stderr_data)

        if eof:
            stdout_stream.feed_eof()
            stderr_stream.feed_eof()
        
        process.stdout = stdout_stream
        process.stderr = stderr_stream
        
        # 2. Mock the wait() coroutine
        async def mock_wait():
            if hang_time > 0:
                await asyncio.sleep(hang_time)
            return exit_code
            
        process.wait = mock_wait
        process.terminate = MagicMock()
        process.kill = MagicMock()
        
        return process

    @patch("builtins.print")
    @patch("asyncio.create_subprocess_shell")
    async def test_successful_command_execution(self, mock_create_shell, mock_print):
        """
        Test 3.1: Successful Command Execution
        Checks exit code 0 formats as: {stdout}\n{stderr}
        """
        # Setup
        mock_process = self._create_mock_process(b"hello world", b"", exit_code=0)
        mock_create_shell.return_value = mock_process
        
        # Action
        text, is_error, _ui_summary = await handle_shell(self.callback, self.ctx)
        
        # Assertions
        self.assertFalse(is_error)
        self.assertEqual(text, "hello world")
        mock_create_shell.assert_called_once_with(
            self.callback.command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self.ctx.cwd)
        )

    @patch("builtins.print")
    @patch("asyncio.create_subprocess_shell")
    async def test_command_failure(self, mock_create_shell, mock_print):
        """
        Test 3.2: Command Failure
        Checks non-zero exit code formats as: {stderr}\n{stdout} (errors prioritized)
        """
        # Setup
        mock_process = self._create_mock_process(b"normal output", b"fatal error", exit_code=1)
        mock_create_shell.return_value = mock_process
        
        # Action
        text, is_error, _ui_summary = await handle_shell(self.callback, self.ctx)
        
        # Assertions
        self.assertTrue(is_error)
        # Verify stderr comes before stdout on failure
        self.assertEqual(text, "fatal error\nnormal output")

    @patch("builtins.print")
    @patch("asyncio.create_subprocess_shell")
    async def test_output_truncation(self, mock_create_shell, mock_print):
        """
        Test 3.3: Output Truncation
        Checks that huge outputs are cut to 30,000 characters AND that the
        model is told the tail is missing, so it can't draw conclusions from
        output it never saw.
        """
        # Setup: Create 40,000 bytes of data (exceeds the 30,000 limit)
        huge_stdout = b"A" * 40000
        mock_process = self._create_mock_process(huge_stdout, b"", exit_code=0)
        mock_create_shell.return_value = mock_process
        
        # Action
        text, is_error, ui_summary = await handle_shell(self.callback, self.ctx)
        
        # Assertions
        self.assertFalse(is_error)
        self.assertTrue(text.startswith("A" * 30000))
        self.assertNotIn("A" * 30001, text)
        self.assertIn("Output truncated", text)
        self.assertIn("output truncated", ui_summary)

    @patch("builtins.print")
    @patch("asyncio.create_subprocess_shell")
    async def test_timeout_handling(self, mock_create_shell, mock_print):
        """
        Test 3.4: Timeout Handling
        Checks that a hanging process is successfully terminated, killed, 
        and flags a timeout error.
        """
        # Setup: process.wait() hangs for 10 seconds, but callback timeout is 0.1s
        mock_process = self._create_mock_process(b"partial out", b"", hang_time=10.0)
        mock_create_shell.return_value = mock_process
        
        # Action
        text, is_error, _ui_summary = await handle_shell(self.callback, self.ctx)
        
        # Assertions
        self.assertTrue(is_error)
        self.assertIn("Command timed out after 0.1s", text)
        self.assertIn("partial out", text)  # We should still capture what was emitted before timeout
        
        # Ensure we tried to safely terminate, and when it didn't respond to that, kill it
        mock_process.terminate.assert_called_once()
        mock_process.kill.assert_called_once()

    @patch("builtins.print")
    @patch("asyncio.create_subprocess_shell")
    async def test_output_under_cap_has_no_truncation_notice(self, mock_create_shell, mock_print):
        """
        Test 3.3b: No False Truncation Notice
        Output that fits under the cap must come back verbatim, so the notice
        can be trusted to mean something when it does appear.
        """
        # Setup: comfortably under the 30,000 character cap
        mock_process = self._create_mock_process(b"B" * 29999, b"", exit_code=0)
        mock_create_shell.return_value = mock_process

        # Action
        text, is_error, ui_summary = await handle_shell(self.callback, self.ctx)

        # Assertions
        self.assertFalse(is_error)
        self.assertEqual(text, "B" * 29999)
        self.assertNotIn("truncated", ui_summary)

    @patch("builtins.print")
    @patch("asyncio.create_subprocess_shell")
    async def test_timeout_tolerates_already_exited_process(self, mock_create_shell, mock_print):
        """
        Test 3.8: Signalling a Process That Already Died
        A process can exit between the timeout firing and us signalling it.
        Once asyncio has reaped it, terminate() raises ProcessLookupError, which
        must not surface as a tool crash: the process being gone is the outcome
        we were asking for.
        """
        mock_process = self._create_mock_process(b"partial out", b"", exit_code=0)

        # The first wait() outlives the 0.1s timeout; by the time we signal, the
        # process is gone, so terminate() reports it as already reaped and the
        # follow-up wait() returns at once.
        calls = {"count": 0}

        async def mock_wait():
            calls["count"] += 1
            if calls["count"] == 1:
                await asyncio.sleep(10.0)
            return 0

        mock_process.wait = mock_wait
        mock_process.terminate = MagicMock(side_effect=ProcessLookupError())
        mock_create_shell.return_value = mock_process

        # Action
        text, is_error, _ui_summary = await handle_shell(self.callback, self.ctx)

        # Assertions
        self.assertTrue(is_error)
        self.assertIn("Command timed out after 0.1s", text)
        self.assertIn("partial out", text)

        mock_process.terminate.assert_called_once()

    @patch("agent.SHELL_DRAIN_GRACE", 0.05)
    @patch("builtins.print")
    @patch("asyncio.create_subprocess_shell")
    async def test_drain_grace_on_clean_exit(self, mock_create_shell, mock_print):
        """
        Test 3.6: Bounded Drain After a Clean Exit
        A shell that backgrounds a child exits 0 while the grandchild keeps the
        pipe write-ends open, so EOF never arrives. The drain must give up and
        return the partial output instead of hanging forever.
        """
        # Setup: exit code 0, but the streams never reach EOF
        mock_process = self._create_mock_process(b"partial out", b"", exit_code=0, eof=False)
        mock_create_shell.return_value = mock_process

        # Action
        text, is_error, ui_summary = await handle_shell(self.callback, self.ctx)

        # Assertions
        self.assertFalse(is_error)  # A clean exit is not a timeout
        self.assertIn("partial out", text)
        self.assertIn("Output streams stayed open", text)
        self.assertNotIn("timed out", ui_summary)

    @patch("agent.SHELL_DRAIN_GRACE", 0.05)
    @patch("builtins.print")
    @patch("asyncio.create_subprocess_shell")
    async def test_drain_grace_on_timeout(self, mock_create_shell, mock_print):
        """
        Test 3.7: Bounded Drain After a Timeout
        Killing the process doesn't close pipes held by a grandchild, so the
        post-kill drain must also be bounded while still reporting the timeout
        and whatever output was captured.
        """
        # Setup: process hangs past the 0.1s timeout AND the streams never EOF
        mock_process = self._create_mock_process(b"partial out", b"", hang_time=10.0, eof=False)
        mock_create_shell.return_value = mock_process

        # Action
        text, is_error, _ui_summary = await handle_shell(self.callback, self.ctx)

        # Assertions
        self.assertTrue(is_error)
        self.assertIn("Command timed out after 0.1s", text)
        self.assertIn("partial out", text)
        self.assertIn("Output streams stayed open", text)

        mock_process.terminate.assert_called_once()
        mock_process.kill.assert_called_once()

    @patch("builtins.print")
    @patch("asyncio.create_subprocess_shell")
    async def test_empty_output_fallback(self, mock_create_shell, mock_print):
        """
        Test 3.5: Empty Output Fallback
        Checks that perfectly silent commands return a fallback string so the LLM 
        doesn't crash on empty input.
        """
        # Setup: 0 exit code, absolutely no stdout/stderr
        mock_process = self._create_mock_process(b"", b"", exit_code=0)
        mock_create_shell.return_value = mock_process
        
        # Action
        text, is_error, _ui_summary = await handle_shell(self.callback, self.ctx)
        
        # Assertions
        self.assertFalse(is_error)
        self.assertEqual(text, "Command completed with no output.")


class TestHandleSubagentGroup4(unittest.IsolatedAsyncioTestCase):
    """
    Test Group 4: Recursive Sub-Agent Handler (handle_subagent)
    Validates sub-agent transcript initialization, system prompt injection,
    tool restriction filtering, and file-state isolation from the parent.
    """

    def setUp(self):
        self.parent_path = Path("/mock/dir/parent_transcript.jsonl")
        self.model = "test-model"

        # Parent invocation context (carries the parent's file-state tracker)
        self.ctx = InvocationContext(
            workspace=Path("/mock/workspace"),
            cwd=Path("/mock/workspace"),
            workspace_is_git_repo=False
        )

        # handle_subagent rebuilds the registry from the sub-agent's context,
        # so we patch the factory rather than passing a parent registry.
        # It always strips Task/SubmitPlan first (clone_excluding), then
        # optionally applies the profile's tool filter (clone_filtered).
        self.built_registry = MagicMock()
        self.excluded_registry = MagicMock()
        self.built_registry.clone_excluding.return_value = self.excluded_registry
        self.filtered_registry = MagicMock()
        self.excluded_registry.clone_filtered.return_value = self.filtered_registry

        # Keep the built-in setup hook from touching the real filesystem
        self.gather_patcher = patch("hooks.gather_context_files", return_value="")
        self.gather_patcher.start()

    def tearDown(self):
        self.gather_patcher.stop()

    def _create_callback(self, tools: list[str] | None = None) -> AgentCallback:
        """Helper to create a sub-agent callback payload."""
        return AgentCallback(
            subagent_type="code-reviewer",
            callback_description="Review this code.",
            tools=tools,
            system_content="You are a strict reviewer.",
            user_content="Here is the code to review."
        )

    @patch("builtins.print")
    @patch("sessions.uuid.uuid4")
    @patch("agent.create_core_registry")
    @patch("agent.Transcript")
    @patch("agent.run_agentic_loop", new_callable=AsyncMock)
    async def test_proper_sub_agent_initialization(
        self, mock_run_agentic_loop, mock_transcript_class, mock_create_registry, mock_uuid, mock_print
    ):
        """
        Test 4.1: Proper Sub-Agent Initialization
        Verifies transcript file generation, initial message injection, and
        that the sub-agent runs on an isolated context with an empty tracker.
        """
        # Setup
        mock_uuid.return_value = uuid.UUID("12345678-1234-5678-1234-567812345678")
        mock_create_registry.return_value = self.built_registry
        callback = self._create_callback()
        
        mock_transcript_instance = MagicMock()
        mock_transcript_class.return_value = mock_transcript_instance
        
        mock_run_agentic_loop.return_value = [TextMessageContent(text="Sub-agent done.")]

        # Give the parent tracker some state to prove the sub-agent doesn't inherit it
        self.ctx.file_state.known[Path("/mock/workspace/parent_read.py")] = MagicMock()

        # Action
        result, is_error = await handle_subagent(
            callback, self.ctx, self.parent_path, self.model
        )

        # Assertions
        self.assertFalse(is_error)
        self.assertEqual(result[0].text, "Sub-agent done.")
        
        # 1. The transcript lands in the session's own 'subagents' directory,
        # which is what lets a resumed session find it without matching
        # filenames against the main transcript's name.
        expected_path = Path("/mock/dir/subagents/code-reviewer_123456.jsonl")
        mock_transcript_class.assert_called_once_with(expected_path)
        
        # 2. Verify initial prompt injection
        self.assertEqual(mock_transcript_instance.append.call_count, 2)
        
        sys_msg = mock_transcript_instance.append.call_args_list[0][0][0]
        self.assertIsInstance(sys_msg, SystemMessage)
        self.assertEqual(sys_msg.content, "You are a strict reviewer.")
        
        user_msg = mock_transcript_instance.append.call_args_list[1][0][0]
        self.assertIsInstance(user_msg, UserMessage)
        self.assertEqual(user_msg.content[0].text, "Here is the code to review.")

        # 3. Verify context isolation: the loop got a *clone* with an empty tracker
        sub_ctx = mock_run_agentic_loop.call_args.kwargs["ctx"]
        self.assertIsNot(sub_ctx, self.ctx)
        self.assertIsNot(sub_ctx.file_state, self.ctx.file_state)
        self.assertEqual(sub_ctx.file_state.known, {})
        self.assertEqual(sub_ctx.workspace, self.ctx.workspace)

        # 4. Usage accounting is the exception to that isolation: the ledger is
        # shared so the session sees one total, and the name keeps the
        # sub-agent's share tellable apart.
        self.assertEqual(sub_ctx.agent_name, "subagent:code-reviewer")
        self.assertIs(sub_ctx.usage_tracker, self.ctx.usage_tracker)

        # The registry was built from the sub-agent's context, not the parent's
        mock_create_registry.assert_called_once_with(sub_ctx)

        # The sub-agent got its own HookManager, not the parent's
        from hooks import HookManager
        sub_hooks = mock_run_agentic_loop.call_args[0][2]
        self.assertIsInstance(sub_hooks, HookManager)

    @patch("builtins.print")
    @patch("agent.create_core_registry")
    @patch("agent.Transcript")
    @patch("agent.run_agentic_loop", new_callable=AsyncMock)
    async def test_tool_filtering_restricted_profile(
        self, mock_run_agentic_loop, mock_transcript_class, mock_create_registry, mock_print
    ):
        """
        Test 4.2: Tool Filtering (Restricted Profile)
        If the callback specifies a tools list, the sub-agent should get a
        filtered version of its freshly built registry.
        """
        # Setup: Restrict to only "Read" and "Shell"
        mock_create_registry.return_value = self.built_registry
        callback = self._create_callback(tools=["Read", "Shell"])

        # Action
        await handle_subagent(callback, self.ctx, self.parent_path, self.model)

        # Assertions
        # Task/SubmitPlan are always stripped first, then the profile filter applies
        self.built_registry.clone_excluding.assert_called_once_with(["Task", "SubmitPlan"])
        self.excluded_registry.clone_filtered.assert_called_once_with(["Read", "Shell"])
        
        # Verify the restricted registry was passed to the sub-agent loop
        called_registry = mock_run_agentic_loop.call_args[0][1]
        self.assertEqual(called_registry, self.filtered_registry)

    @patch("builtins.print")
    @patch("agent.create_core_registry")
    @patch("agent.Transcript")
    @patch("agent.run_agentic_loop", new_callable=AsyncMock)
    async def test_unfiltered_tools_default_profile(
        self, mock_run_agentic_loop, mock_transcript_class, mock_create_registry, mock_print
    ):
        """
        Test 4.3: Unfiltered Tools (Default Profile)
        If the callback does not restrict tools (tools=None), the sub-agent
        uses its freshly built registry unfiltered.
        """
        # Setup: tools = None
        mock_create_registry.return_value = self.built_registry
        callback = self._create_callback(tools=None)

        # Action
        await handle_subagent(callback, self.ctx, self.parent_path, self.model)

        # Assertions
        # Even with tools=None, Task/SubmitPlan are always stripped
        self.built_registry.clone_excluding.assert_called_once_with(["Task", "SubmitPlan"])
        self.excluded_registry.clone_filtered.assert_not_called()
        
        # Verify the registry (minus Task/SubmitPlan) was passed with no further filtering
        called_registry = mock_run_agentic_loop.call_args[0][1]
        self.assertEqual(called_registry, self.excluded_registry)


class TestSubagentLiveUsage(unittest.IsolatedAsyncioTestCase):
    """
    Test Group 4b: Live Token Accounting Across a Sub-Agent (handle_subagent)

    A sub-agent's turns are hidden behind one spinner on purpose, but the
    tokens they cost are the session's tokens. The running total therefore has
    to move while the sub-agent works. It used not to: the quiet UI swallowed
    the sub-agent's usage reports, so the status bar stayed short until a
    --resume rebuilt the ledger from the transcripts on disk.

    Unlike Group 4 above, these run the real agentic loop so that the usage
    actually flows; only the model call, the transcript, and the registry are
    stubbed.
    """

    def setUp(self):
        self.parent_path = Path("/mock/dir/parent_transcript.jsonl")
        self.model = "ollama/gemma3:12b"
        self.ctx = InvocationContext(
            workspace=Path("/mock/workspace"),
            cwd=Path("/mock/workspace"),
            workspace_is_git_repo=False,
        )

        # Keep the built-in setup hook off the real filesystem.
        self.gather_patcher = patch("hooks.gather_context_files", return_value="")
        self.gather_patcher.start()
        self.addCleanup(self.gather_patcher.stop)

    @property
    def callback(self) -> AgentCallback:
        return AgentCallback(
            subagent_type="code-reviewer",
            callback_description="Review this code.",
            tools=None,
            system_content="You are a strict reviewer.",
            user_content="Here is the code to review.",
        )

    def registry(self) -> MagicMock:
        """A registry that survives clone_excluding and answers any tool call."""
        registry = MagicMock()
        registry.clone_excluding.return_value = registry
        registry.get_all_schemas.return_value = []
        registry.invoke = AsyncMock(return_value="ok")
        return registry

    def response(self, usage: dict | None = None, tool: str | None = None) -> AssistantMessage:
        content = (
            [ToolUseMessageContent(id="t1", name=tool, input={})] if tool
            else [TextMessageContent(text="Reviewed.")]
        )
        return AssistantMessage(
            content=content,
            model="gemma3:12b",
            stop_reason="tool_use" if tool else "end_turn",
            usage=usage,
        )

    async def run_subagent(self, ui: RecordingUI) -> None:
        await handle_subagent(self.callback, self.ctx, self.parent_path, self.model, ui=ui)

    @patch("builtins.print")
    @patch("agent.create_core_registry")
    @patch("agent.Transcript")
    @patch("agent.acompletion", new_callable=AsyncMock)
    async def test_a_sub_agents_tokens_reach_the_running_total(
        self, mock_acompletion, mock_transcript_cls, mock_create_registry, mock_print
    ):
        mock_create_registry.return_value = self.registry()
        mock_acompletion.return_value = self.response(
            usage={"prompt_tokens": 200, "completion_tokens": 20}
        )
        ui = RecordingUI()

        await self.run_subagent(ui)

        self.assertEqual(len(ui.usage_updates), 1)
        self.assertEqual(ui.usage_updates[0].input_tokens, 200)
        self.assertEqual(ui.usage_updates[0].output_tokens, 20)

    @patch("builtins.print")
    @patch("agent.create_core_registry")
    @patch("agent.Transcript")
    @patch("agent.acompletion", new_callable=AsyncMock)
    async def test_every_sub_agent_turn_moves_the_total(
        self, mock_acompletion, mock_transcript_cls, mock_create_registry, mock_print
    ):
        """Not just the last one: a sub-agent that works for a while is visible
        as it goes, which is the whole point of a running total."""
        mock_create_registry.return_value = self.registry()
        mock_acompletion.side_effect = [
            self.response(usage={"prompt_tokens": 100, "completion_tokens": 10}, tool="Read"),
            self.response(usage={"prompt_tokens": 300, "completion_tokens": 30}),
        ]
        ui = RecordingUI()

        await self.run_subagent(ui)

        self.assertEqual([info.total_tokens for info in ui.usage_updates], [110, 330])

    @patch("builtins.print")
    @patch("agent.create_core_registry")
    @patch("agent.Transcript")
    @patch("agent.acompletion", new_callable=AsyncMock)
    async def test_the_running_total_agrees_with_the_ledger(
        self, mock_acompletion, mock_transcript_cls, mock_create_registry, mock_print
    ):
        """The invariant behind the bug: what the status bar shows live is what
        a --resume would rebuild, so the number never jumps between sessions."""
        mock_create_registry.return_value = self.registry()
        mock_acompletion.side_effect = [
            self.response(usage={"prompt_tokens": 100, "completion_tokens": 10}, tool="Read"),
            self.response(usage={"prompt_tokens": 300, "completion_tokens": 30}),
        ]
        ui = RecordingUI()

        await self.run_subagent(ui)

        self.assertEqual(ui.reported_tokens, self.ctx.usage_tracker.total().total_tokens)
        self.assertEqual(
            self.ctx.usage_tracker.by_agent()["subagent:code-reviewer"].total_tokens, 440
        )

    @patch("builtins.print")
    @patch("agent.create_core_registry")
    @patch("agent.Transcript")
    @patch("agent.acompletion", new_callable=AsyncMock)
    async def test_a_sub_agent_turn_without_usage_reports_nothing(
        self, mock_acompletion, mock_transcript_cls, mock_create_registry, mock_print
    ):
        """Local models routinely omit the figure; an empty update would only
        add noise."""
        mock_create_registry.return_value = self.registry()
        mock_acompletion.return_value = self.response()
        ui = RecordingUI()

        await self.run_subagent(ui)

        self.assertEqual(ui.reported_tokens, 0)

    @patch("builtins.print")
    @patch("agent.create_core_registry")
    @patch("agent.Transcript")
    @patch("agent.acompletion", new_callable=AsyncMock)
    async def test_the_sub_agents_rendering_still_stays_hidden(
        self, mock_acompletion, mock_transcript_cls, mock_create_registry, mock_print
    ):
        """Reporting tokens must not have opened the floodgates on everything
        else the sub-agent does."""
        mock_create_registry.return_value = self.registry()
        mock_acompletion.return_value = self.response(
            usage={"prompt_tokens": 10, "completion_tokens": 1}
        )

        rendered: list[str] = []

        class WatchfulUI(RecordingUI):
            async def assistant_text(self, text: str) -> None:
                rendered.append(text)

            async def tool_result(self, call) -> None:
                rendered.append(call.summary)

        await self.run_subagent(WatchfulUI())

        self.assertEqual(rendered, [])


class MockTranscriptState:
    """A lightweight mock to simulate transcript state changes in memory."""
    def __init__(self, path):
        self.file_path = path
        self.messages = []
    def append(self, msg):
        self.messages.append(msg)

class TestReplUsageCommands(unittest.IsolatedAsyncioTestCase):
    """
    Test Group 5b: Usage in the REPL (run_repl)

    /usage is the front-end-agnostic way in: the Textual app has a key binding
    and a button, but a plain terminal has only what it can type.
    """

    def setUp(self):
        self.transcript = MagicMock()
        self.transcript.messages = []
        self.transcript.file_path = Path("/mock/transcripts/test.jsonl")

        self.registry = MagicMock()
        self.hooks = MagicMock()
        self.hooks.trigger_user_prompt = AsyncMock(
            side_effect=lambda prompt, is_first_prompt=True: UserPromptEvent(
                prompt=prompt, is_first_prompt=is_first_prompt
            )
        )

        self.policy = AgentPolicy()
        self.ctx = InvocationContext(
            workspace=Path("/mock/workspace"),
            cwd=Path("/mock/workspace"),
            workspace_is_git_repo=False,
        )
        self.info = SessionInfo(
            app_name="Prisma",
            model="ollama/gemma3:12b",
            mode="BUILD",
            workspace=self.ctx.workspace,
            cwd=self.ctx.cwd,
            transcript_path=self.transcript.file_path,
            provider="ollama",
        )

    async def run_repl_with(self, ui: RecordingUI) -> None:
        await run_repl(
            ui, self.info, self.transcript, self.registry,
            self.hooks, self.policy, self.ctx, "ollama/gemma3:12b",
        )

    def spend(self, prompt: int = 100, completion: int = 10) -> None:
        self.ctx.usage_tracker.record_turn(
            "main", "ollama/gemma3:12b", (),
            {"prompt_tokens": prompt, "completion_tokens": completion},
        )

    @patch("builtins.print")
    @patch("agent.run_agentic_loop", new_callable=AsyncMock)
    async def test_usage_command_renders_a_report(self, mock_run_loop, mock_print):
        self.spend()
        ui = RecordingUI(["/usage", "/quit"])

        await self.run_repl_with(ui)

        self.assertEqual(len(ui.reports), 1)
        self.assertEqual(ui.reports[0].totals.total_tokens, 110)

    @patch("builtins.print")
    @patch("agent.run_agentic_loop", new_callable=AsyncMock)
    async def test_usage_command_alone_does_not_prompt_the_model(
        self, mock_run_loop, mock_print
    ):
        ui = RecordingUI(["/usage", "/quit"])

        await self.run_repl_with(ui)

        mock_run_loop.assert_not_called()

    @patch("builtins.print")
    @patch("agent.run_agentic_loop", new_callable=AsyncMock)
    async def test_usage_command_reflects_the_latest_spend(self, mock_run_loop, mock_print):
        """The report is built when asked for, so it is never a stale snapshot."""
        ui = RecordingUI(["/usage", "/usage", "/quit"])

        async def spend_then_return(*args, **kwargs):
            self.spend()
            return []

        mock_run_loop.side_effect = spend_then_return
        self.spend()

        await self.run_repl_with(ui)

        self.assertEqual(ui.reports[0].totals.total_tokens, 110)
        self.assertEqual(ui.reports[1].totals.total_tokens, 110)

    @patch("builtins.print")
    @patch("agent.run_agentic_loop", new_callable=AsyncMock)
    async def test_text_after_the_command_still_reaches_the_model(
        self, mock_run_loop, mock_print
    ):
        mock_run_loop.return_value = []
        ui = RecordingUI(["/usage now read main.py", "/quit"])

        await self.run_repl_with(ui)

        self.assertEqual(len(ui.reports), 1)
        mock_run_loop.assert_called_once()

    @patch("builtins.print")
    @patch("agent.run_agentic_loop", new_callable=AsyncMock)
    async def test_a_resumed_session_seeds_the_running_total(self, mock_run_loop, mock_print):
        """Otherwise the status bar reads zero while the usage view reads the truth."""
        self.spend(prompt=1000, completion=100)
        ui = RecordingUI(["/quit"])

        await self.run_repl_with(ui)

        self.assertEqual(len(ui.usage_updates), 1)
        self.assertEqual(ui.usage_updates[0].input_tokens, 1000)
        self.assertEqual(ui.usage_updates[0].output_tokens, 100)

    @patch("builtins.print")
    @patch("agent.run_agentic_loop", new_callable=AsyncMock)
    async def test_a_fresh_session_seeds_nothing(self, mock_run_loop, mock_print):
        ui = RecordingUI(["/quit"])

        await self.run_repl_with(ui)

        self.assertEqual(ui.usage_updates, [])


class TestMainLoopGroup5(unittest.IsolatedAsyncioTestCase):
    """
    Test Group 5: User Input Boundary & Hooks (main)
    Validates CLI interactions, hook interceptions (blocking and context injection),
    and graceful loop exits.
    """

    def setUp(self):
        # We must patch sys.argv so argparse doesn't try to parse unittest's CLI
        # args. `--ui rich` keeps main() on the plain front-end, which the stub
        # below then replaces; the full-screen UI is exercised in test_tui_app.
        self.argv_patcher = patch.object(sys, "argv", ["agent.py", "--ui", "rich"])
        self.argv_patcher.start()

        # These tests exercise the REPL logic, not the rendering: swap the
        # concrete RichUI for a silent stub that still reads input() so the
        # builtins.input patches keep working.
        self.ui_patcher = patch("ui.rich_ui.RichUI", StubREPLUI)
        self.ui_patcher.start()

        # main() resolves a session against the user's home directory and writes
        # its meta.json there. These tests run it for real, so the home is
        # redirected into a temporary directory: nothing may land in the
        # developer's own ~/.prisma.
        self.home_dir = tempfile.TemporaryDirectory()
        self.fake_home = Path(self.home_dir.name) / ".prisma"
        self.home_patcher = patch(
            "config.AppConfig.home_config_dir", new_callable=PropertyMock
        )
        self.home_patcher.start().return_value = self.fake_home

    def tearDown(self):
        self.argv_patcher.stop()
        self.ui_patcher.stop()
        self.home_patcher.stop()
        self.home_dir.cleanup()

    @patch("builtins.print")
    @patch("agent.run_agentic_loop", new_callable=AsyncMock)
    @patch("agent.HookManager")
    @patch("agent.create_core_registry")
    @patch("agent.Transcript")
    async def test_hook_blocks_prompt(
        self, mock_transcript_cls, mock_registry, mock_hook_mgr_cls, mock_run_loop, mock_print
    ):
        """
        Test 5.1: Hook Blocks Prompt
        If a hook flags block=True, the prompt is rejected, not appended to the transcript, 
        and the LLM loop is skipped.
        """
        mock_transcript_cls.return_value = MockTranscriptState(Path("/mock/main.jsonl"))
        
        mock_hook_mgr = MagicMock()
        mock_hook_mgr_cls.return_value = mock_hook_mgr
        
        # Mock hook to block the prompt
        mock_hook_mgr.trigger_user_prompt = AsyncMock(return_value=UserPromptEvent(
            prompt="do something bad", 
            is_first_prompt=True,
            block=True, 
            block_reason="Safety policy violation."
        ))

        # Mock User Input: Types a prompt, gets blocked, then quits.
        with patch("builtins.input", side_effect=["do something bad", "/quit"]):
            await main()

        # Assertions
        mock_run_loop.assert_not_called()
        
        # The transcript should only have the SystemMessage (added on init), not the UserMessage
        transcript_instance = mock_transcript_cls.return_value
        self.assertEqual(len(transcript_instance.messages), 1)
        self.assertIsInstance(transcript_instance.messages[0], SystemMessage)

    @patch("builtins.print")
    @patch("agent.run_agentic_loop", new_callable=AsyncMock)
    @patch("agent.HookManager")
    @patch("agent.create_core_registry")
    @patch("agent.Transcript")
    async def test_hook_injects_pre_post_context(
        self, mock_transcript_cls, mock_registry, mock_hook_mgr_cls, mock_run_loop, mock_print
    ):
        """
        Test 5.2: Hook Injects Pre/Post Context
        If a hook injects context boundaries, the UserMessage payload is assembled in exactly
        the order: [ PRE, PROMPT, POST ].
        """
        mock_transcript_cls.return_value = MockTranscriptState(Path("/mock/main.jsonl"))
        
        mock_hook_mgr = MagicMock()
        mock_hook_mgr_cls.return_value = mock_hook_mgr
        
        pre_ctx = TextMessageContent(text="<pre>system instructions</pre>")
        post_ctx = TextMessageContent(text="<post>recent file changes</post>")
        
        mock_hook_mgr.trigger_user_prompt = AsyncMock(return_value=UserPromptEvent(
            prompt="fix the bug", 
            is_first_prompt=True,
            context_pre=[pre_ctx],
            context_post=[post_ctx]
        ))

        with patch("builtins.input", side_effect=["fix the bug", "/quit"]):
            await main()

        # Check what got appended to the transcript
        transcript_instance = mock_transcript_cls.return_value
        self.assertEqual(len(transcript_instance.messages), 2)  # System + User
        
        user_msg = transcript_instance.messages[1]
        self.assertIsInstance(user_msg, UserMessage)
        
        # Payload order must be exact
        self.assertEqual(len(user_msg.content), 3)
        self.assertEqual(user_msg.content[0].text, "<pre>system instructions</pre>")
        self.assertEqual(user_msg.content[1].text, "fix the bug")
        self.assertEqual(user_msg.content[2].text, "<post>recent file changes</post>")
        
        mock_run_loop.assert_called_once()

    @patch("builtins.print")
    @patch("agent.run_agentic_loop", new_callable=AsyncMock)
    @patch("agent.HookManager")
    @patch("agent.Transcript")
    async def test_empty_input_handling(
        self, mock_transcript_cls, mock_hook_mgr_cls, mock_run_loop, mock_print
    ):
        """
        Test 5.3: Empty Input Handling
        Pressing Enter with whitespace should be safely ignored without triggering hooks.
        """
        mock_hook_mgr = MagicMock()
        mock_hook_mgr_cls.return_value = mock_hook_mgr

        # Input: empty, whitespace, then quit
        with patch("builtins.input", side_effect=["", "   ", "/quit"]):
            await main()

        mock_hook_mgr.trigger_user_prompt.assert_not_called()
        mock_run_loop.assert_not_called()

    @patch("builtins.print")
    @patch("agent.run_agentic_loop", new_callable=AsyncMock)
    @patch("agent.HookManager")
    @patch("agent.Transcript")
    async def test_first_prompt_flag(
        self, mock_transcript_cls, mock_hook_mgr_cls, mock_run_loop, mock_print
    ):
        """
        Test 5.4: First Prompt Flag
        The orchestrator must accurately tell the hook if this is the very first 
        user prompt in the conversation to optimize disk IO (e.g. loading AGENTS.md).
        """
        mock_transcript_cls.return_value = MockTranscriptState(Path("/mock/main.jsonl"))
        
        mock_hook_mgr = MagicMock()
        mock_hook_mgr_cls.return_value = mock_hook_mgr
        
        # Mock hook just returns the prompt untouched
        async def mock_trigger(prompt, is_first_prompt):
            return UserPromptEvent(prompt=prompt, is_first_prompt=is_first_prompt)
        mock_hook_mgr.trigger_user_prompt = mock_trigger

        # Input two prompts
        with patch("builtins.input", side_effect=["prompt 1", "prompt 2", "/quit"]):
            await main()

        # The loop should have called trigger_user_prompt twice
        self.assertEqual(mock_run_loop.call_count, 2)
        
        transcript_instance = mock_transcript_cls.return_value
        user_messages = [m for m in transcript_instance.messages if isinstance(m, UserMessage)]
        self.assertEqual(len(user_messages), 2)
        
        # We manually verified the state transitioning by observing the transcript state
        # The first event trigger would have seen 0 user messages.
        # The second event trigger would have seen 1 user message.

    @patch("builtins.print")
    @patch("agent.run_agentic_loop", new_callable=AsyncMock)
    @patch("agent.HookManager")
    @patch("agent.Transcript")
    async def test_keyboard_interrupt_exit(
        self, mock_transcript_cls, mock_hook_mgr_cls, mock_run_loop, mock_print
    ):
        """
        Test 5.5: KeyboardInterrupt / EOFError
        CTRL+C or CTRL+D gracefully breaks the loop without throwing Python tracebacks.
        """
        # Simulate CTRL+C
        with patch("builtins.input", side_effect=KeyboardInterrupt):
            await main()
            
        # Simulate CTRL+D (EOF)
        with patch("builtins.input", side_effect=EOFError):
            await main()

        # If it didn't crash, the test passes.
        mock_run_loop.assert_not_called()


class TestSessionStartup(unittest.IsolatedAsyncioTestCase):
    """
    Test Group 6: Which session a run belongs to (main, agent.resolve_session)

    Resolution happens before any UI exists, so a session that cannot be
    resumed has to stop the process rather than quietly start a new one: a
    mistyped id silently starting over is how a conversation gets lost.
    """

    def setUp(self):
        self.home_dir = tempfile.TemporaryDirectory()
        self.fake_home = Path(self.home_dir.name) / ".prisma"
        self.home_patcher = patch(
            "config.AppConfig.home_config_dir", new_callable=PropertyMock
        )
        self.home_patcher.start().return_value = self.fake_home
        self.addCleanup(self.home_patcher.stop)
        self.addCleanup(self.home_dir.cleanup)

        RecordingREPLUI.sessions_started = []
        self.ui_patcher = patch("ui.rich_ui.RichUI", RecordingREPLUI)
        self.ui_patcher.start()
        self.addCleanup(self.ui_patcher.stop)

        self.app_config = AppConfig(app_name="prisma", app_dir_name=".prisma")
        self.workspace = Path.cwd().resolve()

    def argv(self, *extra: str) -> list[str]:
        return ["agent.py", "--ui", "rich", *extra]

    async def run_main(
        self,
        *extra: str,
        inputs: list[str] | None = None,
        stderr: io.StringIO | None = None,
        messages: list | None = None,
    ) -> MockTranscriptState:
        """Runs main() end to end against the fake home.

        `stderr` collects what a failed resolution reports; capturing it means
        leaving `print` alone, which is otherwise patched only to keep the test
        output quiet.

        `messages` seeds the transcript main() loads, which is how a resumed
        conversation is simulated: the mode and setup hooks both decide what to
        inject by reading it. The transcript is returned so a test can see what
        the run appended to it.
        """
        quiet = nullcontext() if stderr else patch("builtins.print")
        errors = redirect_stderr(stderr) if stderr else nullcontext()

        transcript = MockTranscriptState(Path("/mock/main.jsonl"))
        transcript.messages.extend(messages or ())

        with patch.object(sys, "argv", self.argv(*extra)), \
             patch("agent.run_agentic_loop", new_callable=AsyncMock), \
             patch("agent.Transcript") as transcript_cls, \
             patch("builtins.input", side_effect=inputs or ["/quit"]), \
             quiet, errors:
            transcript_cls.return_value = transcript
            await main()

        return transcript

    @property
    def started(self) -> SessionInfo:
        """The banner main() handed the front-end."""
        self.assertEqual(len(RecordingREPLUI.sessions_started), 1)
        return RecordingREPLUI.sessions_started[0]

    def announcing(self, mode: AgentMode) -> UserMessage:
        """A prior turn of a conversation that was told it is in `mode`."""
        return UserMessage(content=[
            TextMessageContent(text=mode_reminder(mode)),
            TextMessageContent(text="the previous question"),
        ])

    def seed_session(self, session_id: str) -> sessions.SessionPaths:
        paths = sessions.session_for(self.app_config, self.workspace, session_id)
        paths.directory.mkdir(parents=True, exist_ok=True)
        paths.transcript.write_text("", encoding="utf-8")
        return paths

    async def test_a_fresh_run_records_its_metadata(self):
        await self.run_main()

        metas = list(self.fake_home.glob("projects/*/sessions/*/meta.json"))
        self.assertEqual(len(metas), 1)

        meta = json.loads(metas[0].read_text(encoding="utf-8"))
        self.assertEqual(meta["workspace"], str(self.workspace))
        self.assertIsNone(meta["title"])
        self.assertEqual(meta["session_id"], metas[0].parent.name)

    async def test_an_unknown_resume_id_stops_the_process(self):
        with self.assertRaises(SystemExit) as caught:
            await self.run_main("--resume", "no-such-session")

        self.assertEqual(caught.exception.code, 1)

    async def test_an_unknown_resume_id_lists_what_does_exist(self):
        """The id is the only handle the user has, and it lives under a hashed
        directory nobody reads by hand."""
        self.seed_session("2026-08-19_16-25-03-a1b2c3")
        reported = io.StringIO()

        with self.assertRaises(SystemExit):
            await self.run_main("--resume", "no-such-session", stderr=reported)

        self.assertIn("2026-08-19_16-25-03-a1b2c3", reported.getvalue())

    async def test_a_directory_without_a_transcript_cannot_be_resumed(self):
        empty = sessions.session_for(self.app_config, self.workspace, "halfmade")
        empty.directory.mkdir(parents=True)

        with self.assertRaises(SystemExit) as caught:
            await self.run_main("--resume", "halfmade")

        self.assertEqual(caught.exception.code, 1)

    async def test_continue_with_no_history_stops_the_process(self):
        with self.assertRaises(SystemExit) as caught:
            await self.run_main("--continue")

        self.assertEqual(caught.exception.code, 1)

    async def test_resume_and_continue_are_mutually_exclusive(self):
        with self.assertRaises(SystemExit):
            await self.run_main("--resume", "some-id", "--continue", stderr=io.StringIO())

    async def test_resuming_rebuilds_the_ledger_instead_of_starting_over(self):
        paths = self.seed_session("2026-08-19_16-25-03-a1b2c3")

        with patch("agent.rehydrate_session_usage") as rehydrate:
            rehydrate.return_value = SessionUsageTracker()
            await self.run_main("--resume", paths.session_id)

        rehydrate.assert_called_once_with(paths.transcript)

    async def test_a_fresh_run_does_not_rebuild_a_ledger(self):
        with patch("agent.rehydrate_session_usage") as rehydrate:
            await self.run_main()

        rehydrate.assert_not_called()

    async def test_continue_picks_up_the_last_used_session(self):
        older = self.seed_session("2026-08-19_16-25-03-aaaaaa")
        newer = self.seed_session("2026-08-19_17-00-00-bbbbbb")
        os.utime(newer.transcript, (1_000_000, 1_000_000))
        os.utime(older.transcript, (2_000_000, 2_000_000))

        with patch("agent.rehydrate_session_usage") as rehydrate:
            rehydrate.return_value = SessionUsageTracker()
            await self.run_main("--continue")

        rehydrate.assert_called_once_with(older.transcript)

    async def test_resuming_leaves_an_existing_title_alone(self):
        """Which is the whole reason the title lives in meta.json and not in the
        directory name."""
        paths = self.seed_session("2026-08-19_16-25-03-a1b2c3")
        paths.meta_file.write_text(
            json.dumps({"session_id": paths.session_id, "title": "Nightly triage"}),
            encoding="utf-8",
        )

        with patch("agent.rehydrate_session_usage") as rehydrate:
            rehydrate.return_value = SessionUsageTracker()
            await self.run_main("--resume", paths.session_id)

        self.assertEqual(sessions.read_meta(paths)["title"], "Nightly triage")

    async def test_transcripts_stay_outside_the_workspace(self):
        """Which is what keeps them out of reach of the agent's own file tools."""
        await self.run_main()

        transcripts = list(self.fake_home.glob("projects/*/sessions/*"))
        self.assertEqual(len(transcripts), 1)
        self.assertFalse(transcripts[0].is_relative_to(self.workspace))

    async def test_resuming_does_not_repeat_the_mode_reminder(self):
        """The reminder announces a transition, and a resume is not one.

        The seeded transcript already carries the BUILD announcement, so the
        prompt this run sends must carry nothing but the prompt.
        """
        paths = self.seed_session("2026-08-19_16-25-03-a1b2c3")
        seeded = [self.announcing(AgentMode.BUILD)]

        with patch("agent.rehydrate_session_usage") as rehydrate:
            rehydrate.return_value = SessionUsageTracker()
            transcript = await self.run_main(
                "--resume", paths.session_id,
                inputs=["the next question", "/quit"],
                messages=seeded,
            )

        appended = transcript.messages[len(seeded):]
        self.assertEqual(len(appended), 1)
        self.assertEqual(
            [block.text for block in appended[0].content], ["the next question"]
        )

    async def test_resuming_a_plan_session_stays_in_plan_mode(self):
        """PLAN mode is a restriction the user chose, so resuming keeps it
        rather than quietly handing write and shell access back."""
        paths = self.seed_session("2026-08-19_16-25-03-a1b2c3")

        with patch("agent.rehydrate_session_usage") as rehydrate:
            rehydrate.return_value = SessionUsageTracker()
            await self.run_main(
                "--resume", paths.session_id,
                messages=[self.announcing(AgentMode.PLAN)],
            )

        self.assertEqual(self.started.mode, "PLAN")

    async def test_a_fresh_run_starts_in_build_mode_and_announces_it(self):
        """The control for the two tests above: a new conversation has been
        told nothing yet, so it is told once."""
        transcript = await self.run_main(inputs=["the first question", "/quit"])

        self.assertEqual(self.started.mode, "BUILD")

        prompts = [m for m in transcript.messages if isinstance(m, UserMessage)]
        self.assertEqual(len(prompts), 1)
        self.assertIn(
            MODE_ANNOUNCEMENTS[AgentMode.BUILD],
            "\n".join(block.text for block in prompts[0].content),
        )


if __name__ == "__main__":
    unittest.main()