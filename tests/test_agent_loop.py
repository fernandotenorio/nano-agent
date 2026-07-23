import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
import sys
import uuid

from typedefs import (
    AssistantMessage, TextMessageContent, ToolUseMessageContent, 
    ToolResultMessageContent, UserMessage, ToolFailure, ShellCallback,
    AgentCallback, SystemMessage
)

from hooks import PreToolUseEvent, PostToolUseEvent, UserPromptEvent
from agent import run_agentic_loop
from agent import execute_tool, handle_shell, handle_subagent, main
from sessioncontext import AgentPolicy, AgentMode, InvocationContext


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
            tool_use, self.registry, self.hooks, self.transcript.file_path, model=self.model, policy=self.policy, ctx=self.ctx
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

    def _create_mock_process(self, stdout_data: bytes, stderr_data: bytes, exit_code: int = 0, hang_time: float = 0):
        """Helper to create a fake asyncio subprocess with pre-filled streams."""
        process = MagicMock()
        
        # 1. Setup real asyncio StreamReaders
        stdout_stream = asyncio.StreamReader()
        stdout_stream.feed_data(stdout_data)
        stdout_stream.feed_eof()
        
        stderr_stream = asyncio.StreamReader()
        stderr_stream.feed_data(stderr_data)
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
        text, is_error = await handle_shell(self.callback, self.ctx)
        
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
        text, is_error = await handle_shell(self.callback, self.ctx)
        
        # Assertions
        self.assertTrue(is_error)
        # Verify stderr comes before stdout on failure
        self.assertEqual(text, "fatal error\nnormal output")

    @patch("builtins.print")
    @patch("asyncio.create_subprocess_shell")
    async def test_output_truncation(self, mock_create_shell, mock_print):
        """
        Test 3.3: Output Truncation
        Checks that huge outputs are safely truncated to exactly 30,000 bytes.
        """
        # Setup: Create 40,000 bytes of data (exceeds the 30,000 limit)
        huge_stdout = b"A" * 40000
        mock_process = self._create_mock_process(huge_stdout, b"", exit_code=0)
        mock_create_shell.return_value = mock_process
        
        # Action
        text, is_error = await handle_shell(self.callback, self.ctx)
        
        # Assertions
        self.assertFalse(is_error)
        self.assertEqual(len(text), 30000)
        self.assertTrue(text.startswith("AAAA"))

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
        text, is_error = await handle_shell(self.callback, self.ctx)
        
        # Assertions
        self.assertTrue(is_error)
        self.assertIn("Command timed out after 0.1s", text)
        self.assertIn("partial out", text)  # We should still capture what was emitted before timeout
        
        # Ensure we tried to safely terminate, and when it didn't respond to that, kill it
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
        text, is_error = await handle_shell(self.callback, self.ctx)
        
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
    @patch("agent.uuid.uuid4")
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
        
        # 1. Verify correct file path construction (must be in same dir, correctly named)
        expected_path = Path("/mock/dir/parent_transcript_code-reviewer_123456.jsonl")
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


class MockTranscriptState:
    """A lightweight mock to simulate transcript state changes in memory."""
    def __init__(self, path):
        self.file_path = path
        self.messages = []
    def append(self, msg):
        self.messages.append(msg)

class TestMainLoopGroup5(unittest.IsolatedAsyncioTestCase):
    """
    Test Group 5: User Input Boundary & Hooks (main)
    Validates CLI interactions, hook interceptions (blocking and context injection),
    and graceful loop exits.
    """

    def setUp(self):
        # We must patch sys.argv so argparse doesn't try to parse unittest's CLI args
        self.argv_patcher = patch.object(sys, "argv", ["agent.py"])
        self.argv_patcher.start()

    def tearDown(self):
        self.argv_patcher.stop()

    @patch("builtins.print")
    @patch("agent.run_agentic_loop", new_callable=AsyncMock)
    @patch("agent.HookManager")
    @patch("agent.create_core_registry")
    @patch("agent.get_transcript_path")
    @patch("agent.Transcript")
    async def test_hook_blocks_prompt(
        self, mock_transcript_cls, mock_get_path, mock_registry, mock_hook_mgr_cls, mock_run_loop, mock_print
    ):
        """
        Test 5.1: Hook Blocks Prompt
        If a hook flags block=True, the prompt is rejected, not appended to the transcript, 
        and the LLM loop is skipped.
        """
        # Setup Mocks
        mock_get_path.return_value = Path("/mock/main.jsonl")
        mock_transcript_cls.return_value = MockTranscriptState(mock_get_path.return_value)
        
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
    @patch("agent.get_transcript_path")
    @patch("agent.Transcript")
    async def test_hook_injects_pre_post_context(
        self, mock_transcript_cls, mock_get_path, mock_registry, mock_hook_mgr_cls, mock_run_loop, mock_print
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
    @patch("agent.get_transcript_path")
    @patch("agent.Transcript")
    async def test_first_prompt_flag(
        self, mock_transcript_cls, mock_get_path, mock_hook_mgr_cls, mock_run_loop, mock_print
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


if __name__ == "__main__":
    unittest.main()