import unittest
from unittest.mock import patch

from capabilities import Capabilities
from hooks import (
    HookManager,
    MODE_ANNOUNCEMENTS,
    PLAN_ACCEPTED_TO_BUILD,
    UserPromptEvent,
    agent_mode_hook,
    capabilities_hook,
    initial_setup_hook,
    last_notified_mode,
    mode_reminder,
    restore_policy,
)
from sessioncontext import AgentMode, AgentPolicy
from tools.ignore import GitStatus
from typedefs import (
    AssistantMessage,
    SystemMessage,
    TextMessageContent,
    ToolResultMessageContent,
    UserMessage,
)


class TestHooks(unittest.IsolatedAsyncioTestCase):
    """
    Test Suite for Hook Manager (hooks.py)
    Validates hook execution chaining, context modification, short-circuit blocking, 
    and the built-in initial_setup_hook (AGENTS.md injector).
    """

    def setUp(self):
        self.mgr = HookManager()

    # ---------------------------------------------------------
    # GROUP 1: User Prompt Hooks
    # ---------------------------------------------------------

    async def test_user_prompt_single_hook(self):
        """Test 1.1: A single hook can append pre/post context."""
        async def mock_hook(event: UserPromptEvent):
            event.context_pre.append(TextMessageContent(text="Pre-context"))
            return event
            
        self.mgr.register_user_prompt(mock_hook)
        
        result = await self.mgr.trigger_user_prompt("hello", is_first_prompt=True)
        
        self.assertEqual(len(result.context_pre), 1)
        self.assertEqual(result.context_pre[0].text, "Pre-context")
        self.assertEqual(result.prompt, "hello")

    async def test_user_prompt_hook_chaining(self):
        """Test 1.2: Multiple hooks sequentially modify the same event payload."""
        async def hook_a(event: UserPromptEvent):
            event.context_post.append(TextMessageContent(text="[A]"))
            return event
            
        async def hook_b(event: UserPromptEvent):
            event.context_post.append(TextMessageContent(text="[B]"))
            return event
            
        self.mgr.register_user_prompt(hook_a)
        self.mgr.register_user_prompt(hook_b)
        
        result = await self.mgr.trigger_user_prompt("hello", is_first_prompt=True)
        
        self.assertEqual(len(result.context_post), 2)
        self.assertEqual(result.context_post[0].text, "[A]")
        self.assertEqual(result.context_post[1].text, "[B]")

    async def test_user_prompt_short_circuit_on_block(self):
        """Test 1.3: If a hook flags block=True, subsequent hooks are skipped."""
        call_order = []
        
        async def hook_1(e): call_order.append(1); return e
        async def hook_2(e): call_order.append(2); e.block = True; return e
        async def hook_3(e): call_order.append(3); return e
        
        self.mgr.register_user_prompt(hook_1)
        self.mgr.register_user_prompt(hook_2)
        self.mgr.register_user_prompt(hook_3)
        
        result = await self.mgr.trigger_user_prompt("hello", is_first_prompt=True)
        
        # Hook 3 should never be reached!
        self.assertEqual(call_order, [1, 2])
        self.assertTrue(result.block)


    # ---------------------------------------------------------
    # GROUP 2: Pre-Tool Hooks
    # ---------------------------------------------------------

    async def test_pre_tool_default_allow(self):
        """Test 2.1: With no hooks registered, execution defaults to allow."""
        result = await self.mgr.trigger_pre_tool("Read", {"file": "main.py"})
        self.assertEqual(result.decision, "allow")

    async def test_pre_tool_short_circuit_on_deny(self):
        """Test 2.2: If a hook denies tool execution, subsequent hooks are skipped."""
        call_order = []
        
        async def hook_a(e): call_order.append('A'); return e
        async def hook_b(e): call_order.append('B'); e.decision = "deny"; e.deny_reason = "Unsafe"; return e
        async def hook_c(e): call_order.append('C'); return e
        
        self.mgr.register_pre_tool(hook_a)
        self.mgr.register_pre_tool(hook_b)
        self.mgr.register_pre_tool(hook_c)
        
        result = await self.mgr.trigger_pre_tool("Shell", {"cmd": "rm -rf"})
        
        # Hook C should never be reached!
        self.assertEqual(call_order, ['A', 'B'])
        self.assertEqual(result.decision, "deny")
        self.assertEqual(result.deny_reason, "Unsafe")


    # ---------------------------------------------------------
    # GROUP 3: Post-Tool Hooks
    # ---------------------------------------------------------

    async def test_post_tool_context_accumulation(self):
        """Test 3.1: Post-tool hooks accumulate extra context successfully."""
        async def hook_x(e): 
            e.additional_context.append(TextMessageContent(text="X-Context"))
            return e
            
        async def hook_y(e): 
            e.additional_context.append(TextMessageContent(text="Y-Context"))
            return e
            
        self.mgr.register_post_tool(hook_x)
        self.mgr.register_post_tool(hook_y)
        
        result = await self.mgr.trigger_post_tool("Shell", {}, "command output")
        
        self.assertEqual(len(result.additional_context), 2)
        self.assertEqual(result.additional_context[0].text, "X-Context")
        self.assertEqual(result.additional_context[1].text, "Y-Context")


    # ---------------------------------------------------------
    # GROUP 4: Built-in AGENTS.md Hook (initial_setup_hook)
    # ---------------------------------------------------------

    @patch("hooks.gather_context_files")
    async def test_context_injection_fast_exit(self, mock_gather):
        """Test 4.1: Hook exits immediately with zero IO if is_first_prompt=False."""
        from config import AppConfig
        from pathlib import Path
        
        event = UserPromptEvent(prompt="continue task", is_first_prompt=False)
        app_config = AppConfig(app_name="test", app_dir_name=".test")
        root = Path("/dummy")
        cwd = Path("/dummy")
        
        result = await initial_setup_hook(event, app_config, root, cwd)
        
        # Assert gather_context_files was NEVER called
        mock_gather.assert_not_called()
        self.assertEqual(len(result.context_pre), 0)

    @patch("hooks.gather_context_files")
    async def test_context_injection_empty(self, mock_gather):
        """Test 4.2: Hook degrades gracefully if no AGENTS.md files are found."""
        from config import AppConfig
        from pathlib import Path
        
        # Setup gather_context_files to return an empty string
        mock_gather.return_value = ""
        
        event = UserPromptEvent(prompt="start task", is_first_prompt=True)
        app_config = AppConfig(app_name="test", app_dir_name=".test")
        root = Path("/dummy")
        cwd = Path("/dummy")
        
        result = await initial_setup_hook(event, app_config, root, cwd)
        
        # Verify it was called with the right arguments, but injected nothing
        mock_gather.assert_called_once_with(app_config, root, cwd)
        self.assertEqual(len(result.context_pre), 0)

    @patch("hooks.gather_context_files")
    async def test_context_injection_success(self, mock_gather):
        """Test 4.3: Hook successfully reads and wraps AGENTS.md text."""
        from config import AppConfig
        from pathlib import Path
        
        # Setup gather_context_files mock
        mock_gather.return_value = "Always write unit tests."
        
        event = UserPromptEvent(prompt="start task", is_first_prompt=True)
        app_config = AppConfig(app_name="test", app_dir_name=".test")
        root = Path("/dummy")
        cwd = Path("/dummy/src")
        
        result = await initial_setup_hook(event, app_config, root, cwd)
        
        self.assertEqual(len(result.context_pre), 1)
        
        # Verify XML/wrapper formatting
        injected_text = result.context_pre[0].text
        self.assertIn("<system-reminder>", injected_text)
        self.assertIn("Always write unit tests.", injected_text)
        self.assertIn("</system-reminder>", injected_text)

    # ---------------------------------------------------------
    # GROUP 5: Built-in capabilities_hook
    # ---------------------------------------------------------

    async def test_degraded_git_is_announced_on_the_first_prompt(self):
        event = UserPromptEvent(prompt="start task", is_first_prompt=True)
        capabilities = Capabilities(git_status=GitStatus.UNAVAILABLE, git_error="no git")

        result = await capabilities_hook(event, capabilities)

        self.assertEqual(len(result.context_pre), 1)
        injected_text = result.context_pre[0].text
        self.assertIn("<system-reminder>", injected_text)
        self.assertIn(".gitignore", injected_text)

    async def test_nothing_is_announced_later_in_the_conversation(self):
        """The warning belongs to the session, so it is stated once."""
        event = UserPromptEvent(prompt="next task", is_first_prompt=False)
        capabilities = Capabilities(git_status=GitStatus.UNAVAILABLE)

        result = await capabilities_hook(event, capabilities)

        self.assertEqual(len(result.context_pre), 0)

    async def test_healthy_environment_injects_nothing(self):
        event = UserPromptEvent(prompt="start task", is_first_prompt=True)
        capabilities = Capabilities(ripgrep="/usr/bin/rg", git_status=GitStatus.OK)

        result = await capabilities_hook(event, capabilities)

        self.assertEqual(len(result.context_pre), 0)

    async def test_missing_ripgrep_alone_injects_nothing(self):
        """A missing rg changes nothing the model can observe, so it is not told.

        This hook exists because a resumed transcript carries a stale system
        prompt; repeating a warning the model cannot act on would only add noise.
        """
        event = UserPromptEvent(prompt="start task", is_first_prompt=True)
        capabilities = Capabilities(ripgrep=None, git_status=GitStatus.OK)

        result = await capabilities_hook(event, capabilities)

        self.assertEqual(len(result.context_pre), 0)


# ---------------------------------------------------------
# GROUP 6: Built-in agent_mode_hook, and resuming its state
# ---------------------------------------------------------

class TestAgentModeHook(unittest.IsolatedAsyncioTestCase):
    """
    Test Suite for the mode reminder (agent_mode_hook) and for rebuilding a
    resumed session's policy from its transcript (restore_policy).

    The reminder announces a *transition*, so it has to fire exactly once per
    transition. That includes across a --resume: the announcement lives in the
    transcript, while the policy that remembers making it does not, so a
    resumed run has to read it back rather than start over.
    """

    def announcing(self, mode: AgentMode) -> UserMessage:
        """A transcript message shaped like the one the hook produces.

        The reminder arrives as `context_pre`, so it precedes the user's own
        text in the same message; both are plain text blocks.
        """
        return UserMessage(content=[
            TextMessageContent(text=mode_reminder(mode)),
            TextMessageContent(text="carry on"),
        ])

    def accepting_a_plan(self) -> UserMessage:
        """The tool result recorded when a plan is approved into BUILD mode."""
        return UserMessage(content=[
            ToolResultMessageContent(
                tool_use_id="call_1",
                content=PLAN_ACCEPTED_TO_BUILD,
                tool_name="SubmitPlan",
            )
        ])

    async def fire(self, policy: AgentPolicy, is_first_prompt: bool = False) -> UserPromptEvent:
        return await agent_mode_hook(
            UserPromptEvent(prompt="carry on", is_first_prompt=is_first_prompt), policy
        )

    # --- The hook -------------------------------------------------------

    async def test_the_starting_mode_is_announced(self):
        """A policy that has told the model nothing announces where it stands."""
        result = await self.fire(AgentPolicy(mode=AgentMode.BUILD), is_first_prompt=True)

        self.assertEqual(len(result.context_pre), 1)
        self.assertIn(MODE_ANNOUNCEMENTS[AgentMode.BUILD], result.context_pre[0].text)

    async def test_the_reminder_is_not_repeated_while_the_mode_holds(self):
        policy = AgentPolicy(mode=AgentMode.BUILD)

        await self.fire(policy, is_first_prompt=True)
        result = await self.fire(policy)

        self.assertEqual(result.context_pre, [])

    async def test_a_switch_into_plan_mode_is_announced(self):
        policy = AgentPolicy(mode=AgentMode.PLAN, notified_mode=AgentMode.BUILD)

        result = await self.fire(policy)

        self.assertEqual(len(result.context_pre), 1)
        self.assertIn(MODE_ANNOUNCEMENTS[AgentMode.PLAN], result.context_pre[0].text)

    # --- Resuming -------------------------------------------------------

    async def test_resuming_does_not_repeat_the_reminder(self):
        """The regression: a resumed session used to re-announce its own mode.

        The transcript already carries the announcement, so repeating it tells
        the model something it has read, on every resume.
        """
        policy = restore_policy([self.announcing(AgentMode.BUILD)])

        result = await self.fire(policy)

        self.assertEqual(result.context_pre, [])

    def test_a_session_left_in_plan_mode_resumes_in_plan_mode(self):
        """PLAN mode is a restriction the user chose; resuming keeps it."""
        policy = restore_policy([self.announcing(AgentMode.PLAN)])

        self.assertIs(policy.mode, AgentMode.PLAN)
        self.assertIs(policy.notified_mode, AgentMode.PLAN)

    def test_an_accepted_plan_resumes_in_build_mode(self):
        """Approving a plan is announced by a tool result, not by a reminder.

        The newest *reminder* in such a transcript still says PLAN, so reading
        only reminders would revoke the write access the user just granted.
        """
        policy = restore_policy([
            self.announcing(AgentMode.PLAN),
            self.accepting_a_plan(),
        ])

        self.assertIs(policy.mode, AgentMode.BUILD)
        self.assertIs(policy.notified_mode, AgentMode.BUILD)

    def test_the_most_recent_announcement_wins(self):
        policy = restore_policy([
            self.announcing(AgentMode.PLAN),
            self.announcing(AgentMode.BUILD),
        ])

        self.assertIs(policy.mode, AgentMode.BUILD)

    async def test_a_fresh_transcript_still_gets_its_first_reminder(self):
        policy = restore_policy([])

        self.assertIs(policy.mode, AgentMode.BUILD)
        self.assertIsNone(policy.notified_mode)

        result = await self.fire(policy, is_first_prompt=True)
        self.assertEqual(len(result.context_pre), 1)

    def test_only_user_messages_are_examined(self):
        """The reminder rides on a user message; nothing else can announce."""
        messages = [
            SystemMessage(content=mode_reminder(AgentMode.PLAN)),
            AssistantMessage(content=[TextMessageContent(text=mode_reminder(AgentMode.PLAN))]),
        ]

        self.assertIsNone(last_notified_mode(messages))

    def test_a_string_bodied_message_is_ignored(self):
        """Transcripts predating block content must not crash a resume."""
        self.assertIsNone(last_notified_mode([UserMessage(content="carry on")]))

    def test_a_user_repeating_the_announcement_is_not_an_announcement(self):
        """Typed text and injected context are indistinguishable by type, so
        the reminder wrapper is what makes an announcement official."""
        typed = UserMessage(content=[
            TextMessageContent(text=f"{MODE_ANNOUNCEMENTS[AgentMode.PLAN]} Or so I am told.")
        ])

        self.assertIsNone(last_notified_mode([typed]))

    def test_a_failed_tool_result_is_not_an_acceptance(self):
        rejected = UserMessage(content=[
            ToolResultMessageContent(
                tool_use_id="call_1",
                content=PLAN_ACCEPTED_TO_BUILD,
                is_error=True,
                tool_name="SubmitPlan",
            )
        ])

        self.assertIsNone(last_notified_mode([rejected]))

    def test_every_reminder_can_be_read_back(self):
        """Guards the two halves against drifting apart: whatever the hook
        writes for a mode is what the recovery has to recognise."""
        for mode in AgentMode:
            with self.subTest(mode=mode):
                self.assertIs(last_notified_mode([self.announcing(mode)]), mode)


if __name__ == "__main__":
    unittest.main()