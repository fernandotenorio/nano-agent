import unittest
from unittest.mock import patch

from hooks import HookManager, initial_setup_hook, UserPromptEvent
from typedefs import TextMessageContent


class TestHooks(unittest.IsolatedAsyncioTestCase):
    """
    Test Suite for Hook Manager (hooks.py)
    Validates hook execution chaining, context modification, short-circuit blocking, 
    and the built-in initial_setup_hook (CLAUDE.md injector).
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
        
        result = await self.mgr.trigger_pre_tool("Bash", {"cmd": "rm -rf"})
        
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
        
        result = await self.mgr.trigger_post_tool("Bash", {}, "command output")
        
        self.assertEqual(len(result.additional_context), 2)
        self.assertEqual(result.additional_context[0].text, "X-Context")
        self.assertEqual(result.additional_context[1].text, "Y-Context")


    # ---------------------------------------------------------
    # GROUP 4: Built-in CLAUDE.md Hook (initial_setup_hook)
    # ---------------------------------------------------------

    @patch("hooks.Path")
    async def test_claude_md_fast_exit(self, mock_path):
        """Test 4.1: Hook exits immediately with zero IO if is_first_prompt=False."""
        event = UserPromptEvent(prompt="continue task", is_first_prompt=False)
        
        result = await initial_setup_hook(event)
        
        # Assert Path("CLAUDE.md") was NEVER even instantiated
        mock_path.assert_not_called()
        self.assertEqual(len(result.context_pre), 0)

    @patch("hooks.Path")
    async def test_claude_md_missing(self, mock_path):
        """Test 4.2: Hook degrades gracefully if CLAUDE.md does not exist."""
        # Setup Path("CLAUDE.md").exists() to return False
        mock_path_instance = mock_path.return_value
        mock_path_instance.exists.return_value = False
        
        event = UserPromptEvent(prompt="start task", is_first_prompt=True)
        result = await initial_setup_hook(event)
        
        # Verify it checked for existence, but didn't read
        mock_path_instance.exists.assert_called_once()
        mock_path_instance.read_text.assert_not_called()
        self.assertEqual(len(result.context_pre), 0)

    @patch("hooks.Path")
    async def test_claude_md_injection_success(self, mock_path):
        """Test 4.3: Hook successfully reads and wraps CLAUDE.md text."""
        # Setup Path("CLAUDE.md") mock
        mock_path_instance = mock_path.return_value
        mock_path_instance.exists.return_value = True
        mock_path_instance.read_text.return_value = "Always write unit tests."
        
        event = UserPromptEvent(prompt="start task", is_first_prompt=True)
        result = await initial_setup_hook(event)
        
        self.assertEqual(len(result.context_pre), 1)
        
        # Verify XML/wrapper formatting
        injected_text = result.context_pre[0].text
        self.assertIn("<system-reminder>", injected_text)
        self.assertIn("Always write unit tests.", injected_text)
        self.assertIn("</system-reminder>", injected_text)


if __name__ == "__main__":
    unittest.main()