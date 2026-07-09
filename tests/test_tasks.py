import unittest
from unittest.mock import patch, MagicMock

# Import the target module functions and state
from tools.tasks import (
    _task_impl, register_tasks_tools, get_subagent_system_prompt, _SUB_AGENTS
)
from typedefs import ToolFailure, AgentCallback


class TestSubAgentTasks(unittest.IsolatedAsyncioTestCase):
    """
    Test Suite for Sub-Agent Tasks (tasks.py)
    Validates parameter routing, profile selection, CLAUDE.md injection,
    and dynamic LLM schema generation.
    """

    # ---------------------------------------------------------
    # GROUP 1: Validation & Error Handling
    # ---------------------------------------------------------

    async def test_missing_prompt(self):
        """Test 1.1: Task safely rejects if 'prompt' is missing."""
        result = await _task_impl({"subagent_type": "default-agent"})
        self.assertIsInstance(result, ToolFailure)
        self.assertIn("prompt is required", result.error_message)

    async def test_unrecognized_subagent_type(self):
        """Test 1.2: Task rejects invalid types and dynamically lists valid ones."""
        result = await _task_impl({"prompt": "do work", "subagent_type": "hacker-agent"})
        self.assertIsInstance(result, ToolFailure)
        self.assertIn("not recognized", result.error_message)
        
        # Verify the fallback list contains actual available agents
        self.assertIn("default-agent", result.error_message)
        self.assertIn("code-reviewer", result.error_message)


    # ---------------------------------------------------------
    # GROUP 2: Successful Profile Routing
    # ---------------------------------------------------------

    @patch("tools.tasks.Path")
    async def test_default_fallbacks(self, mock_path):
        """Test 2.1: Missing optional parameters default to standard values."""
        # Ensure no CLAUDE.md interference
        mock_path.return_value.exists.return_value = False
        
        result = await _task_impl({"prompt": "Explore the code."})
        
        self.assertIsInstance(result, AgentCallback)
        self.assertEqual(result.subagent_type, "default-agent")
        self.assertEqual(result.callback_description, "Delegated sub-task")
        self.assertIsNone(result.tools)  # Means access to ALL tools
        self.assertEqual(result.user_content, "Explore the code.")

    @patch("tools.tasks.Path")
    async def test_explicit_profile_request(self, mock_path):
        """Test 2.2: Explicit requests correctly fetch the mapped profile data."""
        mock_path.return_value.exists.return_value = False
        
        result = await _task_impl({
            "prompt": "Review PR", 
            "subagent_type": "code-reviewer",
            "description": "Checking for security bugs"
        })
        
        self.assertIsInstance(result, AgentCallback)
        self.assertEqual(result.subagent_type, "code-reviewer")
        self.assertEqual(result.callback_description, "Checking for security bugs")
        self.assertEqual(result.tools, ["Read", "Bash"])  # Strictly restricted


    # ---------------------------------------------------------
    # GROUP 3: Context Injection (CLAUDE.md)
    # ---------------------------------------------------------

    @patch("tools.tasks.Path")
    async def test_context_injection_file_missing(self, mock_path):
        """Test 3.1: If CLAUDE.md is missing, prompt is untouched."""
        mock_path_instance = mock_path.return_value
        mock_path_instance.exists.return_value = False
        
        result = await _task_impl({"prompt": "Raw prompt."})
        
        self.assertIsInstance(result, AgentCallback)
        self.assertEqual(result.user_content, "Raw prompt.")
        mock_path_instance.read_text.assert_not_called()

    @patch("tools.tasks.Path")
    async def test_context_injection_success(self, mock_path):
        """Test 3.2: If CLAUDE.md exists, it wraps into the prompt automatically."""
        mock_path_instance = mock_path.return_value
        mock_path_instance.exists.return_value = True
        mock_path_instance.read_text.return_value = "Use 4 spaces."
        
        result = await _task_impl({"prompt": "Fix formatting."})
        
        self.assertIsInstance(result, AgentCallback)
        self.assertIn("<system-reminder>", result.user_content)
        self.assertIn("Project instructions:", result.user_content)
        self.assertIn("Use 4 spaces.", result.user_content)
        self.assertIn("</system-reminder>", result.user_content)
        
        # Ensure the actual LLM prompt is at the END of the string
        self.assertTrue(result.user_content.endswith("Fix formatting."))


    # ---------------------------------------------------------
    # GROUP 4: System Prompt & Environment
    # ---------------------------------------------------------

    @patch("tools.tasks.get_environment_details")
    def test_environment_appending(self, mock_env):
        """Test 4.1: System prompts correctly combine core identity with dynamic OS env."""
        mock_env.return_value = "<env>Mac OS</env>"
        
        # Grab the first profile in the list (default-agent)
        profile = _SUB_AGENTS[0]
        
        sys_prompt = get_subagent_system_prompt(profile)
        
        self.assertTrue(sys_prompt.startswith(profile.core_system_prompt))
        self.assertTrue(sys_prompt.endswith("<env>Mac OS</env>"))


    # ---------------------------------------------------------
    # GROUP 5: Dynamic Tool Registration
    # ---------------------------------------------------------

    def test_schema_generation_matches_state(self):
        """Test 5.1: The registry description dynamically advertises all available profiles."""
        mock_registry = MagicMock()
        
        register_tasks_tools(mock_registry)
        
        mock_registry.register.assert_called_once()
        
        # Extract the kwargs passed to registry.register()
        call_kwargs = mock_registry.register.call_args[1]
        
        self.assertEqual(call_kwargs["name"], "Task")
        description = call_kwargs["description"]
        
        # Verify it lists the default-agent and says it has all tools (*)
        self.assertIn("default-agent", description)
        self.assertIn("(Tools: *)", description)
        
        # Verify it lists the code-reviewer and documents its restricted tools
        self.assertIn("code-reviewer", description)
        self.assertIn("Tools: Read, Bash", description)


if __name__ == "__main__":
    unittest.main()