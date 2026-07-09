import unittest
from unittest.mock import MagicMock

# Import the target module functions and internal types
from tools.shell import _bash_impl, register_shell_tools
from typedefs import ToolFailure, ShellCallback


class TestShellTool(unittest.IsolatedAsyncioTestCase):
    """
    Test Suite for Shell Tool (shell.py)
    Validates command parsing, timeout conversion, argument handling,
    and schema registration.
    """

    # ---------------------------------------------------------
    # GROUP 1: Validation & Execution (_bash_impl)
    # ---------------------------------------------------------

    async def test_missing_command_argument(self):
        """Test 1.1: _bash_impl returns ToolFailure if 'command' is missing."""
        result = await _bash_impl({})
        self.assertIsInstance(result, ToolFailure)
        self.assertIn("command is required", result.error_message)

    async def test_empty_string_command(self):
        """Test 1.2: _bash_impl returns ToolFailure if 'command' is empty/whitespace."""
        result = await _bash_impl({"command": "   "})
        self.assertIsInstance(result, ToolFailure)
        self.assertIn("command is required", result.error_message)

    async def test_default_timeout_parsing(self):
        """Test 1.3: _bash_impl defaults timeout to 120.0 seconds."""
        result = await _bash_impl({"command": "ls"})
        self.assertIsInstance(result, ShellCallback)
        self.assertEqual(result.command, "ls")
        self.assertEqual(result.timeout, 120.0)  # Default 120000ms / 1000

    async def test_explicit_timeout_parsing(self):
        """Test 1.4: _bash_impl correctly converts ms timeout to seconds."""
        result = await _bash_impl({"command": "sleep 5", "timeout": 5000})
        self.assertIsInstance(result, ShellCallback)
        self.assertEqual(result.command, "sleep 5")
        self.assertEqual(result.timeout, 5.0)  # 5000ms / 1000

    async def test_optional_description_parsing(self):
        """Test 1.5: _bash_impl captures the optional description field."""
        result = await _bash_impl({
            "command": "python -m pytest", 
            "description": "Running unit tests"
        })
        self.assertIsInstance(result, ShellCallback)
        self.assertEqual(result.command, "python -m pytest")
        self.assertEqual(result.callback_description, "Running unit tests")
        self.assertEqual(result.timeout, 120.0) # Ensure other defaults are kept


    # ---------------------------------------------------------
    # GROUP 2: Tool Schema Registration (register_shell_tools)
    # ---------------------------------------------------------

    def test_registry_binding(self):
        """Test 2.1: register_shell_tools calls registry.register exactly once."""
        mock_registry = MagicMock()
        register_shell_tools(mock_registry)
        mock_registry.register.assert_called_once()

    def test_json_schema_structure(self):
        """Test 2.2: The registered JSON schema is accurate and contains safety warnings."""
        mock_registry = MagicMock()
        register_shell_tools(mock_registry)
        
        # Extract the arguments passed to registry.register()
        call_kwargs = mock_registry.register.call_args[1]
        
        self.assertEqual(call_kwargs["name"], "Shell")
        
        input_schema = call_kwargs["input_schema"]
        self.assertEqual(input_schema["type"], "object")
        self.assertEqual(input_schema["required"], ["command"])
        
        properties = input_schema["properties"]
        self.assertIn("command", properties)
        self.assertEqual(properties["command"]["type"], "string")
        self.assertIn("description", properties)
        self.assertEqual(properties["description"]["type"], "string")
        self.assertIn("timeout", properties)
        self.assertEqual(properties["timeout"]["type"], "number")
        
        # Verify the presence of the critical safety warning in the description
        description = call_kwargs["description"]
        self.assertIn("BE CAREFUL! THERE ARE MANY DANGEROUS BASH COMMANDS", description)
        self.assertIn("Do not use shell commands if you have an equivalent tool at your disposal.", description)


if __name__ == "__main__":
    unittest.main()