import unittest
from unittest.mock import AsyncMock

from tools.registry import ToolRegistry


class TestToolRegistry(unittest.IsolatedAsyncioTestCase):
    """
    Test Suite for Tool Registry (registry.py)
    Validates tool registration, schema retrieval, dynamic execution routing,
    and safe error handling during execution.
    """

    def setUp(self):
        self.registry = ToolRegistry()

    # ---------------------------------------------------------
    # GROUP 1: Registration and Schema Retrieval
    # ---------------------------------------------------------

    def test_tool_registration(self):
        """Test 1.1: Tools are stored with proper OpenAI function schema formatting."""
        mock_func = AsyncMock()
        
        self.registry.register(
            name="TestTool",
            description="A test tool.",
            input_schema={"type": "object", "properties": {}},
            func=mock_func
        )
        
        # Verify callable mapping
        self.assertIn("TestTool", self.registry._callables)
        self.assertEqual(self.registry._callables["TestTool"], mock_func)
        
        # Verify schema mapping format
        self.assertIn("TestTool", self.registry._tools)
        schema = self.registry._tools["TestTool"]
        
        self.assertEqual(schema["type"], "function")
        self.assertEqual(schema["function"]["name"], "TestTool")
        self.assertEqual(schema["function"]["description"], "A test tool.")
        self.assertEqual(schema["function"]["parameters"], {"type": "object", "properties": {}})

    def test_get_all_schemas(self):
        """Test 1.2: get_all_schemas returns a list of all registered tool schemas."""
        self.registry.register("ToolA", "Desc A", {}, AsyncMock())
        self.registry.register("ToolB", "Desc B", {}, AsyncMock())
        
        schemas = self.registry.get_all_schemas()
        
        self.assertEqual(len(schemas), 2)
        names = [s["function"]["name"] for s in schemas]
        self.assertIn("ToolA", names)
        self.assertIn("ToolB", names)


    # ---------------------------------------------------------
    # GROUP 2: Dynamic Invocation (invoke)
    # ---------------------------------------------------------

    async def test_successful_invocation(self):
        """Test 2.1: invoke correctly routes kwargs to the underlying async function."""
        mock_func = AsyncMock(return_value="Success Result")
        self.registry.register("MyTool", "Desc", {}, mock_func)
        
        result = await self.registry.invoke("MyTool", {"arg1": 123})
        
        self.assertEqual(result, "Success Result")
        mock_func.assert_called_once_with({"arg1": 123})

    async def test_missing_tool_invocation(self):
        """Test 2.2: invoke safely handles requests for tools that do not exist."""
        result = await self.registry.invoke("GhostTool", {})
        
        self.assertIsInstance(result, str)
        self.assertEqual(result, "Error: Tool 'GhostTool' not found.")

    async def test_exception_handling_during_invocation(self):
        """Test 2.3: native Python exceptions inside tools are caught and converted to error strings."""
        mock_func = AsyncMock(side_effect=ValueError("Corrupted state variables!"))
        self.registry.register("CrashTool", "Desc", {}, mock_func)
        
        result = await self.registry.invoke("CrashTool", {})
        
        self.assertIsInstance(result, str)
        self.assertEqual(result, "Error: tool 'CrashTool': Corrupted state variables!")


    # ---------------------------------------------------------
    # GROUP 3: Tool Filtering (clone_filtered)
    # ---------------------------------------------------------

    def test_valid_filtering(self):
        """Test 3.1: clone_filtered returns a new registry with strictly the allowed tools."""
        self.registry.register("ToolA", "Desc A", {}, AsyncMock())
        self.registry.register("ToolB", "Desc B", {}, AsyncMock())
        self.registry.register("ToolC", "Desc C", {}, AsyncMock())
        
        # Restrict to A and C
        sub_registry = self.registry.clone_filtered(["ToolA", "ToolC"])
        
        # Verify it's a completely new instance
        self.assertNotEqual(self.registry, sub_registry)
        
        schemas = sub_registry.get_all_schemas()
        self.assertEqual(len(schemas), 2)
        
        names = [s["function"]["name"] for s in schemas]
        self.assertIn("ToolA", names)
        self.assertIn("ToolC", names)
        self.assertNotIn("ToolB", names)

    def test_filtering_with_missing_tools(self):
        """Test 3.2: clone_filtered silently ignores unrecognized tool names without crashing."""
        self.registry.register("ToolA", "Desc A", {}, AsyncMock())
        
        # "UnknownTool" is not in the parent registry
        sub_registry = self.registry.clone_filtered(["ToolA", "UnknownTool"])
        
        schemas = sub_registry.get_all_schemas()
        
        # It successfully cloned A, and safely ignored UnknownTool
        self.assertEqual(len(schemas), 1)
        self.assertEqual(schemas[0]["function"]["name"], "ToolA")


if __name__ == "__main__":
    unittest.main()