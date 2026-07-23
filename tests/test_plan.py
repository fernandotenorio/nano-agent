import unittest
from unittest.mock import MagicMock
from pathlib import Path

from sessioncontext import InvocationContext
from tools.plan import _plan_impl, register_plan_tools
from typedefs import PlanApprovalCallback, ToolFailure


class TestPlanTool(unittest.IsolatedAsyncioTestCase):
    """
    Test Suite for the SubmitPlan tool (plan.py).
    Validates argument handling and schema registration.
    """

    def setUp(self):
        self.ctx = InvocationContext(
            workspace=Path("/dummy/workspace"),
            cwd=Path("/dummy/workspace"),
            workspace_is_git_repo=False,
            resume_file=None
        )

    # ---------------------------------------------------------
    # GROUP 1: Validation & Execution (_plan_impl)
    # ---------------------------------------------------------

    async def test_missing_plan_summary(self):
        result = await _plan_impl({}, self.ctx)
        self.assertIsInstance(result, ToolFailure)
        self.assertIn("plan_summary is required", result.error_message)

    async def test_empty_plan_summary(self):
        result = await _plan_impl({"plan_summary": ""}, self.ctx)
        self.assertIsInstance(result, ToolFailure)
        self.assertIn("plan_summary is required", result.error_message)

    async def test_whitespace_plan_summary(self):
        result = await _plan_impl({"plan_summary": "   \n  "}, self.ctx)
        self.assertIsInstance(result, ToolFailure)
        self.assertIn("plan_summary is required", result.error_message)

    async def test_non_string_plan_summary(self):
        result = await _plan_impl({"plan_summary": 42}, self.ctx)
        self.assertIsInstance(result, ToolFailure)
        self.assertIn("plan_summary is required", result.error_message)

    async def test_valid_plan_returns_callback(self):
        plan_text = "1. Add helper\n2. Wire it into the tools\n3. Add tests"
        result = await _plan_impl({"plan_summary": plan_text}, self.ctx)

        self.assertIsInstance(result, PlanApprovalCallback)
        self.assertEqual(result.plan_summary, plan_text)
        self.assertEqual(result.kind, "plan_approval")

    # ---------------------------------------------------------
    # GROUP 2: Tool Schema Registration (register_plan_tools)
    # ---------------------------------------------------------

    def test_registry_binding(self):
        mock_registry = MagicMock()
        register_plan_tools(mock_registry, self.ctx)
        mock_registry.register.assert_called_once()

    def test_json_schema_structure(self):
        mock_registry = MagicMock()
        register_plan_tools(mock_registry, self.ctx)

        call_kwargs = mock_registry.register.call_args[1]

        self.assertEqual(call_kwargs["name"], "SubmitPlan")

        input_schema = call_kwargs["input_schema"]
        self.assertEqual(input_schema["type"], "object")
        self.assertEqual(input_schema["required"], ["plan_summary"])
        self.assertIn("plan_summary", input_schema["properties"])
        self.assertEqual(input_schema["properties"]["plan_summary"]["type"], "string")

    def test_registered_as_readonly(self):
        # SubmitPlan MUST be read-only, otherwise it disappears in PLAN mode
        # (clone_readonly filters the registry down to read-only tools).
        mock_registry = MagicMock()
        register_plan_tools(mock_registry, self.ctx)

        call_kwargs = mock_registry.register.call_args[1]
        self.assertTrue(call_kwargs["is_readonly"])


if __name__ == "__main__":
    unittest.main()
