import unittest

from ui.base import PlanDecision, ShellDecision, ToolCallView, UsageInfo
from ui.null_ui import NullUI, QuietUI


class ParentUI(NullUI):
    """Stands in for a real front-end, keeping whatever reaches it."""

    def __init__(self):
        self.usage_updates: list[UsageInfo] = []
        self.commands: list[str] = []
        self.plans: list[str] = []
        self.rendered: list[str] = []

    async def usage(self, info: UsageInfo) -> None:
        self.usage_updates.append(info)

    async def assistant_text(self, text: str) -> None:
        self.rendered.append(text)

    async def tool_result(self, call: ToolCallView) -> None:
        self.rendered.append(call.summary)

    async def notice(self, text: str) -> None:
        self.rendered.append(text)

    async def confirm_shell(self, command: str, description: str | None = None) -> ShellDecision:
        self.commands.append(command)
        return ShellDecision(approved=True)

    async def approve_plan(self, plan_summary: str) -> PlanDecision:
        self.plans.append(plan_summary)
        return PlanDecision(choice="build")


class TestQuietUI(unittest.IsolatedAsyncioTestCase):
    """
    Test Suite for the sub-agent front-end (ui/null_ui.py).

    A sub-agent runs behind a single spinner, so its own rendering is thrown
    away on purpose. Two things must still get through to the parent: the
    safety prompts, and the tokens it spends.
    """

    def setUp(self):
        self.parent = ParentUI()
        self.quiet = QuietUI(self.parent)

    async def test_rendering_is_thrown_away(self):
        await self.quiet.assistant_text("an internal thought")
        await self.quiet.tool_result(ToolCallView(name="Read", args={}, summary="Read a.py"))
        await self.quiet.notice("a notice")

        self.assertEqual(self.parent.rendered, [])

    async def test_tokens_reach_the_parent(self):
        """The running total covers the session, not just the visible agent."""
        await self.quiet.usage(UsageInfo(input_tokens=200, output_tokens=20))

        self.assertEqual(len(self.parent.usage_updates), 1)
        self.assertEqual(self.parent.usage_updates[0].total_tokens, 220)

    async def test_shell_confirmation_reaches_the_parent(self):
        decision = await self.quiet.confirm_shell("pytest -x", "Run the tests")

        self.assertEqual(self.parent.commands, ["pytest -x"])
        self.assertTrue(decision.approved)

    async def test_plan_approval_reaches_the_parent(self):
        decision = await self.quiet.approve_plan("1. Do the thing")

        self.assertEqual(self.parent.plans, ["1. Do the thing"])
        self.assertEqual(decision.choice, "build")


class TestNullUI(unittest.IsolatedAsyncioTestCase):
    """
    Test Suite for the silent front-end (ui/null_ui.py).

    Without a user to ask, every interactive answer has to be the one that
    cannot do harm.
    """

    def setUp(self):
        self.ui = NullUI()

    async def test_usage_is_silently_dropped(self):
        """Nothing to report to, and nowhere to report it: this must not raise."""
        await self.ui.usage(UsageInfo(input_tokens=10, output_tokens=1))

    async def test_shell_confirmation_fails_closed(self):
        self.assertFalse((await self.ui.confirm_shell("rm -rf /")).approved)

    async def test_plan_approval_fails_closed(self):
        self.assertEqual((await self.ui.approve_plan("a plan")).choice, "reject")


if __name__ == "__main__":
    unittest.main()
