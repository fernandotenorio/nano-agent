import io
import unittest

from rich.console import Console

from ui.base import UsageReport, UsageRow, UsageSection
from ui.rich_ui import USAGE_EMPTY_MESSAGE, RichUI
from usagetracker import (
    MAIN_AGENT,
    TOOL_SPLIT_NOTE,
    SessionUsageTracker,
    build_report,
    subagent_name,
)


def usage(prompt: int = 0, completion: int = 0, **extra) -> dict:
    return {"prompt_tokens": prompt, "completion_tokens": completion, **extra}


def titles(report: UsageReport) -> list[str]:
    return [section.title for section in report.sections]


def section(report: UsageReport, title: str) -> UsageSection:
    return next(item for item in report.sections if item.title == title)


def labels(report: UsageReport, title: str) -> list[str]:
    return [row.label for row in section(report, title).rows]


class TestEmptyReport(unittest.TestCase):
    """
    Test Suite for a session that has not spent anything yet
    (usagetracker.build_report).
    """

    def setUp(self):
        self.report = build_report(SessionUsageTracker())

    def test_report_is_empty(self):
        self.assertTrue(self.report.is_empty)

    def test_no_sections_are_offered(self):
        self.assertEqual(self.report.sections, ())

    def test_totals_are_zero(self):
        self.assertEqual(self.report.totals.total_tokens, 0)
        self.assertEqual(self.report.totals.calls, 0)


class TestReportSections(unittest.TestCase):
    """
    Test Suite for the tables the usage view renders
    (usagetracker.build_report).
    """

    def setUp(self):
        self.tracker = SessionUsageTracker()
        self.tracker.record_turn(MAIN_AGENT, "ollama/gemma3:12b", (), usage(prompt=100, completion=10))
        self.tracker.record_turn(
            MAIN_AGENT, "ollama/gemma3:12b", ("Read", "Grep"), usage(prompt=200, completion=20)
        )
        self.tracker.record_turn(
            subagent_name("code-reviewer"), "ollama/gemma3:12b", ("ls",), usage(prompt=40, completion=4)
        )
        self.report = build_report(self.tracker)

    def test_report_is_not_empty(self):
        self.assertFalse(self.report.is_empty)

    def test_totals_cover_the_whole_session(self):
        self.assertEqual(self.report.totals.total_tokens, 374)
        self.assertEqual(self.report.totals.calls, 3)
        self.assertEqual(self.report.totals.label, "Total")

    def test_agent_and_model_sections_are_always_present(self):
        self.assertIn("By agent", titles(self.report))
        self.assertIn("By model", titles(self.report))

    def test_agent_section_lists_every_agent(self):
        self.assertEqual(sorted(labels(self.report, "By agent")), ["main", "subagent:code-reviewer"])

    def test_tool_section_lists_every_tool(self):
        self.assertEqual(sorted(labels(self.report, "By tool")), ["Grep", "Read", "ls"])

    def test_tool_section_explains_the_split(self):
        self.assertEqual(section(self.report, "By tool").note, TOOL_SPLIT_NOTE)

    def test_other_sections_carry_no_note(self):
        self.assertEqual(section(self.report, "By agent").note, "")

    def test_single_provider_section_is_omitted(self):
        """It would repeat the model table a row at a time."""
        self.assertNotIn("By provider", titles(self.report))

    def test_rows_are_ordered_by_cost(self):
        rows = section(self.report, "By agent").rows

        self.assertEqual([row.label for row in rows], ["main", "subagent:code-reviewer"])
        self.assertGreater(rows[0].total_tokens, rows[1].total_tokens)

    def test_section_rows_sum_to_the_reported_total(self):
        rows = section(self.report, "By agent").rows

        self.assertEqual(
            sum(row.total_tokens for row in rows), self.report.totals.total_tokens
        )


class TestProviderSection(unittest.TestCase):
    """
    Test Suite for the conditional provider table (usagetracker.build_report).
    """

    def test_multiple_providers_earn_a_section(self):
        tracker = SessionUsageTracker()
        tracker.record_turn(MAIN_AGENT, "ollama/gemma3:12b", (), usage(prompt=10, completion=1))
        tracker.record_turn(MAIN_AGENT, "anthropic/claude", (), usage(prompt=20, completion=2))

        report = build_report(tracker)

        self.assertIn("By provider", titles(report))
        self.assertEqual(sorted(labels(report, "By provider")), ["anthropic", "ollama"])

    def test_provider_rows_are_ordered_by_cost(self):
        tracker = SessionUsageTracker()
        tracker.record_turn(MAIN_AGENT, "ollama/gemma3:12b", (), usage(prompt=10, completion=1))
        tracker.record_turn(MAIN_AGENT, "anthropic/claude", (), usage(prompt=20, completion=2))

        report = build_report(tracker)

        self.assertEqual(labels(report, "By provider")[0], "anthropic")


class TestToolSectionOmission(unittest.TestCase):
    """
    Test Suite for a session that only ever talked
    (usagetracker.build_report).
    """

    def setUp(self):
        tracker = SessionUsageTracker()
        tracker.record_turn(MAIN_AGENT, "ollama/gemma3:12b", (), usage(prompt=10, completion=1))
        self.report = build_report(tracker)

    def test_tool_section_is_omitted_when_no_tool_ran(self):
        self.assertNotIn("By tool", titles(self.report))

    def test_the_rest_of_the_report_still_stands(self):
        self.assertEqual(titles(self.report), ["By agent", "By model"])
        self.assertEqual(self.report.totals.total_tokens, 11)


class TestUsageRow(unittest.TestCase):
    """Test Suite for the UI-facing row type (ui.base.UsageRow)."""

    def test_total_excludes_cached(self):
        row = UsageRow(label="main", input_tokens=100, output_tokens=20, cached_tokens=80)

        self.assertEqual(row.total_tokens, 120)

    def test_a_default_report_is_empty(self):
        self.assertTrue(UsageReport().is_empty)


class TestRichUsageRendering(unittest.IsolatedAsyncioTestCase):
    """
    Test Suite for the scrolling front-end's usage table (ui/rich_ui.py).

    The Rich UI has no status bar and no modal, so this printed table is the
    only place a plain-terminal session ever sees its token spend.
    """

    def setUp(self):
        self.output = io.StringIO()
        # A fixed width keeps the table from wrapping mid-number, which would
        # break the assertions for reasons that have nothing to do with usage.
        self.ui = RichUI(Console(file=self.output, width=120, force_terminal=False))

    @property
    def rendered(self) -> str:
        return self.output.getvalue()

    async def test_every_section_is_printed(self):
        tracker = SessionUsageTracker()
        tracker.record_turn(MAIN_AGENT, "ollama/gemma3:12b", ("Read",), usage(prompt=100, completion=10))

        await self.ui.show_usage(build_report(tracker))

        self.assertIn("By agent", self.rendered)
        self.assertIn("By model", self.rendered)
        self.assertIn("By tool", self.rendered)

    async def test_rows_and_totals_are_printed(self):
        tracker = SessionUsageTracker()
        tracker.record_turn(MAIN_AGENT, "ollama/gemma3:12b", ("Read",), usage(prompt=1000, completion=100))

        await self.ui.show_usage(build_report(tracker))

        self.assertIn("main", self.rendered)
        self.assertIn("Read", self.rendered)
        self.assertIn("1,100 tokens", self.rendered)

    async def test_the_split_caveat_is_printed(self):
        tracker = SessionUsageTracker()
        tracker.record_turn(MAIN_AGENT, "ollama/x", ("Read", "Grep"), usage(prompt=100, completion=10))

        await self.ui.show_usage(build_report(tracker))

        self.assertIn("split evenly", self.rendered)

    async def test_an_empty_session_says_so(self):
        await self.ui.show_usage(build_report(SessionUsageTracker()))

        self.assertIn(USAGE_EMPTY_MESSAGE, self.rendered)

    async def test_a_running_spinner_is_paused_and_restored(self):
        """Rich allows one live display at a time; a table over one corrupts both."""
        async with self.ui.tool_status("thinking"):
            status = self.ui._active_status

            await self.ui.show_usage(build_report(SessionUsageTracker()))

            self.assertIs(self.ui._active_status, status)


if __name__ == "__main__":
    unittest.main()
