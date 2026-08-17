import unittest

from usagetracker import (
    MAIN_AGENT,
    SessionUsageTracker,
    UsageRecord,
    UsageTotals,
    _split,
    subagent_name,
)


def usage(prompt: int = 0, completion: int = 0, **extra) -> dict:
    """Builds a LiteLLM-shaped usage dict."""
    return {"prompt_tokens": prompt, "completion_tokens": completion, **extra}


class TestUsageParsing(unittest.TestCase):
    """
    Test Suite for reading a provider's usage dict (usagetracker).

    Providers disagree about which fields exist and what they are called, and
    a usage figure is never worth crashing a session over, so every shape that
    has been seen in the wild has to land somewhere sane.
    """

    def setUp(self):
        self.tracker = SessionUsageTracker()

    def record(self, usage_dict) -> None:
        self.tracker.record_turn(MAIN_AGENT, "test/model", (), usage_dict)

    def test_plain_prompt_and_completion_tokens(self):
        self.record(usage(prompt=100, completion=25))

        totals = self.tracker.total()
        self.assertEqual(totals.input_tokens, 100)
        self.assertEqual(totals.output_tokens, 25)
        self.assertEqual(totals.total_tokens, 125)
        self.assertEqual(totals.calls, 1)

    def test_missing_usage_records_nothing(self):
        self.record(None)

        self.assertEqual(self.tracker.records, [])

    def test_non_mapping_usage_records_nothing(self):
        self.record("120 tokens")

        self.assertEqual(self.tracker.records, [])

    def test_all_zero_usage_records_nothing(self):
        """An empty figure is noise: it would add a call to every view."""
        self.record(usage(prompt=0, completion=0))

        self.assertEqual(self.tracker.records, [])

    def test_openai_style_cached_tokens(self):
        self.record(usage(
            prompt=100,
            completion=10,
            prompt_tokens_details={"cached_tokens": 60},
        ))

        self.assertEqual(self.tracker.total().cached_tokens, 60)

    def test_anthropic_style_cached_tokens(self):
        self.record(usage(prompt=100, completion=10, cache_read_input_tokens=40))

        self.assertEqual(self.tracker.total().cached_tokens, 40)

    def test_null_prompt_tokens_details_does_not_crash(self):
        """LiteLLM sends the key with a null value often enough to matter."""
        self.record(usage(prompt=100, completion=10, prompt_tokens_details=None))

        self.assertEqual(self.tracker.total().cached_tokens, 0)
        self.assertEqual(self.tracker.total().input_tokens, 100)

    def test_details_without_cached_tokens_falls_back(self):
        self.record(usage(
            prompt=100,
            completion=10,
            prompt_tokens_details={"audio_tokens": 5},
            cache_read_input_tokens=7,
        ))

        self.assertEqual(self.tracker.total().cached_tokens, 7)

    def test_negative_and_null_counts_read_as_zero(self):
        self.record(usage(prompt=-5, completion=None))

        self.assertEqual(self.tracker.records, [])

    def test_float_counts_are_accepted(self):
        self.record(usage(prompt=100.0, completion=25.0))

        self.assertEqual(self.tracker.total().total_tokens, 125)

    def test_booleans_are_not_counted_as_numbers(self):
        self.record(usage(prompt=True, completion=True))

        self.assertEqual(self.tracker.records, [])

    def test_blank_agent_and_model_become_unknown(self):
        self.tracker.record_turn("", "", (), usage(prompt=10, completion=1))

        record = self.tracker.records[0]
        self.assertEqual(record.agent, "unknown")
        self.assertEqual(record.model, "unknown")


class TestUsageRecord(unittest.TestCase):
    """
    Test Suite for a single ledger entry (usagetracker.UsageRecord).
    The activity keys are derived from the record, never stored.
    """

    def test_text_turn_has_one_text_activity(self):
        record = UsageRecord(agent=MAIN_AGENT, model="m")

        self.assertEqual(record.activities, ("text",))

    def test_tool_turn_names_every_tool(self):
        record = UsageRecord(agent=MAIN_AGENT, model="m", tools=("Read", "Grep"))

        self.assertEqual(record.activities, ("tool:Read", "tool:Grep"))

    def test_internal_turn_is_marked_as_internal(self):
        record = UsageRecord(agent=MAIN_AGENT, model="m", tools=("WebFetch",), internal=True)

        self.assertEqual(record.activities, ("tool_internal:WebFetch",))

    def test_internal_turn_without_a_tool_still_has_an_activity(self):
        record = UsageRecord(agent=MAIN_AGENT, model="m", internal=True)

        self.assertEqual(record.activities, ("tool_internal:unknown",))

    def test_total_tokens_excludes_cached(self):
        """Cached tokens are a subset of the input, not an extra charge."""
        record = UsageRecord(
            agent=MAIN_AGENT, model="m", input_tokens=100, output_tokens=20, cached_tokens=90
        )

        self.assertEqual(record.total_tokens, 120)


class TestEvenSplit(unittest.TestCase):
    """
    Test Suite for dividing one turn between the tools it called
    (usagetracker._split). The shares must always add back up.
    """

    def test_exact_division(self):
        self.assertEqual(_split(100, 4), [25, 25, 25, 25])

    def test_remainder_goes_to_the_first_share(self):
        self.assertEqual(_split(10, 3), [4, 3, 3])

    def test_single_part_keeps_everything(self):
        self.assertEqual(_split(7, 1), [7])

    def test_zero_parts_yields_nothing(self):
        self.assertEqual(_split(7, 0), [])

    def test_more_parts_than_tokens(self):
        self.assertEqual(_split(2, 5), [2, 0, 0, 0, 0])

    def test_shares_always_sum_to_the_original(self):
        for value in (0, 1, 7, 99, 1000, 12345):
            for parts in range(1, 8):
                self.assertEqual(sum(_split(value, parts)), value, (value, parts))


class TestUsageTotals(unittest.TestCase):
    """Test Suite for the running sums (usagetracker.UsageTotals)."""

    def test_add_accumulates_and_counts_a_call(self):
        totals = UsageTotals()
        totals.add(10, 2, 1)
        totals.add(5, 1, 0)

        self.assertEqual(totals.input_tokens, 15)
        self.assertEqual(totals.output_tokens, 3)
        self.assertEqual(totals.cached_tokens, 1)
        self.assertEqual(totals.calls, 2)

    def test_as_row_carries_every_figure(self):
        row = UsageTotals(input_tokens=10, output_tokens=2, cached_tokens=1, calls=3).as_row("main")

        self.assertEqual(row.label, "main")
        self.assertEqual(row.input_tokens, 10)
        self.assertEqual(row.output_tokens, 2)
        self.assertEqual(row.cached_tokens, 1)
        self.assertEqual(row.calls, 3)
        self.assertEqual(row.total_tokens, 12)


class TestUsageViews(unittest.TestCase):
    """
    Test Suite for the aggregations over the ledger (usagetracker).

    The property that matters throughout: a response is one entry however many
    tools it called, so the session total never inflates.
    """

    def setUp(self):
        self.tracker = SessionUsageTracker()

    def test_multi_tool_turn_is_counted_once(self):
        self.tracker.record_turn(
            MAIN_AGENT, "anthropic/claude", ("Read", "Grep", "ls"), usage(prompt=300, completion=30)
        )

        totals = self.tracker.total()
        self.assertEqual(totals.total_tokens, 330)
        self.assertEqual(totals.calls, 1)

    def test_by_agent_separates_the_main_loop_from_subagents(self):
        self.tracker.record_turn(MAIN_AGENT, "m", (), usage(prompt=100, completion=10))
        self.tracker.record_turn(
            subagent_name("code-reviewer"), "m", (), usage(prompt=50, completion=5)
        )

        by_agent = self.tracker.by_agent()
        self.assertEqual(by_agent[MAIN_AGENT].total_tokens, 110)
        self.assertEqual(by_agent["subagent:code-reviewer"].total_tokens, 55)

    def test_by_model_groups_on_the_requested_string(self):
        self.tracker.record_turn(MAIN_AGENT, "ollama/gemma3:12b", (), usage(prompt=10, completion=1))
        self.tracker.record_turn(MAIN_AGENT, "ollama/gemma3:12b", (), usage(prompt=20, completion=2))
        self.tracker.record_turn(MAIN_AGENT, "anthropic/claude", (), usage(prompt=5, completion=1))

        by_model = self.tracker.by_model()
        self.assertEqual(by_model["ollama/gemma3:12b"].total_tokens, 33)
        self.assertEqual(by_model["ollama/gemma3:12b"].calls, 2)
        self.assertEqual(by_model["anthropic/claude"].total_tokens, 6)

    def test_by_provider_reads_the_prefix(self):
        self.tracker.record_turn(MAIN_AGENT, "ollama/gemma3:12b", (), usage(prompt=10, completion=1))
        self.tracker.record_turn(MAIN_AGENT, "anthropic/claude", (), usage(prompt=20, completion=2))

        by_provider = self.tracker.by_provider()
        self.assertEqual(by_provider["ollama"].total_tokens, 11)
        self.assertEqual(by_provider["anthropic"].total_tokens, 22)

    def test_by_provider_infers_from_a_bare_model_family(self):
        self.tracker.record_turn(MAIN_AGENT, "gpt-4o", (), usage(prompt=10, completion=1))

        self.assertIn("openai", self.tracker.by_provider())

    def test_by_provider_falls_back_to_unknown(self):
        self.tracker.record_turn(MAIN_AGENT, "some-local-thing", (), usage(prompt=10, completion=1))

        self.assertIn("unknown", self.tracker.by_provider())

    def test_by_tool_splits_a_multi_tool_turn_evenly(self):
        self.tracker.record_turn(MAIN_AGENT, "m", ("Read", "Grep"), usage(prompt=100, completion=10))

        by_tool = self.tracker.by_tool()
        self.assertEqual(by_tool["Read"].input_tokens, 50)
        self.assertEqual(by_tool["Grep"].input_tokens, 50)
        self.assertEqual(by_tool["Read"].output_tokens, 5)
        self.assertEqual(by_tool["Grep"].output_tokens, 5)

    def test_by_tool_gives_an_odd_remainder_to_the_first_tool(self):
        self.tracker.record_turn(MAIN_AGENT, "m", ("Read", "Grep"), usage(prompt=101, completion=0))

        by_tool = self.tracker.by_tool()
        self.assertEqual(by_tool["Read"].input_tokens, 51)
        self.assertEqual(by_tool["Grep"].input_tokens, 50)

    def test_by_tool_counts_invocations_not_model_calls(self):
        """One response calling two tools is one call but two invocations."""
        self.tracker.record_turn(MAIN_AGENT, "m", ("Read", "Grep"), usage(prompt=100, completion=10))

        by_tool = self.tracker.by_tool()
        self.assertEqual(by_tool["Read"].calls, 1)
        self.assertEqual(by_tool["Grep"].calls, 1)
        self.assertEqual(self.tracker.total().calls, 1)

    def test_by_tool_ignores_text_only_turns(self):
        self.tracker.record_turn(MAIN_AGENT, "m", (), usage(prompt=100, completion=10))

        self.assertEqual(self.tracker.by_tool(), {})

    def test_by_tool_includes_llms_run_inside_a_tool(self):
        self.tracker.record_tool_internal(
            MAIN_AGENT, "openai/gpt-4o-mini", "WebFetch", usage(prompt=80, completion=8)
        )

        self.assertEqual(self.tracker.by_tool()["WebFetch"].total_tokens, 88)

    def test_by_activity_uses_the_three_part_key(self):
        self.tracker.record_turn(MAIN_AGENT, "ollama/x", (), usage(prompt=10, completion=1))
        self.tracker.record_turn(MAIN_AGENT, "ollama/x", ("Read",), usage(prompt=20, completion=2))
        self.tracker.record_tool_internal(MAIN_AGENT, "ollama/y", "WebFetch", usage(prompt=30, completion=3))

        by_activity = self.tracker.by_activity()
        self.assertEqual(by_activity[(MAIN_AGENT, "text", "ollama/x")].total_tokens, 11)
        self.assertEqual(by_activity[(MAIN_AGENT, "tool:Read", "ollama/x")].total_tokens, 22)
        self.assertEqual(
            by_activity[(MAIN_AGENT, "tool_internal:WebFetch", "ollama/y")].total_tokens, 33
        )

    def test_by_activity_splits_a_multi_tool_turn(self):
        self.tracker.record_turn(MAIN_AGENT, "m", ("Read", "Grep"), usage(prompt=100, completion=10))

        by_activity = self.tracker.by_activity()
        self.assertEqual(by_activity[(MAIN_AGENT, "tool:Read", "m")].total_tokens, 55)
        self.assertEqual(by_activity[(MAIN_AGENT, "tool:Grep", "m")].total_tokens, 55)

    def test_repeated_tool_in_one_turn_accumulates(self):
        """Two parallel Reads in a single response are two invocations of Read."""
        self.tracker.record_turn(MAIN_AGENT, "m", ("Read", "Read"), usage(prompt=100, completion=10))

        by_tool = self.tracker.by_tool()
        self.assertEqual(by_tool["Read"].total_tokens, 110)
        self.assertEqual(by_tool["Read"].calls, 2)

    def test_empty_tracker_has_empty_views(self):
        self.assertEqual(self.tracker.total().total_tokens, 0)
        self.assertEqual(self.tracker.total_tokens(), 0)
        self.assertEqual(self.tracker.by_agent(), {})
        self.assertEqual(self.tracker.by_model(), {})
        self.assertEqual(self.tracker.by_tool(), {})
        self.assertEqual(self.tracker.by_activity(), {})


class TestViewInvariants(unittest.TestCase):
    """
    Test Suite for the arithmetic the usage table depends on.

    Splitting a turn between its tools is only defensible while the shares add
    back up; if they did not, the columns on screen would contradict the total
    printed above them.
    """

    def setUp(self):
        self.tracker = SessionUsageTracker()
        self.tracker.record_turn(MAIN_AGENT, "ollama/a", (), usage(prompt=101, completion=13))
        self.tracker.record_turn(MAIN_AGENT, "ollama/a", ("Read",), usage(prompt=7, completion=3))
        self.tracker.record_turn(
            MAIN_AGENT, "anthropic/b", ("Read", "Grep", "ls"), usage(prompt=100, completion=11)
        )
        self.tracker.record_turn(
            subagent_name("code-reviewer"), "ollama/a", ("Glob", "Grep"),
            usage(prompt=55, completion=5, cache_read_input_tokens=9),
        )
        self.tracker.record_tool_internal(
            MAIN_AGENT, "openai/gpt-4o-mini", "WebFetch", usage(prompt=31, completion=2)
        )

    def total_of(self, buckets) -> int:
        return sum(totals.total_tokens for totals in buckets.values())

    def test_by_agent_sums_to_the_session_total(self):
        self.assertEqual(self.total_of(self.tracker.by_agent()), self.tracker.total_tokens())

    def test_by_model_sums_to_the_session_total(self):
        self.assertEqual(self.total_of(self.tracker.by_model()), self.tracker.total_tokens())

    def test_by_provider_sums_to_the_session_total(self):
        self.assertEqual(self.total_of(self.tracker.by_provider()), self.tracker.total_tokens())

    def test_by_activity_sums_to_the_session_total(self):
        """Text turns are an activity too, so nothing falls outside this view."""
        self.assertEqual(self.total_of(self.tracker.by_activity()), self.tracker.total_tokens())

    def test_by_tool_sums_to_the_tool_calling_part_of_the_session(self):
        """Text-only turns belong to no tool, so this view is a strict subset."""
        expected = sum(
            record.total_tokens for record in self.tracker.records if record.tools
        )

        self.assertEqual(self.total_of(self.tracker.by_tool()), expected)
        self.assertLess(self.total_of(self.tracker.by_tool()), self.tracker.total_tokens())

    def test_cached_tokens_survive_a_split(self):
        by_tool = self.tracker.by_tool()

        self.assertEqual(by_tool["Glob"].cached_tokens + by_tool["Grep"].cached_tokens, 9)

    def test_call_counts_match_the_number_of_recorded_calls(self):
        self.assertEqual(self.tracker.total().calls, len(self.tracker.records))


class TestSubagentNaming(unittest.TestCase):
    """Test Suite for the agent-name convention (usagetracker.subagent_name)."""

    def test_subagent_names_are_prefixed(self):
        self.assertEqual(subagent_name("code-reviewer"), "subagent:code-reviewer")

    def test_main_agent_is_not_prefixed(self):
        self.assertEqual(MAIN_AGENT, "main")


if __name__ == "__main__":
    unittest.main()
