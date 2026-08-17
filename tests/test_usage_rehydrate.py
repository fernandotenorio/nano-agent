import tempfile
import unittest
from pathlib import Path

from transcript import Transcript
from typedefs import (
    AssistantMessage,
    SystemMessage,
    TextMessageContent,
    ToolResultMessageContent,
    ToolUseMessageContent,
    UserMessage,
)
from usagetracker import (
    MAIN_AGENT,
    SessionUsageTracker,
    rehydrate_session_usage,
    subagent_type_from_path,
)


def usage(prompt: int = 0, completion: int = 0, **extra) -> dict:
    return {"prompt_tokens": prompt, "completion_tokens": completion, **extra}


def text_turn(model: str, request_model: str | None, tokens: dict) -> AssistantMessage:
    return AssistantMessage(
        content=[TextMessageContent(text="All done.")],
        model=model,
        request_model=request_model,
        usage=tokens,
    )


def tool_turn(model: str, request_model: str | None, tokens: dict, *names: str) -> AssistantMessage:
    return AssistantMessage(
        content=[
            ToolUseMessageContent(id=f"call_{index}", name=name, input={})
            for index, name in enumerate(names)
        ],
        model=model,
        request_model=request_model,
        usage=tokens,
    )


class RehydrateTestCase(unittest.TestCase):
    """Shared temporary transcript directory."""

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.test_dir.name)
        self.main_path = self.base_path / "session.jsonl"

    def tearDown(self):
        self.test_dir.cleanup()

    def write(self, path: Path, messages: list) -> None:
        transcript = Transcript(path)
        for message in messages:
            transcript.append(message)


class TestRehydrateMainTranscript(RehydrateTestCase):
    """
    Test Suite for rebuilding a session's ledger from disk
    (usagetracker.rehydrate_session_usage).

    Resuming has to reproduce what a live session recorded; if it did not, the
    totals would silently reset every time the user picked a conversation back
    up.
    """

    def test_missing_transcript_yields_an_empty_tracker(self):
        tracker = rehydrate_session_usage(self.base_path / "nothing.jsonl")

        self.assertEqual(tracker.records, [])

    def test_a_rehydrated_session_matches_a_live_one(self):
        messages = [
            SystemMessage(content="You are a coding assistant."),
            UserMessage(content=[TextMessageContent(text="read main.py")]),
            tool_turn("gemma3:12b", "ollama/gemma3:12b", usage(prompt=200, completion=20), "Read", "Grep"),
            UserMessage(content=[
                ToolResultMessageContent(tool_use_id="call_0", content="ok", tool_name="Read"),
                ToolResultMessageContent(tool_use_id="call_1", content="ok", tool_name="Grep"),
            ]),
            text_turn("gemma3:12b", "ollama/gemma3:12b", usage(prompt=300, completion=30)),
        ]
        self.write(self.main_path, messages)

        live = SessionUsageTracker()
        live.record_turn(MAIN_AGENT, "ollama/gemma3:12b", ("Read", "Grep"), usage(prompt=200, completion=20))
        live.record_turn(MAIN_AGENT, "ollama/gemma3:12b", (), usage(prompt=300, completion=30))

        resumed = rehydrate_session_usage(self.main_path)

        self.assertEqual(resumed.records, live.records)

    def test_responses_without_usage_are_skipped(self):
        self.write(self.main_path, [
            text_turn("gemma3:12b", "ollama/gemma3:12b", None),
            text_turn("gemma3:12b", "ollama/gemma3:12b", usage(prompt=10, completion=1)),
        ])

        tracker = rehydrate_session_usage(self.main_path)

        self.assertEqual(len(tracker.records), 1)

    def test_tool_names_survive_the_round_trip(self):
        self.write(self.main_path, [
            tool_turn("m", "ollama/m", usage(prompt=100, completion=10), "Read", "Grep"),
        ])

        tracker = rehydrate_session_usage(self.main_path)

        self.assertEqual(tracker.records[0].tools, ("Read", "Grep"))
        self.assertEqual(tracker.by_tool()["Read"].input_tokens, 50)

    def test_an_llm_run_inside_a_tool_is_recovered(self):
        self.write(self.main_path, [
            UserMessage(content=[
                ToolResultMessageContent(
                    tool_use_id="call_0",
                    content="summary",
                    tool_name="WebFetch",
                    usage=usage(prompt=80, completion=8),
                    internal_model="openai/gpt-4o-mini",
                )
            ]),
        ])

        tracker = rehydrate_session_usage(self.main_path)

        record = tracker.records[0]
        self.assertTrue(record.internal)
        self.assertEqual(record.tools, ("WebFetch",))
        self.assertEqual(record.model, "openai/gpt-4o-mini")
        self.assertEqual(record.total_tokens, 88)

    def test_tool_results_without_usage_are_ignored(self):
        self.write(self.main_path, [
            UserMessage(content=[
                ToolResultMessageContent(tool_use_id="call_0", content="ok", tool_name="Read")
            ]),
        ])

        self.assertEqual(rehydrate_session_usage(self.main_path).records, [])

    def test_plain_text_user_messages_are_ignored(self):
        self.write(self.main_path, [UserMessage(content="just a string")])

        self.assertEqual(rehydrate_session_usage(self.main_path).records, [])


class TestRehydrateModelIdentity(RehydrateTestCase):
    """
    Test Suite for which model string a rehydrated record is keyed on
    (usagetracker.rehydrate_session_usage).
    """

    def test_the_requested_model_wins_over_the_echoed_one(self):
        """Otherwise one model splits into two rows across a resume."""
        self.write(self.main_path, [
            text_turn("gemma3:12b", "ollama/gemma3:12b", usage(prompt=10, completion=1)),
        ])

        tracker = rehydrate_session_usage(self.main_path)

        self.assertEqual(tracker.records[0].model, "ollama/gemma3:12b")
        self.assertIn("ollama", tracker.by_provider())

    def test_an_older_transcript_falls_back_to_the_echoed_model(self):
        """Transcripts written before request_model existed still have to load."""
        self.main_path.write_text(
            '{"role": "assistant", "id": "1", "type": "message", '
            '"content": [{"type": "text", "text": "hi"}], "model": "gemma3:12b", '
            '"usage": {"prompt_tokens": 10, "completion_tokens": 1}}\n',
            encoding="utf-8",
        )

        tracker = rehydrate_session_usage(self.main_path)

        self.assertEqual(tracker.records[0].model, "gemma3:12b")

    def test_a_response_with_no_model_at_all_becomes_unknown(self):
        self.write(self.main_path, [
            AssistantMessage(
                content=[TextMessageContent(text="hi")],
                usage=usage(prompt=10, completion=1),
            ),
        ])

        self.assertEqual(rehydrate_session_usage(self.main_path).records[0].model, "unknown")

    def test_an_older_tool_result_without_the_new_fields_still_loads(self):
        self.main_path.write_text(
            '{"role": "user", "content": [{"type": "tool_result", '
            '"tool_use_id": "call_0", "content": "ok"}]}\n',
            encoding="utf-8",
        )

        tracker = rehydrate_session_usage(self.main_path)

        self.assertEqual(tracker.records, [])


class TestRehydrateSubagents(RehydrateTestCase):
    """
    Test Suite for picking up sub-agent transcripts
    (usagetracker.rehydrate_session_usage).

    Sub-agents write their own sibling files, so reading only the main one
    would quietly drop everything they spent.
    """

    def subagent_path(self, subagent_type: str, run_id: str = "123456") -> Path:
        return self.base_path / f"session_{subagent_type}_{run_id}.jsonl"

    def test_sibling_transcripts_are_absorbed(self):
        self.write(self.main_path, [
            tool_turn("m", "ollama/m", usage(prompt=100, completion=10), "Task"),
        ])
        self.write(self.subagent_path("code-reviewer"), [
            text_turn("m", "ollama/m", usage(prompt=50, completion=5)),
        ])

        tracker = rehydrate_session_usage(self.main_path)

        by_agent = tracker.by_agent()
        self.assertEqual(by_agent[MAIN_AGENT].total_tokens, 110)
        self.assertEqual(by_agent["subagent:code-reviewer"].total_tokens, 55)

    def test_several_sub_agents_are_kept_apart(self):
        self.write(self.main_path, [])
        self.write(self.subagent_path("code-reviewer", "aaa111"), [
            text_turn("m", "ollama/m", usage(prompt=10, completion=1)),
        ])
        self.write(self.subagent_path("explore", "bbb222"), [
            text_turn("m", "ollama/m", usage(prompt=20, completion=2)),
        ])

        by_agent = rehydrate_session_usage(self.main_path).by_agent()

        self.assertEqual(by_agent["subagent:code-reviewer"].total_tokens, 11)
        self.assertEqual(by_agent["subagent:explore"].total_tokens, 22)

    def test_two_runs_of_one_sub_agent_type_share_a_row(self):
        self.write(self.main_path, [])
        self.write(self.subagent_path("explore", "aaa111"), [
            text_turn("m", "ollama/m", usage(prompt=10, completion=1)),
        ])
        self.write(self.subagent_path("explore", "bbb222"), [
            text_turn("m", "ollama/m", usage(prompt=20, completion=2)),
        ])

        by_agent = rehydrate_session_usage(self.main_path).by_agent()

        self.assertEqual(by_agent["subagent:explore"].total_tokens, 33)
        self.assertEqual(by_agent["subagent:explore"].calls, 2)

    def test_an_unrelated_transcript_is_not_swept_up(self):
        self.write(self.main_path, [])
        self.write(self.base_path / "other.jsonl", [
            text_turn("m", "ollama/m", usage(prompt=999, completion=99)),
        ])

        self.assertEqual(rehydrate_session_usage(self.main_path).records, [])

    def test_a_sub_agents_own_tool_usage_lands_under_the_sub_agent(self):
        self.write(self.main_path, [])
        self.write(self.subagent_path("explore"), [
            UserMessage(content=[
                ToolResultMessageContent(
                    tool_use_id="call_0",
                    content="ok",
                    tool_name="WebFetch",
                    usage=usage(prompt=80, completion=8),
                    internal_model="openai/gpt-4o-mini",
                )
            ]),
        ])

        tracker = rehydrate_session_usage(self.main_path)

        self.assertEqual(tracker.records[0].agent, "subagent:explore")


class TestSubagentTypeFromPath(unittest.TestCase):
    """
    Test Suite for recovering a sub-agent's type from its filename
    (usagetracker.subagent_type_from_path).
    """

    def setUp(self):
        self.main_path = Path("/t/session.jsonl")

    def test_a_standard_name_is_parsed(self):
        sibling = Path("/t/session_code-reviewer_123456.jsonl")

        self.assertEqual(subagent_type_from_path(self.main_path, sibling), "code-reviewer")

    def test_a_type_containing_underscores_survives(self):
        """Only the trailing run id is dropped, not everything after the first _."""
        sibling = Path("/t/session_deep_research_agent_abc123.jsonl")

        self.assertEqual(
            subagent_type_from_path(self.main_path, sibling), "deep_research_agent"
        )

    def test_a_name_with_no_type_left_becomes_unknown(self):
        """A run id alone is not a type, so it is not reported as one."""
        sibling = Path("/t/session_123456.jsonl")

        self.assertEqual(subagent_type_from_path(self.main_path, sibling), "unknown")

    def test_an_empty_type_becomes_unknown(self):
        sibling = Path("/t/session__123456.jsonl")

        self.assertEqual(subagent_type_from_path(self.main_path, sibling), "unknown")


if __name__ == "__main__":
    unittest.main()
