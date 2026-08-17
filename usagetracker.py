# usagetracker.py
"""
Token accounting for one session.

Every LLM call in the session lands here as one immutable `UsageRecord`, and
nothing is summed at write time. That matters because the interesting questions
are not known in advance: what did the sub-agents cost, which model ran the
most, how much went to Grep. Aggregating on the way in would answer one of them
and destroy the evidence for the rest, so the ledger stays flat and every
breakdown is a query over it.

Each record answers the three-part question the session cares about:

    (agent, activity, model)

  * `agent` is "main" or "subagent:<type>".
  * `activity` is derived, not stored: "text" for a response that only spoke,
    "tool:<name>" for each tool a response called, and "tool_internal:<name>"
    when a tool ran an LLM of its own inside its implementation.
  * `model` is the string the session *asked* for ("ollama/gemma3:12b"), not
    the one the provider echoed back ("gemma3:12b"), so that live and
    rehydrated records group together and the provider stays readable.

One response can call several tools while carrying a single usage figure, and
that figure cannot be honestly divided: the tokens paid for the whole turn, not
for any one call within it. Rather than count it once per tool (which inflates
the session total by the number of parallel calls), the record keeps the full
tuple of tool names, and the per-tool views split the turn evenly with the
remainder going to the first tool. The approximation therefore lives in the
view, where it can be labelled, and the ledger stays exact.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from transcript import Transcript
from typedefs import AssistantMessage, ToolResultMessageContent, UserMessage
from ui.base import UsageReport, UsageRow, UsageSection, split_model

MAIN_AGENT = "main"
"""The agent name of the top-level loop."""

SUBAGENT_PREFIX = "subagent:"

UNKNOWN = "unknown"
"""Stand-in for a field a resumed transcript cannot supply."""

TOOL_SPLIT_NOTE = (
    "A response that calls several tools is split evenly between them: the "
    "tokens paid for the whole turn, not for one call within it."
)


def subagent_name(subagent_type: str) -> str:
    """Builds the agent name for a sub-agent of the given type."""
    return f"{SUBAGENT_PREFIX}{subagent_type}"


@dataclass(frozen=True)
class UsageRecord:
    """One LLM call: who made it, with what, and what it cost.

    `tools` is empty for a response that only produced text. `internal` marks a
    call made *inside* a tool implementation rather than by an agent loop, in
    which case `tools` names the tool that made it.
    """

    agent: str
    model: str
    tools: tuple[str, ...] = ()
    internal: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def activities(self) -> tuple[str, ...]:
        """The activity keys this record contributes to.

        More than one only when a single response called several tools.
        """
        if self.internal:
            return tuple(f"tool_internal:{name}" for name in self.tools) or (
                f"tool_internal:{UNKNOWN}",
            )

        if self.tools:
            return tuple(f"tool:{name}" for name in self.tools)

        return ("text",)


@dataclass
class UsageTotals:
    """A running sum over some subset of the ledger.

    `calls` counts whatever the bucket is keyed by, so it means LLM calls in
    most views and tool invocations in the per-tool view: a response calling
    two tools is one call but two invocations.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def add(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_tokens: int = 0,
        calls: int = 1,
    ) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.cached_tokens += cached_tokens
        self.calls += calls

    def as_row(self, label: str) -> UsageRow:
        return UsageRow(
            label=label,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cached_tokens=self.cached_tokens,
            calls=self.calls,
        )


def _count(value: Any) -> int:
    """Reads a token count, treating anything odd as zero.

    Providers disagree about which fields exist and occasionally send None or a
    float; a usage figure is never worth raising over.
    """
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value if value > 0 else 0
    if isinstance(value, float) and value > 0:
        return int(value)
    return 0


def _cached_tokens(usage: Mapping[str, Any]) -> int:
    """Reads the cache-hit count under either name providers use for it.

    OpenAI nests it in `prompt_tokens_details`, Anthropic reports
    `cache_read_input_tokens` at the top level. The nested dict is present but
    None often enough that indexing straight into it is a real crash.
    """
    details = usage.get("prompt_tokens_details")
    if isinstance(details, Mapping):
        cached = _count(details.get("cached_tokens"))
        if cached:
            return cached

    return _count(usage.get("cache_read_input_tokens"))


def _split(value: int, parts: int) -> list[int]:
    """Divides `value` into `parts` shares, remainder to the first.

    The shares always add back up to `value`, which is what keeps a split
    column consistent with the total beside it.
    """
    if parts <= 0:
        return []

    base, remainder = divmod(value, parts)
    return [base + remainder if index == 0 else base for index in range(parts)]


class SessionUsageTracker:
    """The session's ledger of LLM calls, and the views over it."""

    def __init__(self, records: Iterable[UsageRecord] | None = None) -> None:
        self.records: list[UsageRecord] = list(records or ())

    # --- Recording -----------------------------------------------------------

    def record_turn(
        self,
        agent: str,
        model: str,
        tools: Sequence[str] = (),
        usage: Mapping[str, Any] | None = None,
    ) -> None:
        """Records one agent-loop response.

        `tools` is every tool the response called, so the turn is stored once
        no matter how many calls it made.
        """
        self._record(agent, model, tuple(tools), internal=False, usage=usage)

    def record_tool_internal(
        self,
        agent: str,
        model: str,
        tool: str,
        usage: Mapping[str, Any] | None = None,
    ) -> None:
        """Records an LLM call made inside a tool implementation."""
        self._record(agent, model, (tool,), internal=True, usage=usage)

    def _record(
        self,
        agent: str,
        model: str,
        tools: tuple[str, ...],
        internal: bool,
        usage: Mapping[str, Any] | None,
    ) -> None:
        # No usage figure means nothing to account for. Local models routinely
        # omit it, and an empty record would only add noise to every view.
        if not isinstance(usage, Mapping):
            return

        input_tokens = _count(usage.get("prompt_tokens"))
        output_tokens = _count(usage.get("completion_tokens"))
        cached_tokens = _cached_tokens(usage)

        if not (input_tokens or output_tokens or cached_tokens):
            return

        self.records.append(
            UsageRecord(
                agent=agent or UNKNOWN,
                model=model or UNKNOWN,
                tools=tools,
                internal=internal,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_tokens=cached_tokens,
            )
        )

    # --- Views ---------------------------------------------------------------

    def total(self) -> UsageTotals:
        """The session total. Every record counted exactly once."""
        totals = UsageTotals()
        for record in self.records:
            totals.add(
                record.input_tokens,
                record.output_tokens,
                record.cached_tokens,
            )
        return totals

    def total_tokens(self) -> int:
        return self.total().total_tokens

    def by_agent(self) -> dict[str, UsageTotals]:
        """Main loop against each sub-agent. Sums to the session total."""
        return self._grouped(lambda record: record.agent)

    def by_model(self) -> dict[str, UsageTotals]:
        """Sums to the session total."""
        return self._grouped(lambda record: record.model)

    def by_provider(self) -> dict[str, UsageTotals]:
        """Sums to the session total. Unprefixed models group under 'unknown'."""
        return self._grouped(lambda record: split_model(record.model)[0] or UNKNOWN)

    def _grouped(self, key: Callable[[UsageRecord], str]) -> dict[str, UsageTotals]:
        buckets: dict[str, UsageTotals] = defaultdict(UsageTotals)
        for record in self.records:
            buckets[key(record)].add(
                record.input_tokens,
                record.output_tokens,
                record.cached_tokens,
            )
        return dict(buckets)

    def by_tool(self) -> dict[str, UsageTotals]:
        """Tokens attributed to each tool, splitting multi-tool turns evenly.

        Text-only responses belong to no tool, so this sums to the tool-calling
        part of the session rather than to the whole of it.
        """
        buckets: dict[str, UsageTotals] = defaultdict(UsageTotals)

        for record in self.records:
            for name, share in self._shares(record, record.tools):
                buckets[name].add(*share)

        return dict(buckets)

    def by_activity(self) -> dict[tuple[str, str, str], UsageTotals]:
        """The full (agent, activity, model) breakdown. Sums to the session total."""
        buckets: dict[tuple[str, str, str], UsageTotals] = defaultdict(UsageTotals)

        for record in self.records:
            for activity, share in self._shares(record, record.activities):
                buckets[(record.agent, activity, record.model)].add(*share)

        return dict(buckets)

    @staticmethod
    def _shares(
        record: UsageRecord,
        keys: Sequence[str],
    ) -> list[tuple[str, tuple[int, int, int]]]:
        """Pairs each key with its slice of the record's tokens."""
        parts = len(keys)
        if parts == 0:
            return []

        inputs = _split(record.input_tokens, parts)
        outputs = _split(record.output_tokens, parts)
        cached = _split(record.cached_tokens, parts)

        return [
            (key, (inputs[index], outputs[index], cached[index]))
            for index, key in enumerate(keys)
        ]


# ---------------------------------------------------------
# Report building (the UI-facing view)
# ---------------------------------------------------------


def _rows(buckets: Mapping[str, UsageTotals]) -> tuple[UsageRow, ...]:
    """Orders a bucket map for display: costliest first, ties broken by name."""
    ordered = sorted(
        buckets.items(),
        key=lambda item: (-item[1].total_tokens, item[0]),
    )
    return tuple(totals.as_row(label) for label, totals in ordered)


def build_report(tracker: SessionUsageTracker) -> UsageReport:
    """Turns the ledger into the tables the usage view renders.

    The provider table is omitted when everything came from one provider: it
    would repeat the model table a row at a time and say nothing new.
    """
    if not tracker.records:
        return UsageReport()

    totals = tracker.total()
    sections: list[UsageSection] = [
        UsageSection(title="By agent", rows=_rows(tracker.by_agent())),
        UsageSection(title="By model", rows=_rows(tracker.by_model())),
    ]

    providers = tracker.by_provider()
    if len(providers) > 1:
        sections.append(UsageSection(title="By provider", rows=_rows(providers)))

    tools = tracker.by_tool()
    if tools:
        sections.append(
            UsageSection(title="By tool", rows=_rows(tools), note=TOOL_SPLIT_NOTE)
        )

    return UsageReport(sections=tuple(sections), totals=totals.as_row("Total"))


# ---------------------------------------------------------
# Rehydration (--resume)
# ---------------------------------------------------------


def absorb_messages(
    messages: Iterable[Any],
    agent: str,
    tracker: SessionUsageTracker,
) -> None:
    """Replays one transcript's messages into `tracker` under `agent`.

    Reads the same two things the live session records: an assistant response
    with a usage figure, and a tool result carrying usage from an LLM the tool
    ran itself.
    """
    for message in messages:
        if isinstance(message, AssistantMessage):
            if not message.usage:
                continue

            tools = tuple(
                block.name
                for block in message.content
                if getattr(block, "type", None) == "tool_use"
            )
            tracker.record_turn(
                agent=agent,
                model=message.request_model or message.model or UNKNOWN,
                tools=tools,
                usage=message.usage,
            )

        elif isinstance(message, UserMessage) and isinstance(message.content, list):
            for block in message.content:
                if isinstance(block, ToolResultMessageContent) and block.usage:
                    tracker.record_tool_internal(
                        agent=agent,
                        model=block.internal_model or UNKNOWN,
                        tool=block.tool_name or UNKNOWN,
                        usage=block.usage,
                    )


def subagent_type_from_path(main_path: Path, sibling: Path) -> str:
    """Recovers a sub-agent's type from its transcript filename.

    Sub-agent transcripts are named '<parent stem>_<type>_<id>.jsonl' (see
    `agent.handle_subagent`), and the id is a short hex string, so the type is
    whatever is left after dropping it. A name that does not fit the shape is
    reported as unknown rather than guessed at.
    """
    remainder = sibling.stem[len(main_path.stem) + 1:]
    subagent_type, separator, _run_id = remainder.rpartition("_")

    return subagent_type if separator and subagent_type else UNKNOWN


def rehydrate_session_usage(main_path: Path) -> SessionUsageTracker:
    """Rebuilds the ledger of a previous session from its transcripts.

    Sub-agents write their own sibling transcripts, so reading only the main
    one would quietly drop everything they spent. They are picked up by name
    from the same directory; a sub-agent cannot launch another sub-agent, so
    there is never a deeper level to chase.
    """
    tracker = SessionUsageTracker()

    if main_path.exists():
        absorb_messages(Transcript(main_path).messages, MAIN_AGENT, tracker)

    for sibling in sorted(main_path.parent.glob(f"{main_path.stem}_*.jsonl")):
        absorb_messages(
            Transcript(sibling).messages,
            subagent_name(subagent_type_from_path(main_path, sibling)),
            tracker,
        )

    return tracker
