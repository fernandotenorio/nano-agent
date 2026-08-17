# ui/tui/widgets/usage.py
"""
The usage view: what this session has spent, and on what.

A modal rather than a transcript block. The breakdown is a snapshot of the
session as a whole, not a thing that happened at a point in it, so filing it
between two turns would date it the moment the next response arrived. It is
also wide, and the transcript is not.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import DataTable, Static

from ui.base import UsageReport, UsageSection

EMPTY_MESSAGE = "No model usage recorded yet."

_COLUMNS = ("Input", "Output", "Cached", "Total", "Calls")


def _table(section: UsageSection) -> DataTable:
    """One section as a table, its title doubling as the label column header."""
    table = DataTable(classes="usage-table", show_cursor=False, zebra_stripes=True)
    table.add_column(section.title, key="label")

    for title in _COLUMNS:
        table.add_column(title, key=title.lower())

    for row in section.rows:
        table.add_row(
            row.label,
            f"{row.input_tokens:,}",
            f"{row.output_tokens:,}",
            f"{row.cached_tokens:,}",
            f"{row.total_tokens:,}",
            f"{row.calls:,}",
        )

    return table


class UsageScreen(ModalScreen[None]):
    """Token usage for the session, dismissed with Escape."""

    BINDINGS = [
        Binding("escape", "dismiss_usage", "Close", show=False),
        Binding("q", "dismiss_usage", "Close", show=False),
        Binding("ctrl+u", "dismiss_usage", "Close", show=False),
    ]

    def __init__(self, report: UsageReport) -> None:
        super().__init__()
        self._report = report

    def compose(self) -> ComposeResult:
        with Vertical(id="usage-dialog"):
            yield Static("Token usage", markup=False, id="usage-title")

            if self._report.is_empty:
                yield Static(EMPTY_MESSAGE, markup=False, classes="usage-empty")
            else:
                yield Static(self._totals_line(), markup=False, id="usage-totals")

                with VerticalScroll(id="usage-sections"):
                    for section in self._report.sections:
                        yield _table(section)
                        if section.note:
                            yield Static(section.note, markup=False, classes="usage-note")

            yield Static("esc  close", markup=False, id="usage-hint")

    def _totals_line(self) -> str:
        totals = self._report.totals
        parts = [
            f"{totals.total_tokens:,} tokens",
            f"{totals.input_tokens:,} in",
            f"{totals.output_tokens:,} out",
        ]

        # Only worth a column of its own when the provider actually reported
        # cache hits; a permanent "0 cached" says nothing.
        if totals.cached_tokens:
            parts.append(f"{totals.cached_tokens:,} cached")

        parts.append(f"{totals.calls:,} model call{'' if totals.calls == 1 else 's'}")

        return "  ".join(parts)

    def action_dismiss_usage(self) -> None:
        self.dismiss(None)
