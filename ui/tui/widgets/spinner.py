# ui/tui/widgets/spinner.py
"""
The one moving thing on screen.

Nothing streams from the model, so a long wait is otherwise indistinguishable
from a hang. The elapsed count is part of the point: it says how long, not
merely that something is happening.
"""

from __future__ import annotations

import time

from textual.widgets import Static

FRAMES = "\u280b\u2819\u2839\u2838\u283c\u2834\u2826\u2827\u2807\u280f"
INTERVAL = 1 / 12.5


class SpinnerLine(Static):
    """An animated 'summary (12s)' line, mounted while work is in flight."""

    def __init__(self, summary: str) -> None:
        super().__init__("", markup=False, classes="spinner")
        self._summary = summary
        self._frame = 0
        self._started = time.monotonic()

    def on_mount(self) -> None:
        self._tick()
        self.set_interval(INTERVAL, self._tick)

    def _tick(self) -> None:
        self._frame = (self._frame + 1) % len(FRAMES)
        elapsed = time.monotonic() - self._started
        self.update(f"{FRAMES[self._frame]} {self._summary} ({elapsed:.0f}s)")
