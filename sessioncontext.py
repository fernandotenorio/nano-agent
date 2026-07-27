# sessioncontext.py
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from enum import Enum, auto

from capabilities import Capabilities
from filestate import FileStateTracker
from notices import NoticeLog

class AgentMode(Enum):
    BUILD = auto()
    PLAN = auto()


@dataclass
class AgentPolicy:
    mode: AgentMode = AgentMode.BUILD
    notified_mode: AgentMode | None = None


@dataclass(frozen=True)
class InvocationContext:
    workspace: Path
    cwd: Path
    workspace_is_git_repo: bool
    resume_file: Optional[Path] = None

    # What this machine can do, probed once at startup. None means unprobed
    # (unit tests, throwaway contexts), which every consumer reads as "assume
    # nothing is broken".
    capabilities: Optional[Capabilities] = None

    # Per-agent file freshness tracking (read-before-write gate + change diffs).
    # Mutable state on a frozen dataclass is intentional: the context identity
    # is fixed for the life of an agent loop, while the tracker contents evolve.
    file_state: FileStateTracker = field(default_factory=FileStateTracker)

    # One-shot warnings already delivered to this agent (same mutable-state
    # rationale as file_state).
    notices: NoticeLog = field(default_factory=NoticeLog)

    # You can easily add more CLI-derived state here later
    # (e.g., debug_mode: bool = False)

    def clone_for_subagent(self) -> "InvocationContext":
        """Returns a copy of this context with an *empty* file-state tracker.

        Sub-agents must Read files themselves before writing them; they never
        inherit the parent's read history. Notices reset for the same reason:
        a sub-agent has its own conversation and has not read the parent's
        warnings.
        """
        return dataclasses.replace(
            self,
            file_state=FileStateTracker(),
            notices=NoticeLog(),
        )