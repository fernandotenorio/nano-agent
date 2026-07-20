# sessioncontext.py
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from enum import Enum, auto

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
    
    # You can easily add more CLI-derived state here later
    # (e.g., debug_mode: bool = False)