import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    """
    Application-level configuration.

    Values here describe Prisma itself, not a single invocation.
    CLI arguments should remain separate.

    Everything Prisma owns lives under the user's home directory. Nothing is
    kept inside a workspace: a project's own instructions belong in a committed
    AGENTS.md, and what Prisma records about a conversation is not part of the
    project it was working on.
    """

    app_name: str
    app_dir_name: str

    @property
    def home_config_dir(self) -> Path:
        """
        Global Prisma configuration directory.
        Example: ~/.prisma
        """
        return Path.home() / self.app_dir_name

    @property
    def projects_dir(self) -> Path:
        """
        Global store of per-project state, keyed by workspace root.
        Example: ~/.prisma/projects

        Session transcripts live here rather than inside the workspace: they are
        a record of the conversation, not part of the project, and keeping them
        out of the tree means they can never be listed, searched, or edited by
        the agent's own tools.
        """
        return self.home_config_dir / "projects"

    def global_system_prompt_file(self) -> Path:
        """
        Global SYSTEM.md location.
        """
        return self.home_config_dir / "SYSTEM.md"


def load_app_config() -> AppConfig:
    """
    Load application configuration.
    """
    app_name = "prisma"

    return AppConfig(
        app_name=app_name,
        app_dir_name=f".{app_name}",
    )