import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    """
    Application-level configuration.

    Values here describe Prisma itself, not a single invocation.
    CLI arguments should remain separate.
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

    def project_config_dir(self, cwd: Path) -> Path:
        """
        Project-local Prisma configuration directory.
        Example: <cwd>/.prisma
        """
        return cwd / self.app_dir_name

    def global_system_prompt_file(self) -> Path:
        """
        Global SYSTEM.md location.
        """
        return self.home_config_dir / "SYSTEM.md"

    def project_system_prompt_file(self, cwd: Path) -> Path:
        """
        Project SYSTEM.md location.
        """
        return self.project_config_dir(cwd) / "SYSTEM.md"

    def project_transcripts_dir(self, cwd: Path) -> Path:
        """
        Project transcript storage directory.
        """
        return self.project_config_dir(cwd) / "transcripts"


def load_app_config() -> AppConfig:
    """
    Load application configuration.
    """
    app_name = "prisma"

    return AppConfig(
        app_name=app_name,
        app_dir_name=f".{app_name}",
    )