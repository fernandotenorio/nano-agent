# tools/ignore.py

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pathspec

#
# Patterns that Prisma should almost never expose.
# These use gitignore (gitwildmatch) syntax.
#
BUILTIN_IGNORE_PATTERNS = (
    # VCS
    ".git/",

    # Python
    "__pycache__/",
    "*.pyc",
    "*.pyo",
    ".venv/",
    "venv/",

    # JavaScript
    "node_modules/",

    # We
    ".prisma/"
)


class IgnoreMatcher:
    """
    Determines whether workspace-relative paths should be ignored.

    Ignore sources (lowest precedence first):

        1. Built-in patterns
        2. <workspace>/.prismaignore
        3. Runtime exclude patterns

    Pattern syntax follows Git's gitignore specification.
    """

    def __init__(
        self,
        workspace: Path,
        extra_patterns: Iterable[str] | None = None,
    ) -> None:

        self.workspace = workspace.resolve()

        patterns: list[str] = list(BUILTIN_IGNORE_PATTERNS)
        patterns.extend(self._load_prismaignore())

        if extra_patterns:
            patterns.extend(
                p.strip()
                for p in extra_patterns
                if isinstance(p, str) and p.strip()
            )

        self._spec = pathspec.PathSpec.from_lines(
            "gitignore",
            patterns,
        )

    def ignores_relative(
        self,
        relative_path: str,
        *,
        is_dir: bool,
    ) -> bool:
        """
        Returns True if a workspace-relative path should be ignored.

        Parameters
        ----------
        relative_path:
            Path relative to the workspace root, using either '/' or the
            platform separator.

        is_dir:
            True if the path represents a directory. Gitignore semantics
            distinguish between 'foo' and 'foo/'.
        """

        normalized = relative_path.replace("\\", "/").lstrip("/")

        if is_dir and normalized and not normalized.endswith("/"):
            normalized += "/"

        return self._spec.match_file(normalized)

    def _load_prismaignore(self) -> list[str]:
        """
        Loads <workspace>/.prismaignore if present.
        """

        ignore_file = self.workspace / ".prismaignore"

        try:
            return ignore_file.read_text(
                encoding="utf-8",
                errors="ignore",
            ).splitlines()
        except OSError:
            return []