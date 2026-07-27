# tools/ignore.py

from __future__ import annotations

import subprocess
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

# Upper bound for the `git ls-files` query. Git is fast even on large repos
# (fully ignored directories are collapsed), but we must never hang a tool.
GIT_QUERY_TIMEOUT: float = 5.0

#
# Neutralizes the glob metacharacters that can appear in a real filename.
#
# Bracket expressions are used rather than backslash escapes because a
# backslash is not an escape character in gitignore globs on Windows (it is a
# path separator), so '\[' would silently fail to match there. ']' needs no
# escape: with every '[' neutralized, no character class is ever open.
#
_GLOB_METACHARACTERS = {
    "*": "[*]",
    "?": "[?]",
    "[": "[[]",
}


def as_literal_pattern(relative_path: str) -> str:
    """
    Converts a workspace-relative path into a gitignore pattern matching only
    that exact path.

    The leading '/' anchors the pattern to the workspace root, which also
    sidesteps gitignore's line-level syntax: a pattern starting with '/' can
    never be read as a '#' comment or a '!' negation.
    """
    escaped = "".join(_GLOB_METACHARACTERS.get(char, char) for char in relative_path)

    return f"/{escaped}"


class IgnoreMatcher:
    """
    Determines whether workspace-relative paths should be ignored.

    Ignore sources:

        1. Built-in patterns
        2. <workspace>/.prismaignore
        3. Runtime exclude patterns
        4. Paths git already ignores (asked, never parsed)

    Sources 1-3 are gitignore-syntax patterns and follow Git's precedence
    rules among themselves. Source 4 is a set of concrete paths and is applied
    as a union: git's verdict cannot be negated by a '!' rule.

    Pattern syntax follows Git's gitignore specification.
    """

    def __init__(
        self,
        workspace: Path,
        extra_patterns: Iterable[str] | None = None,
        use_git: bool = True,
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

        self._patterns = patterns

        self._spec = pathspec.PathSpec.from_lines(
            "gitignore",
            patterns,
        )

        # Concrete paths git ignores: files as exact matches, directories as
        # prefixes (a collapsed 'build/' entry stands for its whole subtree).
        self._git_files: frozenset[str] = frozenset()
        self._git_dirs: tuple[str, ...] = ()

        if use_git and self._in_git_repo():
            self._git_files, self._git_dirs = self._load_git_ignored()

    def ignores(self, path: Path, *, is_dir: bool) -> bool:
        """
        Returns True if an absolute path should be ignored.

        The path is relativized against the workspace root before matching.
        Paths that fall outside the workspace are always ignored (fail-closed):
        tools enforce the workspace boundary before walking, so this case
        should be unreachable — but if it ever happens, we must not expose
        unfiltered content from outside the workspace.
        """
        try:
            rel_path = path.absolute().relative_to(self.workspace)
        except ValueError:
            return True

        return self.ignores_relative(rel_path.as_posix(), is_dir=is_dir)

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

        return self._spec.match_file(normalized) or self._git_ignores(normalized)

    def export_patterns(self) -> list[str]:
        """
        Returns this matcher's full verdict as gitignore-syntax patterns:
        built-ins, .prismaignore, runtime excludes, and git's answers rendered
        as anchored literal paths.

        This exists so an external tool that already speaks gitignore (ripgrep's
        --ignore-file) can reproduce exactly what Glob and ls see. Exporting
        git's paths rather than letting the tool read .gitignore itself is what
        keeps the two views identical: git reports only *untracked* ignored
        paths, so a tracked file that happens to match a pattern stays
        searchable, just as it stays listable.

        Git paths are appended last, mirroring the union semantics of
        ignores_relative: a '!' rule cannot bring them back.
        """
        return [
            *self._patterns,
            *(as_literal_pattern(path) for path in sorted(self._git_files)),
            *(as_literal_pattern(path) for path in self._git_dirs),
        ]

    def _git_ignores(self, normalized: str) -> bool:
        """
        Returns True if git reported this workspace-relative path as ignored.

        Directory entries stand for their entire subtree, so children are
        matched by prefix even when traversal never pruned the parent.
        """
        if normalized in self._git_files:
            return True

        return bool(self._git_dirs) and normalized.startswith(self._git_dirs)

    def _in_git_repo(self) -> bool:
        """
        Returns True if the workspace, or any ancestor, holds a .git entry.

        Cheap pre-check that keeps the subprocess out of non-repo workspaces.
        '.git' can be a file (worktrees, submodules), hence exists() and not
        is_dir().
        """
        for directory in (self.workspace, *self.workspace.parents):
            if (directory / ".git").exists():
                return True

        return False

    def _load_git_ignored(self) -> tuple[frozenset[str], tuple[str, ...]]:
        """
        Asks git which paths it currently ignores.

        We ask rather than parse: reimplementing gitignore precedence (nested
        ignore files, .git/info/exclude, the global config, negations) would be
        a bug farm. Only *untracked* ignored paths are reported, which is the
        behaviour we want — a tracked file stays visible even if a pattern
        would otherwise match it, exactly as git itself shows it.

        Fails open (empty result) on every error: hiding files is a
        convenience, so a missing or unhappy git must never blank out a
        workspace.
        """
        try:
            completed = subprocess.run(
                [
                    "git", "ls-files",
                    "--others",            # untracked paths...
                    "--ignored",           # ...that are ignored
                    "--exclude-standard",  # honour every standard exclude source
                    "--directory",         # collapse fully ignored directories
                    "-z",                  # NUL-separated: never quoted or escaped
                ],
                cwd=self.workspace,
                capture_output=True,
                timeout=GIT_QUERY_TIMEOUT,
            )
        except (OSError, subprocess.SubprocessError):
            return frozenset(), ()

        if completed.returncode != 0:
            return frozenset(), ()

        # Paths arrive relative to the queried directory (the workspace root),
        # which is exactly the form ignores_relative matches against.
        files: set[str] = set()
        dirs: list[str] = []

        for entry in completed.stdout.decode("utf-8", errors="replace").split("\0"):
            if not entry:
                continue

            if entry.endswith("/"):
                dirs.append(entry)
            else:
                files.add(entry)

        return frozenset(files), tuple(dirs)

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