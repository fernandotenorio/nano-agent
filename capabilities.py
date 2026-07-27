# capabilities.py
"""
What this machine can actually do, probed once per session.

Two external programs shape how Prisma searches, and each is missing in a
different way:

  * ripgrep is an optimisation. Without it Grep runs on a slower built-in
    engine, which is the user's problem (only they can install rg) and not the
    model's (each Grep backend describes its own capabilities).

  * git is a source of truth. Without it .gitignore stops being applied, so ls,
    Glob, and Grep all start showing files the repo deliberately hides. That
    changes what the model can see, so the model has to be told too.

Hence the two warning functions: `user_warnings` covers both, `model_warnings`
covers only what alters the model's view of the workspace.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

from tools.ignore import GitStatus, IgnoreMatcher


#
# Caveats appended to a tool result when git could not be consulted. Kept here
# with the rest of the degradation wording, and phrased for the model: each one
# says what the result may be wrong about, not what went wrong on the machine.
#
IGNORE_WIDENED_NOTE = (
    "(Note: git could not be consulted, so .gitignore rules were not applied. "
    "These results may include files the repository ignores, such as build "
    "output, caches, and dependency directories.)"
)

RIPGREP_GITIGNORE_NOTE = (
    "(Note: git could not be consulted, so .gitignore was applied directly by "
    "the search engine instead. A file that is committed but matches a "
    ".gitignore pattern may be missing from these results; use Glob or ls to "
    "confirm whether such a file exists.)"
)


def find_ripgrep() -> str | None:
    """Locates the ripgrep binary. The single place that looks for it."""
    return shutil.which("rg")


@dataclass(frozen=True)
class Capabilities:
    """Probe results for one session."""

    ripgrep: str | None = None
    git_status: GitStatus = GitStatus.UNUSED
    git_error: str = ""

    @property
    def git_ignores_degraded(self) -> bool:
        """True when .gitignore rules are silently not being applied."""
        return self.git_status.degraded


def probe_capabilities(workspace: Path) -> Capabilities:
    """Probes the environment once, at session start.

    The git check is the real `git ls-files` call, borrowed from a throwaway
    IgnoreMatcher rather than reimplemented: a lighter probe such as
    `git --version` would report success in exactly the case that matters most
    (a repo git refuses to touch because of dubious ownership).
    """
    matcher = IgnoreMatcher(workspace=workspace)

    return Capabilities(
        ripgrep=find_ripgrep(),
        git_status=matcher.git_status,
        git_error=matcher.git_error,
    )


def _git_warning(caps: Capabilities) -> str:
    reason = (
        "the git binary could not be run"
        if caps.git_status is GitStatus.UNAVAILABLE
        else "git refused to answer"
    )
    detail = f" ({caps.git_error})" if caps.git_error else ""

    return (
        f"This workspace is a Git repository, but {reason}{detail}, so "
        ".gitignore rules are NOT being applied. The ls, Glob, and Grep tools "
        "may show files the repository ignores, such as build output, caches, "
        "and dependency directories."
    )


def _git_remedy(caps: Capabilities) -> str:
    """The fix, which depends on how git failed."""
    if caps.git_status is GitStatus.UNAVAILABLE:
        return "Install git, or make sure it is on PATH."

    return (
        "If this is a repository you trust, "
        "`git config --global --add safe.directory <path>` often fixes it."
    )


def model_warnings(caps: Capabilities) -> list[str]:
    """Degradations that change what the model can see.

    Deliberately excludes ripgrep: which engine backs Grep is stated in that
    tool's own description, so mentioning it here would only invite the model
    to apologise for the environment.
    """
    if not caps.git_ignores_degraded:
        return []

    return [_git_warning(caps)]


def user_warnings(caps: Capabilities) -> list[str]:
    """Degradations worth showing in the startup banner.

    Everything the model is told, plus the things only the user can fix.
    """
    warnings: list[str] = []

    if caps.git_ignores_degraded:
        warnings.append(f"{_git_warning(caps)} {_git_remedy(caps)}")

    if caps.ripgrep is None:
        warnings.append(dedent("""\
            ripgrep (rg) was not found on PATH, so Grep is running on the
            built-in engine: slower, and without file-type filters. Installing
            ripgrep makes content search substantially faster.""").replace("\n", " "))

    return warnings
