import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from sessioncontext import InvocationContext
from tools.filesearch import _glob_impl, _ls_impl
from tools.ignore import GitStatus, IgnoreMatcher
from helpers import unwrapped

# Tests assert on the raw LLM-facing content; unwrap the ToolResult envelope.
_glob_impl = unwrapped(_glob_impl)
_ls_impl = unwrapped(_ls_impl)

GIT = shutil.which("git")


@unittest.skipUnless(GIT, "git is not installed")
class TestGitIgnoreIntegration(unittest.IsolatedAsyncioTestCase):
    """
    IgnoreMatcher must hide whatever git already hides.

    These tests drive real repositories on purpose: the feature exists because
    we ask git instead of parsing .gitignore ourselves, so a mocked git would
    prove nothing about the behaviour that matters.
    """

    def setUp(self):
        # Git leaves read-only objects behind, which can trip cleanup on Windows.
        self.test_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.workspace = Path(self.test_dir.name).resolve()

        self.ctx = InvocationContext(
            workspace=self.workspace,
            cwd=self.workspace,
            workspace_is_git_repo=True,
            resume_file=None,
        )

        self._git("init")

    def tearDown(self):
        self.test_dir.cleanup()

    # ---------------------------------------------------------
    # HELPERS
    # ---------------------------------------------------------

    def _git(self, *args: str, cwd: Path | None = None) -> None:
        subprocess.run(
            [GIT, *args],
            cwd=cwd or self.workspace,
            capture_output=True,
            check=True,
        )

    def _write(self, relative_path: str, content: str = "content") -> Path:
        path = self.workspace / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def _gitignore(self, *patterns: str, directory: Path | None = None) -> None:
        target = (directory or self.workspace) / ".gitignore"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(patterns) + "\n", encoding="utf-8")

    # ---------------------------------------------------------
    # GROUP 1: Git-reported paths are ignored
    # ---------------------------------------------------------

    def test_gitignored_file_is_ignored(self):
        """A file matched by .gitignore is invisible; its neighbour is not."""
        self._gitignore("*.log")
        self._write("app.log")
        self._write("main.py")

        matcher = IgnoreMatcher(self.workspace)

        self.assertTrue(matcher.ignores_relative("app.log", is_dir=False))
        self.assertFalse(matcher.ignores_relative("main.py", is_dir=False))

    def test_gitignored_directory_hides_its_whole_subtree(self):
        """Git collapses fully ignored directories, so children match by prefix."""
        self._gitignore("build/")
        self._write("build/output.bin")
        self._write("build/nested/deep.bin")

        matcher = IgnoreMatcher(self.workspace)

        self.assertTrue(matcher.ignores_relative("build", is_dir=True))
        self.assertTrue(matcher.ignores_relative("build/output.bin", is_dir=False))
        self.assertTrue(matcher.ignores_relative("build/nested/deep.bin", is_dir=False))

    def test_nested_gitignore_is_honoured(self):
        """Ignore files below the root count too, because git resolves them."""
        self._write("src/main.py")
        self._write("src/scratch.tmp")
        self._gitignore("*.tmp", directory=self.workspace / "src")

        matcher = IgnoreMatcher(self.workspace)

        self.assertTrue(matcher.ignores_relative("src/scratch.tmp", is_dir=False))
        self.assertFalse(matcher.ignores_relative("src/main.py", is_dir=False))

    def test_glob_metacharacters_in_names_are_matched_literally(self):
        """Git reports paths, not patterns: '[1]' must not be read as a class."""
        self._gitignore("*.log")
        self._write("weird[1].log")
        self._write("weird1.log", content="also ignored by *.log")
        self._write("weirdX.txt")

        matcher = IgnoreMatcher(self.workspace)

        self.assertTrue(matcher.ignores_relative("weird[1].log", is_dir=False))
        self.assertFalse(matcher.ignores_relative("weirdX.txt", is_dir=False))

    def test_tracked_file_stays_visible(self):
        """Git shows files it tracks, even when a pattern would match them."""
        self._gitignore("*.log")
        self._write("tracked.log")
        self._write("untracked.log")

        self._git("add", "-f", "tracked.log")

        matcher = IgnoreMatcher(self.workspace)

        self.assertFalse(matcher.ignores_relative("tracked.log", is_dir=False))
        self.assertTrue(matcher.ignores_relative("untracked.log", is_dir=False))

    def test_subdirectory_workspace_resolves_paths_relatively(self):
        """A workspace inside a repo gets paths relative to itself, not the repo root."""
        self._gitignore("*.log")
        self._write("sub/app.log")
        self._write("sub/main.py")

        matcher = IgnoreMatcher(self.workspace / "sub")

        self.assertTrue(matcher.ignores_relative("app.log", is_dir=False))
        self.assertFalse(matcher.ignores_relative("main.py", is_dir=False))

    # ---------------------------------------------------------
    # GROUP 2: The other ignore sources still work
    # ---------------------------------------------------------

    def test_all_ignore_sources_stack(self):
        """Built-ins, .prismaignore, runtime excludes, and git all apply at once."""
        self._gitignore("*.log")
        (self.workspace / ".prismaignore").write_text("*.sqlite\n", encoding="utf-8")

        self._write("app.log")
        self._write("database.sqlite")
        self._write("__pycache__/module.pyc")
        self._write("runtime/generated.txt")
        self._write("main.py")

        matcher = IgnoreMatcher(self.workspace, extra_patterns=["runtime/"])

        self.assertTrue(matcher.ignores_relative("app.log", is_dir=False))
        self.assertTrue(matcher.ignores_relative("database.sqlite", is_dir=False))
        self.assertTrue(matcher.ignores_relative("__pycache__", is_dir=True))
        self.assertTrue(matcher.ignores_relative("runtime", is_dir=True))
        self.assertFalse(matcher.ignores_relative("main.py", is_dir=False))

    def test_export_patterns_carries_every_source(self):
        """The ripgrep export must reproduce what Glob and ls already see."""
        self._gitignore("*.log", "build/")
        self._write("untracked.log")
        self._write("committed.log")
        self._write("build/output.bin")
        self._git("add", "-f", "committed.log")
        (self.workspace / ".prismaignore").write_text("*.sqlite\n", encoding="utf-8")

        patterns = IgnoreMatcher(self.workspace, extra_patterns=["runtime/"]).export_patterns()

        self.assertIn("*.sqlite", patterns)          # .prismaignore
        self.assertIn("runtime/", patterns)          # runtime exclude
        self.assertIn("node_modules/", patterns)     # built-in
        self.assertIn("/untracked.log", patterns)    # git's answer, anchored
        self.assertIn("/build/", patterns)           # collapsed directory

        # Tracked files are searchable, so they must not reach the export.
        self.assertNotIn("/committed.log", patterns)

    def test_export_escapes_glob_metacharacters(self):
        """Exported paths are literals: '[1]' must not become a character class."""
        self._gitignore("*.log")
        self._write("weird[1].log")

        patterns = IgnoreMatcher(self.workspace).export_patterns()

        self.assertIn("/weird[[]1].log", patterns)

    # ---------------------------------------------------------
    # GROUP 3: ls and Glob inherit the behaviour for free
    # ---------------------------------------------------------

    async def test_ls_hides_gitignored_entries(self):
        self._gitignore("*.log", "build/")
        self._write("app.log")
        self._write("build/output.bin")
        self._write("main.py")

        result = await _ls_impl({"path": str(self.workspace), "depth": 3}, self.ctx)

        self.assertIn("main.py", result)
        self.assertNotIn("app.log", result)
        self.assertNotIn("output.bin", result)

    async def test_glob_hides_gitignored_entries(self):
        self._gitignore("secrets/")
        self._write("secrets/key.txt")
        self._write("notes.txt")

        result = await _glob_impl({"pattern": "**/*.txt", "path": str(self.workspace)}, self.ctx)

        self.assertIn("notes.txt", result)
        self.assertNotIn("key.txt", result)

    async def test_listings_admit_when_git_could_not_be_consulted(self):
        """Without git's verdict these listings widen, so they say so."""
        self._gitignore("*.log")
        self._write("app.log")
        self._write("main.py")

        with patch("tools.ignore.subprocess.run", side_effect=OSError("no git")):
            listing = await _ls_impl({"path": str(self.workspace)}, self.ctx)
            globbed = await _glob_impl({"pattern": "*.py", "path": str(self.workspace)}, self.ctx)

        self.assertIn("app.log", listing)
        self.assertIn("git could not be consulted", listing)

        # One notice per agent, not one per tool call.
        self.assertNotIn("git could not be consulted", globbed)

    async def test_healthy_listings_carry_no_caveat(self):
        self._gitignore("*.log")
        self._write("app.log")
        self._write("main.py")

        listing = await _ls_impl({"path": str(self.workspace)}, self.ctx)

        self.assertNotIn("git could not be consulted", listing)

    # ---------------------------------------------------------
    # GROUP 4: Opting out, and failing open
    # ---------------------------------------------------------

    def test_use_git_false_skips_git_entirely(self):
        """The opt-out restores the pre-git behaviour for callers that need it."""
        self._gitignore("*.log")
        self._write("app.log")

        with patch("tools.ignore.subprocess.run") as run:
            matcher = IgnoreMatcher(self.workspace, use_git=False)

        run.assert_not_called()
        self.assertFalse(matcher.ignores_relative("app.log", is_dir=False))
        self.assertIs(matcher.git_status, GitStatus.UNUSED)

    def test_git_error_fails_open(self):
        """A broken git must never blank out the workspace."""
        self._gitignore("*.log")
        self._write("app.log")

        with patch("tools.ignore.subprocess.run", side_effect=OSError("git is missing")):
            matcher = IgnoreMatcher(self.workspace)

        self.assertFalse(matcher.ignores_relative("app.log", is_dir=False))
        self.assertTrue(matcher.ignores_relative("__pycache__", is_dir=True))

    def test_git_timeout_fails_open(self):
        self._gitignore("*.log")
        self._write("app.log")

        timeout = subprocess.TimeoutExpired(cmd="git", timeout=5.0)
        with patch("tools.ignore.subprocess.run", side_effect=timeout):
            matcher = IgnoreMatcher(self.workspace)

        self.assertFalse(matcher.ignores_relative("app.log", is_dir=False))

    def test_git_nonzero_exit_fails_open(self):
        self._gitignore("*.log")
        self._write("app.log")

        failed = MagicMock(returncode=128, stdout=b"", stderr=b"fatal: not a git repository")
        with patch("tools.ignore.subprocess.run", return_value=failed):
            matcher = IgnoreMatcher(self.workspace)

        self.assertFalse(matcher.ignores_relative("app.log", is_dir=False))

    # ---------------------------------------------------------
    # GROUP 5: Reporting why git was silent
    # ---------------------------------------------------------

    def test_status_is_ok_in_a_healthy_repo(self):
        matcher = IgnoreMatcher(self.workspace)

        self.assertIs(matcher.git_status, GitStatus.OK)
        self.assertFalse(matcher.git_status.degraded)
        self.assertEqual(matcher.git_error, "")

    def test_missing_binary_is_reported_as_unavailable(self):
        with patch("tools.ignore.subprocess.run", side_effect=OSError("no git")):
            matcher = IgnoreMatcher(self.workspace)

        self.assertIs(matcher.git_status, GitStatus.UNAVAILABLE)
        self.assertTrue(matcher.git_status.degraded)
        self.assertIn("no git", matcher.git_error)

    def test_timeout_is_reported_as_failed(self):
        timeout = subprocess.TimeoutExpired(cmd="git", timeout=5.0)

        with patch("tools.ignore.subprocess.run", side_effect=timeout):
            matcher = IgnoreMatcher(self.workspace)

        self.assertIs(matcher.git_status, GitStatus.FAILED)
        self.assertTrue(matcher.git_status.degraded)

    def test_refusal_keeps_gits_own_explanation(self):
        """'dubious ownership' is the common real-world failure; keep the reason."""
        refused = MagicMock(
            returncode=128,
            stdout=b"",
            stderr=b"\nfatal: detected dubious ownership in repository at 'C:/repo'\n",
        )

        with patch("tools.ignore.subprocess.run", return_value=refused):
            matcher = IgnoreMatcher(self.workspace)

        self.assertIs(matcher.git_status, GitStatus.FAILED)
        self.assertEqual(
            matcher.git_error,
            "fatal: detected dubious ownership in repository at 'C:/repo'",
        )

    def test_refusal_without_stderr_still_explains_itself(self):
        refused = MagicMock(returncode=3, stdout=b"", stderr=b"")

        with patch("tools.ignore.subprocess.run", return_value=refused):
            matcher = IgnoreMatcher(self.workspace)

        self.assertIs(matcher.git_status, GitStatus.FAILED)
        self.assertIn("exited with code 3", matcher.git_error)


class TestNonRepoWorkspace(unittest.TestCase):
    """Outside a repository nothing changes: we never parse .gitignore ourselves."""

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.test_dir.name).resolve()

    def tearDown(self):
        self.test_dir.cleanup()

    def test_no_subprocess_without_a_git_directory(self):
        with patch("tools.ignore.subprocess.run") as run:
            matcher = IgnoreMatcher(self.workspace)

        run.assert_not_called()
        self.assertIs(matcher.git_status, GitStatus.UNUSED)
        self.assertFalse(matcher.git_status.degraded)

    def test_gitignore_is_not_parsed_outside_a_repo(self):
        (self.workspace / ".gitignore").write_text("*.log\n", encoding="utf-8")
        (self.workspace / "app.log").write_text("content", encoding="utf-8")

        matcher = IgnoreMatcher(self.workspace)

        self.assertFalse(matcher.ignores_relative("app.log", is_dir=False))

    def test_an_empty_git_folder_is_not_a_repository(self):
        """A '.git' directory without a HEAD is not worth running git over.

        Without this, every such workspace would warn about .gitignore rules
        that were never in play.
        """
        (self.workspace / ".git").mkdir()

        with patch("tools.ignore.subprocess.run") as run:
            matcher = IgnoreMatcher(self.workspace)

        run.assert_not_called()
        self.assertIs(matcher.git_status, GitStatus.UNUSED)

    def test_a_git_file_marks_a_worktree(self):
        """Worktrees and submodules use a '.git' file rather than a directory."""
        (self.workspace / ".git").write_text("gitdir: /elsewhere/.git/worktrees/x\n", encoding="utf-8")

        with patch("tools.ignore.subprocess.run", side_effect=OSError("no git")) as run:
            matcher = IgnoreMatcher(self.workspace)

        run.assert_called_once()
        self.assertIs(matcher.git_status, GitStatus.UNAVAILABLE)


if __name__ == "__main__":
    unittest.main()
