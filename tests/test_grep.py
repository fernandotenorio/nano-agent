import os
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from sessioncontext import InvocationContext
from tools import grep
from tools.grep import _grep_impl, register_grep_tools
from typedefs import ToolFailure, ToolResult
from helpers import unwrap

RG = shutil.which("rg")
GIT = shutil.which("git")


@unittest.skipUnless(RG, "ripgrep (rg) is not installed")
class TestGrepTool(unittest.IsolatedAsyncioTestCase):
    """
    Test Suite for the Grep tool.

    Covers the three output modes, ripgrep flag plumbing, the ignore rules
    shared with Glob/ls, output caps, and every failure path.
    """

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.test_dir.name).resolve()

        self.ctx = InvocationContext(
            workspace=self.workspace,
            cwd=self.workspace,
            workspace_is_git_repo=False,
            resume_file=None,
        )

    def tearDown(self):
        self.test_dir.cleanup()

    # ---------------------------------------------------------
    # HELPERS
    # ---------------------------------------------------------

    def _create_file(self, relative_path: str, content: str, age_seconds: int = 0) -> Path:
        """Creates a file, optionally backdating it (Grep sorts newest first)."""
        path = self.workspace / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

        if age_seconds > 0:
            past = time.time() - age_seconds
            os.utime(path, (past, past))

        return path

    def _git(self, *args: str) -> None:
        subprocess.run([GIT, *args], cwd=self.workspace, capture_output=True, check=True)

    async def _content(self, kwargs: dict) -> str:
        """Runs Grep and returns the LLM-facing content, asserting success."""
        result = await _grep_impl(kwargs, self.ctx)
        self.assertIsInstance(result, ToolResult, getattr(result, "error_message", ""))
        return unwrap(result)

    # ---------------------------------------------------------
    # GROUP 1: VALIDATION & ERROR HANDLING
    # ---------------------------------------------------------

    async def test_missing_pattern(self):
        result = await _grep_impl({}, self.ctx)
        self.assertIsInstance(result, ToolFailure)
        self.assertIn("pattern is required", result.error_message)

    async def test_non_string_path(self):
        result = await _grep_impl({"pattern": "needle", "path": 42}, self.ctx)
        self.assertIsInstance(result, ToolFailure)
        self.assertIn("path must be a string", result.error_message)

    async def test_unknown_output_mode(self):
        result = await _grep_impl({"pattern": "needle", "output_mode": "sideways"}, self.ctx)
        self.assertIsInstance(result, ToolFailure)
        self.assertIn("output_mode must be one of", result.error_message)

    async def test_missing_path(self):
        result = await _grep_impl({"pattern": "needle", "path": "nowhere"}, self.ctx)
        self.assertIsInstance(result, ToolFailure)
        self.assertIn("Path does not exist", result.error_message)

    async def test_out_of_workspace_denied(self):
        with tempfile.TemporaryDirectory() as external:
            external_path = Path(external).resolve()
            (external_path / "secret.txt").write_text("needle", encoding="utf-8")

            result = await _grep_impl({"pattern": "needle", "path": str(external_path)}, self.ctx)

            self.assertIsInstance(result, ToolFailure)
            self.assertIn("outside", result.error_message)
            self.assertIn("Access denied", result.error_message)
            self.assertNotIn("secret.txt", result.error_message)

    async def test_invalid_regex_reports_ripgreps_explanation(self):
        self._create_file("main.py", "needle")

        result = await _grep_impl({"pattern": "(unclosed"}, self.ctx)

        self.assertIsInstance(result, ToolFailure)
        self.assertIn("regex parse error", result.error_message)

    async def test_unknown_type_is_reported(self):
        self._create_file("main.py", "needle")

        result = await _grep_impl({"pattern": "needle", "type": "bogus-lang"}, self.ctx)

        self.assertIsInstance(result, ToolFailure)
        self.assertIn("file type", result.error_message)

    async def test_missing_ripgrep_is_explained(self):
        self._create_file("main.py", "needle")

        with patch("tools.grep.shutil.which", return_value=None):
            result = await _grep_impl({"pattern": "needle"}, self.ctx)

        self.assertIsInstance(result, ToolFailure)
        self.assertIn("ripgrep", result.error_message)
        self.assertIn("Glob", result.error_message)

    async def test_timeout_is_reported(self):
        self._create_file("main.py", "needle")

        with patch("tools.grep.RG_TIMEOUT", 0.0):
            result = await _grep_impl({"pattern": "needle"}, self.ctx)

        self.assertIsInstance(result, ToolFailure)
        self.assertIn("timed out", result.error_message)

    # ---------------------------------------------------------
    # GROUP 2: CONTENT MODE
    # ---------------------------------------------------------

    async def test_basic_content_search(self):
        self._create_file("main.py", "first line\nneedle in a haystack\nthird line\n")

        content = await self._content({"pattern": "needle"})

        self.assertIn(str(self.workspace / "main.py"), content)
        self.assertIn("2:needle in a haystack", content)
        self.assertNotIn("third line", content)

    async def test_no_matches(self):
        self._create_file("main.py", "nothing to see here")

        result = await _grep_impl({"pattern": "needle"}, self.ctx)

        self.assertEqual(unwrap(result), "No matches found.")
        self.assertEqual(result.ui_summary, "Found 0 matches")

    async def test_regex_syntax_is_supported(self):
        self._create_file("main.py", "def alpha():\n    pass\ndef beta():\n")

        content = await self._content({"pattern": r"def (alpha|beta)\(\)"})

        self.assertIn("1:def alpha():", content)
        self.assertIn("3:def beta():", content)
        self.assertNotIn("pass", content)

    async def test_case_sensitive_by_default(self):
        self._create_file("main.py", "NEEDLE")

        result = await _grep_impl({"pattern": "needle"}, self.ctx)

        self.assertEqual(unwrap(result), "No matches found.")

    async def test_case_insensitive_flag(self):
        self._create_file("main.py", "NEEDLE")

        content = await self._content({"pattern": "needle", "-i": True})

        self.assertIn("1:NEEDLE", content)

    async def test_line_numbers_can_be_disabled(self):
        self._create_file("main.py", "needle here")

        content = await self._content({"pattern": "needle", "-n": False})

        self.assertIn("needle here", content)
        self.assertNotIn("1:needle here", content)

    async def test_context_lines_around_match(self):
        self._create_file("main.py", "one\ntwo\nneedle\nfour\nfive\n")

        content = await self._content({"pattern": "needle", "-C": 1})

        self.assertIn("2-two", content)
        self.assertIn("3:needle", content)
        self.assertIn("4-four", content)
        self.assertNotIn("one", content)

    async def test_asymmetric_context_flags(self):
        self._create_file("main.py", "one\ntwo\nneedle\nfour\nfive\n")

        content = await self._content({"pattern": "needle", "-B": 2, "-A": 1})

        self.assertIn("1-one", content)
        self.assertIn("2-two", content)
        self.assertIn("4-four", content)
        self.assertNotIn("5-five", content)

    async def test_separated_context_blocks_are_marked(self):
        lines = ["filler"] * 12
        lines[1] = "needle one"
        lines[10] = "needle two"
        self._create_file("main.py", "\n".join(lines) + "\n")

        content = await self._content({"pattern": "needle", "-C": 1})

        self.assertIn("2:needle one", content)
        self.assertIn("11:needle two", content)
        self.assertIn("...", content)

    async def test_multiline_pattern(self):
        self._create_file("main.py", "alpha\nbeta\n")

        single_line = await _grep_impl({"pattern": r"alpha\s+beta"}, self.ctx)
        self.assertEqual(unwrap(single_line), "No matches found.")

        content = await self._content({"pattern": r"alpha\s+beta", "multiline": True})
        self.assertIn("alpha", content)
        self.assertIn("beta", content)

    async def test_line_anchors_work_on_crlf_files(self):
        """Windows files are CRLF; '$' must not be defeated by the stray \\r."""
        (self.workspace / "main.py").write_bytes(b"needle\r\nother\r\n")

        content = await self._content({"pattern": "^needle$"})

        self.assertIn("1:needle", content)

    async def test_long_lines_are_clipped(self):
        self._create_file("main.py", "needle " + ("x" * 5000) + "\n")

        content = await self._content({"pattern": "needle"})

        longest = max(len(line) for line in content.splitlines())
        self.assertLess(longest, grep.MAX_LINE_COLUMNS + 150)

    async def test_results_are_ordered_newest_first(self):
        self._create_file("old.py", "needle", age_seconds=500)
        self._create_file("new.py", "needle")

        content = await self._content({"pattern": "needle"})

        self.assertLess(
            content.index(str(self.workspace / "new.py")),
            content.index(str(self.workspace / "old.py")),
        )

    async def test_single_file_target_keeps_its_header(self):
        self._create_file("main.py", "needle")
        self._create_file("other.py", "needle")

        content = await self._content({"pattern": "needle", "path": str(self.workspace / "main.py")})

        self.assertIn(str(self.workspace / "main.py"), content)
        self.assertIn("1:needle", content)
        self.assertNotIn("other.py", content)

    async def test_subdirectory_target(self):
        self._create_file("src/main.py", "needle")
        self._create_file("docs/readme.md", "needle")

        content = await self._content({"pattern": "needle", "path": str(self.workspace / "src")})

        self.assertIn(str(self.workspace / "src" / "main.py"), content)
        self.assertNotIn("readme.md", content)

    async def test_relative_path_resolves_against_cwd(self):
        self._create_file("src/main.py", "needle")
        self._create_file("docs/readme.md", "needle")

        content = await self._content({"pattern": "needle", "path": "src"})

        self.assertIn(str(self.workspace / "src" / "main.py"), content)
        self.assertNotIn("readme.md", content)

    # ---------------------------------------------------------
    # GROUP 3: FILE FILTERS
    # ---------------------------------------------------------

    async def test_glob_filter(self):
        self._create_file("main.py", "needle")
        self._create_file("notes.md", "needle")

        content = await self._content({"pattern": "needle", "glob": "*.py"})

        self.assertIn("main.py", content)
        self.assertNotIn("notes.md", content)

    async def test_glob_brace_expansion(self):
        self._create_file("app.ts", "needle")
        self._create_file("app.tsx", "needle")
        self._create_file("app.js", "needle")

        content = await self._content({"pattern": "needle", "glob": "*.{ts,tsx}"})

        self.assertIn("app.ts", content)
        self.assertIn("app.tsx", content)
        self.assertNotIn("app.js", content)

    async def test_glob_accepts_a_list(self):
        self._create_file("main.py", "needle")
        self._create_file("notes.md", "needle")
        self._create_file("app.js", "needle")

        content = await self._content({"pattern": "needle", "glob": ["*.py", "*.md"]})

        self.assertIn("main.py", content)
        self.assertIn("notes.md", content)
        self.assertNotIn("app.js", content)

    async def test_type_filter(self):
        self._create_file("main.py", "needle")
        self._create_file("notes.md", "needle")

        content = await self._content({"pattern": "needle", "type": "py"})

        self.assertIn("main.py", content)
        self.assertNotIn("notes.md", content)

    # ---------------------------------------------------------
    # GROUP 4: OTHER OUTPUT MODES
    # ---------------------------------------------------------

    async def test_files_with_matches_returns_absolute_paths(self):
        self._create_file("src/main.py", "needle\nneedle\n")
        self._create_file("notes.md", "nothing")

        result = await _grep_impl(
            {"pattern": "needle", "output_mode": "files_with_matches"}, self.ctx
        )
        content = unwrap(result)

        self.assertEqual(content, str(self.workspace / "src" / "main.py"))
        self.assertEqual(result.ui_summary, "Found 1 file with matches")

    async def test_files_with_matches_round_trips_into_read(self):
        from tools.filesystem import _read_impl

        self._create_file("src/main.py", "needle in the source")

        content = await self._content(
            {"pattern": "needle", "output_mode": "files_with_matches"}
        )

        read_result = await _read_impl({"file_path": content.splitlines()[0]}, self.ctx)

        self.assertIn("needle in the source", unwrap(read_result))

    async def test_count_mode(self):
        self._create_file("main.py", "needle\nneedle\nother\n")
        self._create_file("notes.md", "needle\n")

        result = await _grep_impl({"pattern": "needle", "output_mode": "count"}, self.ctx)
        content = unwrap(result)

        self.assertIn(f"{self.workspace / 'main.py'}: 2", content)
        self.assertIn(f"{self.workspace / 'notes.md'}: 1", content)
        self.assertIn("Total: 3 matches in 2 files", content)
        self.assertEqual(result.ui_summary, "Found 3 matches in 2 files")

    # ---------------------------------------------------------
    # GROUP 5: CAPS & TRUNCATION
    # ---------------------------------------------------------

    async def test_head_limit_truncates_content(self):
        self._create_file("main.py", "needle\nneedle\nneedle\n")

        result = await _grep_impl({"pattern": "needle", "head_limit": 2}, self.ctx)
        content = unwrap(result)

        self.assertEqual(content.count(":needle"), 2)
        self.assertIn("Results are truncated to 2 lines", content)
        self.assertIn("truncated", result.ui_summary)

    @patch("tools.grep.MAX_GREP_LINES", 1)
    async def test_content_cap_is_enforced(self):
        self._create_file("main.py", "needle\nneedle\nneedle\n")

        content = await self._content({"pattern": "needle"})

        self.assertEqual(content.count(":needle"), 1)
        self.assertIn("Results are truncated to 1 lines", content)

    @patch("tools.grep.MAX_GREP_FILES", 1)
    async def test_file_list_cap_is_enforced(self):
        self._create_file("new.py", "needle")
        self._create_file("old.py", "needle", age_seconds=500)

        result = await _grep_impl(
            {"pattern": "needle", "output_mode": "files_with_matches"}, self.ctx
        )
        content = unwrap(result)

        self.assertIn(str(self.workspace / "new.py"), content)
        self.assertNotIn(str(self.workspace / "old.py"), content)
        self.assertIn("Results are truncated to 1 files", content)
        self.assertIn("showing 1", result.ui_summary)

    @patch("tools.grep.MAX_GREP_FILES", 1)
    async def test_count_cap_still_reports_the_total(self):
        self._create_file("new.py", "needle")
        self._create_file("old.py", "needle", age_seconds=500)

        content = await self._content({"pattern": "needle", "output_mode": "count"})

        self.assertIn("Results are truncated to 1 files", content)
        self.assertIn("Total: 2 matches in 2 files", content)

    # ---------------------------------------------------------
    # GROUP 6: IGNORE RULES (shared with Glob and ls)
    # ---------------------------------------------------------

    async def test_builtin_ignores_are_applied(self):
        self._create_file("__pycache__/module.py", "needle")
        self._create_file("venv/lib/thing.py", "needle")
        self._create_file("node_modules/pkg/index.js", "needle")
        self._create_file(".git/config", "needle")
        self._create_file("main.py", "needle")

        content = await self._content({"pattern": "needle"})

        self.assertIn("main.py", content)
        self.assertNotIn("__pycache__", content)
        self.assertNotIn("venv", content)
        self.assertNotIn("node_modules", content)
        self.assertNotIn(".git", content)

    async def test_prismaignore_is_applied(self):
        (self.workspace / ".prismaignore").write_text("*.sqlite\nsecrets/\n", encoding="utf-8")
        self._create_file("database.sqlite", "needle")
        self._create_file("secrets/key.txt", "needle")
        self._create_file("main.py", "needle")

        content = await self._content({"pattern": "needle"})

        self.assertIn("main.py", content)
        self.assertNotIn("database.sqlite", content)
        self.assertNotIn("key.txt", content)

    async def test_runtime_exclude_list(self):
        self._create_file("vendor/lib.py", "needle")
        self._create_file("debug.log", "needle")
        self._create_file("main.py", "needle")

        content = await self._content({"pattern": "needle", "exclude": ["vendor/", "*.log"]})

        self.assertIn("main.py", content)
        self.assertNotIn("vendor", content)
        self.assertNotIn("debug.log", content)

    async def test_runtime_exclude_string_defensive_check(self):
        self._create_file("ignored.bak", "needle")
        self._create_file("main.py", "needle")

        content = await self._content({"pattern": "needle", "exclude": "*.bak"})

        self.assertIn("main.py", content)
        self.assertNotIn("ignored.bak", content)

    async def test_hidden_files_are_searched(self):
        """Grep matches Glob's DOTGLOB behaviour: dotfiles are not invisible."""
        self._create_file(".env", "SECRET=needle")

        content = await self._content({"pattern": "needle"})

        self.assertIn(".env", content)

    async def test_temporary_ignore_file_is_cleaned_up(self):
        self._create_file("main.py", "needle")
        created: list[str] = []
        real_writer = grep._write_ignore_file

        def spy(patterns):
            path = real_writer(patterns)
            created.append(path)
            return path

        with patch("tools.grep._write_ignore_file", side_effect=spy):
            await self._content({"pattern": "needle"})

        self.assertEqual(len(created), 1)
        self.assertFalse(Path(created[0]).exists())

    @unittest.skipUnless(GIT, "git is not installed")
    async def test_gitignore_is_respected(self):
        """Git-ignored files are invisible, exactly as they are to Glob and ls."""
        self._git("init")

        (self.workspace / ".gitignore").write_text("*.log\nbuild/\n", encoding="utf-8")
        self._create_file("scratch.log", "needle")
        self._create_file("build/output.bin", "needle")
        self._create_file("main.py", "needle")

        content = await self._content({"pattern": "needle"})

        self.assertIn("main.py", content)
        self.assertNotIn("scratch.log", content)
        self.assertNotIn("output.bin", content)

    @unittest.skipUnless(GIT, "git is not installed")
    async def test_tracked_files_stay_searchable(self):
        """A tracked file must be searchable even when .gitignore matches it.

        ripgrep's native .gitignore handling hides those files, while git (and
        so ls and Glob) keeps showing them. Searching through IgnoreMatcher's
        export instead is what stops Grep from silently missing deliberately
        committed files, such as a '.env.example' under '.env*'.
        """
        self._git("init")

        (self.workspace / ".gitignore").write_text("*.log\ndist/\n", encoding="utf-8")
        self._create_file("scratch.log", "needle")
        self._create_file("committed.log", "needle")
        self._create_file("dist/keep.txt", "needle")
        self._create_file("dist/junk.bin", "needle")
        self._git("add", "-f", "committed.log", "dist/keep.txt")

        content = await self._content({"pattern": "needle"})

        self.assertIn("committed.log", content)
        self.assertIn("keep.txt", content)
        self.assertNotIn("scratch.log", content)
        self.assertNotIn("junk.bin", content)

    @unittest.skipUnless(GIT, "git is not installed")
    async def test_filters_still_apply_to_tracked_gitignored_files(self):
        """Resurrected files go through normal traversal, so glob still filters."""
        self._git("init")

        (self.workspace / ".gitignore").write_text("*.log\n", encoding="utf-8")
        self._create_file("committed.log", "needle")
        self._create_file("main.py", "needle")
        self._git("add", "-f", "committed.log")

        content = await self._content({"pattern": "needle", "glob": "*.py"})

        self.assertIn("main.py", content)
        self.assertNotIn("committed.log", content)

    @unittest.skipUnless(GIT, "git is not installed")
    async def test_gitignored_path_with_glob_characters_is_hidden(self):
        """'[' is legal in a filename but is a character class in glob syntax."""
        self._git("init")

        (self.workspace / ".gitignore").write_text("*.log\n", encoding="utf-8")
        self._create_file("weird[1].log", "needle")
        self._create_file("main.py", "needle")

        content = await self._content({"pattern": "needle"})

        self.assertIn("main.py", content)
        self.assertNotIn("weird[1].log", content)

    # ---------------------------------------------------------
    # GROUP 7: REGISTRATION
    # ---------------------------------------------------------

    def test_registry_binding(self):
        mock_registry = MagicMock()

        register_grep_tools(mock_registry, self.ctx)

        mock_registry.register.assert_called_once()
        call_kwargs = mock_registry.register.call_args[1]

        self.assertEqual(call_kwargs["name"], "Grep")
        self.assertEqual(call_kwargs["input_schema"]["required"], ["pattern"])
        # Grep must survive clone_readonly() so PLAN mode can still search.
        self.assertTrue(call_kwargs["is_readonly"])


if __name__ == "__main__":
    unittest.main()
