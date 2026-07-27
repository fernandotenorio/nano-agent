import os
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from sessioncontext import InvocationContext
from tools import grep_render
from tools.grep_fallback import _grep_impl, register_fallback_grep_tools
from typedefs import ToolFailure, ToolResult
from helpers import unwrap

GIT = shutil.which("git")


class TestGrepFallback(unittest.IsolatedAsyncioTestCase):
    """
    Test Suite for the built-in Grep engine used when ripgrep is absent.

    This engine has to behave like the ripgrep-backed one, because the model
    cannot tell them apart: same output shapes, same ignore rules, same caps.
    What it does not have to do is match ripgrep's feature list, so the `type`
    filter is deliberately absent from its schema.
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

    async def test_invalid_regex_is_explained(self):
        self._create_file("main.py", "needle")

        result = await _grep_impl({"pattern": "(unclosed"}, self.ctx)

        self.assertIsInstance(result, ToolFailure)
        self.assertIn("invalid regular expression", result.error_message)

    async def test_timeout_returns_partial_results(self):
        """A slow search reports what it found rather than stalling the session."""
        self._create_file("main.py", "needle")

        with patch("tools.grep_fallback.SEARCH_TIMEOUT", -1.0):
            content = await self._content({"pattern": "needle"})

        self.assertIn("stopped after", content)

    @patch("tools.grep_fallback.MAX_SCAN_FILES", 1)
    async def test_file_budget_is_admitted_as_incomplete(self):
        self._create_file("a.py", "needle")
        self._create_file("b.py", "needle")
        self._create_file("c.py", "needle")

        content = await self._content({"pattern": "needle"})

        self.assertIn("incomplete", content)

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
        self.assertLess(longest, grep_render.MAX_LINE_COLUMNS + 150)

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

    async def test_nested_files_are_searched(self):
        self._create_file("a/b/c/deep.py", "needle")

        content = await self._content({"pattern": "needle"})

        self.assertIn(str(self.workspace / "a" / "b" / "c" / "deep.py"), content)

    # ---------------------------------------------------------
    # GROUP 3: FILE FILTERS
    # ---------------------------------------------------------

    async def test_glob_filter(self):
        self._create_file("main.py", "needle")
        self._create_file("notes.md", "needle")

        content = await self._content({"pattern": "needle", "glob": "*.py"})

        self.assertIn("main.py", content)
        self.assertNotIn("notes.md", content)

    async def test_slash_free_glob_matches_at_any_depth(self):
        """gitignore semantics: '*.py' is not anchored to the search root."""
        self._create_file("src/deep/main.py", "needle")
        self._create_file("src/deep/notes.md", "needle")

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

    async def test_negated_glob_excludes(self):
        self._create_file("app.js", "needle")
        self._create_file("app.min.js", "needle")

        content = await self._content({"pattern": "needle", "glob": "!*.min.js"})

        self.assertIn("app.js", content)
        self.assertNotIn("app.min.js", content)

    async def test_directory_glob_covers_its_subtree(self):
        self._create_file("src/main.py", "needle")
        self._create_file("docs/readme.md", "needle")

        content = await self._content({"pattern": "needle", "glob": "src/"})

        self.assertIn("main.py", content)
        self.assertNotIn("readme.md", content)

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

    async def test_count_mode_counts_occurrences_not_lines(self):
        """rg --count-matches counts every occurrence; so does this engine."""
        self._create_file("main.py", "needle needle needle\n")

        content = await self._content({"pattern": "needle", "output_mode": "count"})

        self.assertIn("Total: 3 matches in 1 file", content)

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

    @patch("tools.grep_render.MAX_GREP_LINES", 1)
    async def test_content_cap_is_enforced(self):
        self._create_file("main.py", "needle\nneedle\nneedle\n")

        content = await self._content({"pattern": "needle"})

        self.assertEqual(content.count(":needle"), 1)
        self.assertIn("Results are truncated to 1 lines", content)

    @patch("tools.grep_render.MAX_GREP_FILES", 1)
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

    @patch("tools.grep_render.MAX_GREP_FILES", 1)
    async def test_count_cap_still_reports_the_total(self):
        self._create_file("new.py", "needle")
        self._create_file("old.py", "needle", age_seconds=500)

        content = await self._content({"pattern": "needle", "output_mode": "count"})

        self.assertIn("Results are truncated to 1 files", content)
        self.assertIn("Total: 2 matches in 2 files", content)

    # ---------------------------------------------------------
    # GROUP 6: SKIPPED FILES
    # ---------------------------------------------------------

    async def test_binary_files_are_skipped(self):
        (self.workspace / "blob.bin").write_bytes(b"needle\x00\x01\x02needle")
        self._create_file("main.py", "needle")

        content = await self._content({"pattern": "needle"})

        self.assertIn("main.py", content)
        self.assertNotIn("blob.bin", content)

    @patch("tools.grep_fallback.MAX_FILE_BYTES", 16)
    async def test_oversized_files_are_skipped(self):
        self._create_file("huge.py", "needle" + ("x" * 100))
        self._create_file("main.py", "needle")

        content = await self._content({"pattern": "needle"})

        self.assertIn("main.py", content)
        self.assertNotIn("huge.py", content)

    async def test_undecodable_bytes_do_not_abort_the_search(self):
        (self.workspace / "latin.txt").write_bytes(b"caf\xe9 needle\n")

        content = await self._content({"pattern": "needle"})

        self.assertIn("latin.txt", content)

    # ---------------------------------------------------------
    # GROUP 7: IGNORE RULES (shared with Glob and ls)
    # ---------------------------------------------------------

    async def test_builtin_ignores_are_applied(self):
        self._create_file("__pycache__/module.py", "needle")
        self._create_file("venv/lib/thing.py", "needle")
        self._create_file("node_modules/pkg/index.js", "needle")
        self._create_file("main.py", "needle")

        content = await self._content({"pattern": "needle"})

        self.assertIn("main.py", content)
        self.assertNotIn("__pycache__", content)
        self.assertNotIn("venv", content)
        self.assertNotIn("node_modules", content)

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

    @unittest.skipUnless(GIT, "git is not installed")
    async def test_gitignore_is_respected(self):
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
        """Consulting IgnoreMatcher, not .gitignore, is what keeps these visible."""
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
    async def test_degraded_git_is_reported_once(self):
        self._git("init")
        self._create_file("main.py", "needle")

        with patch("tools.ignore.subprocess.run", side_effect=OSError("no git")):
            first = await self._content({"pattern": "needle"})
            second = await self._content({"pattern": "needle"})

        self.assertIn("git could not be consulted", first)
        self.assertNotIn("git could not be consulted", second)

    # ---------------------------------------------------------
    # GROUP 8: REGISTRATION
    # ---------------------------------------------------------

    def test_registry_binding(self):
        mock_registry = MagicMock()

        register_fallback_grep_tools(mock_registry, self.ctx)

        mock_registry.register.assert_called_once()
        call_kwargs = mock_registry.register.call_args[1]

        self.assertEqual(call_kwargs["name"], "Grep")
        self.assertEqual(call_kwargs["input_schema"]["required"], ["pattern"])
        # Grep must survive clone_readonly() so PLAN mode can still search.
        self.assertTrue(call_kwargs["is_readonly"])

    def test_schema_omits_unsupported_filters(self):
        """`type` needs ripgrep's type table, so it is not offered here."""
        mock_registry = MagicMock()

        register_fallback_grep_tools(mock_registry, self.ctx)
        properties = mock_registry.register.call_args[1]["input_schema"]["properties"]

        self.assertNotIn("type", properties)
        self.assertIn("glob", properties)

    def test_description_does_not_mention_the_missing_engine(self):
        """The model is told what this tool does, not what the machine lacks."""
        mock_registry = MagicMock()

        register_fallback_grep_tools(mock_registry, self.ctx)
        description = mock_registry.register.call_args[1]["description"]

        self.assertNotIn("ripgrep", description.lower())
        self.assertNotIn("fallback", description.lower())


if __name__ == "__main__":
    unittest.main()
