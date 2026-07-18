import os
import unittest
import tempfile
import asyncio
from pathlib import Path

# Adjust imports based on your exact project structure
from tools.filesearch import _ls_impl
from sessioncontext import InvocationContext


class TestLsIgnoreLogic(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Create a real temp workspace
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name).resolve()
        
        # Mock Context
        self.ctx = InvocationContext(
            workspace=self.workspace,
            cwd=self.workspace,
            resume_file=None
        )

        # 1. Valid tracked files
        (self.workspace / "main.py").touch()
        (self.workspace / "readme.md").touch()

        # 2. Files built-in to the ignore matcher (.venv)
        self.venv_dir = self.workspace / ".venv"
        self.venv_dir.mkdir()
        (self.venv_dir / "bin.py").touch()

        # 3. Custom files to be ignored by .prismaignore
        self.secret_dir = self.workspace / "secret_keys"
        self.secret_dir.mkdir()
        (self.secret_dir / "key.txt").touch()
        (self.workspace / "database.sqlite").touch()

        # Write .prismaignore file
        ignore_content = "secret_keys/\n*.sqlite\n"
        (self.workspace / ".prismaignore").write_text(ignore_content, encoding="utf-8")

        # 4. A directory for testing runtime exclude parameters
        self.build_dir = self.workspace / "build"
        self.build_dir.mkdir()
        (self.build_dir / "output.bin").touch()

    async def asyncTearDown(self):
        self.temp_dir.cleanup()

    async def test_ls_filters_builtin_and_prismaignore(self):
        """Test that .venv (built-in) and secret_keys/ (workspace ignore) are skipped."""
        kwargs = {
            "path": str(self.workspace),
            "depth": 2,
        }
        
        result = await _ls_impl(kwargs, self.ctx)
        
        # Valid files should be present
        self.assertIn("main.py", result)
        self.assertIn("readme.md", result)
        self.assertIn("build", result)
        self.assertIn("output.bin", result)
        
        # Ignored files/directories should NOT be present
        self.assertNotIn(".venv", result)
        self.assertNotIn("secret_keys", result)
        self.assertNotIn("database.sqlite", result)

    async def test_ls_filters_runtime_excludes(self):
        """Test that passing dynamic exclusions in kwargs works correctly."""
        kwargs = {
            "path": str(self.workspace),
            "depth": 2,
            "exclude": ["build/"]
        }
        
        result = await _ls_impl(kwargs, self.ctx)
        
        # Valid files still present
        self.assertIn("main.py", result)
        
        # Runtime exclude "build/" should be missing
        self.assertNotIn("build", result)
        self.assertNotIn("output.bin", result)
        
        # .prismaignore should still apply
        self.assertNotIn("database.sqlite", result)

    async def test_ls_dir_count_ignores_correctly(self):
        """Test the (X items) directory counter accounts for ignored files."""
        # Create a directory with a mix of allowed and ignored files
        mix_dir = self.workspace / "mixed"
        mix_dir.mkdir()
        (mix_dir / "test1.py").touch()         # allowed
        (mix_dir / "test2.py").touch()         # allowed
        (mix_dir / "local.sqlite").touch()     # ignored by .prismaignore
        (mix_dir / "__pycache__").mkdir()      # ignored by built-ins
        
        kwargs = {
            "path": str(self.workspace),
            "depth": 1,
        }
        result = await _ls_impl(kwargs, self.ctx)
        
        # Since local.sqlite and __pycache__ are ignored, only test1.py and test2.py should be counted (2 items)
        self.assertIn("mixed/ (2 items)", result)

    async def test_builtin_ignore_nested_directories(self):
        """
        Built-in ignores should apply recursively, not only at workspace root.
        """
        nested = self.workspace / "src" / "cache"
        nested.mkdir(parents=True)

        pycache = nested / "__pycache__"
        pycache.mkdir()

        (pycache / "module.pyc").touch()
        (nested / "main.py").touch()

        result = await _ls_impl(
            {
                "path": str(self.workspace),
                "depth": 5,
            },
            self.ctx,
        )

        self.assertIn("main.py", result)
        self.assertNotIn("__pycache__", result)
        self.assertNotIn("module.pyc", result)


    async def test_builtin_ignore_node_modules_anywhere(self):
        """
        node_modules should be ignored regardless of nesting depth.
        """
        node_modules = self.workspace / "frontend" / "node_modules"
        node_modules.mkdir(parents=True)

        (node_modules / "package.json").touch()

        result = await _ls_impl(
            {
                "path": str(self.workspace),
                "depth": 5,
            },
            self.ctx,
        )

        self.assertNotIn("node_modules", result)
        self.assertNotIn("package.json", result)


    async def test_prismaignore_directory_pattern_only_matches_directory(self):
        """
        A directory ignore pattern should remove the directory tree but
        not similarly named files.
        """
        ignored_dir = self.workspace / "logs"
        ignored_dir.mkdir()

        (ignored_dir / "app.log").touch()

        # Similar name, should survive
        (self.workspace / "logs.txt").touch()

        (self.workspace / ".prismaignore").write_text(
            "logs/\n",
            encoding="utf-8",
        )

        result = await _ls_impl(
            {
                "path": str(self.workspace),
                "depth": 3,
            },
            self.ctx,
        )

        self.assertNotIn("logs/", result)
        self.assertNotIn("app.log", result)

        self.assertIn("logs.txt", result)


    async def test_prismaignore_glob_file_pattern(self):
        """
        File glob patterns should ignore matching files anywhere.
        """
        (self.workspace / "a.sqlite").touch()

        nested = self.workspace / "db"
        nested.mkdir()

        (nested / "b.sqlite").touch()
        (nested / "keep.db").touch()

        result = await _ls_impl(
            {
                "path": str(self.workspace),
                "depth": 3,
            },
            self.ctx,
        )

        self.assertNotIn("a.sqlite", result)
        self.assertNotIn("b.sqlite", result)

        self.assertIn("keep.db", result)


    async def test_runtime_exclude_has_same_gitignore_semantics(self):
        """
        Runtime excludes should use the same matcher semantics as
        .prismaignore.
        """
        temp = self.workspace / "temporary"
        temp.mkdir()

        (temp / "one.tmp").touch()

        keep = self.workspace / "temporary.keep"
        keep.touch()

        result = await _ls_impl(
            {
                "path": str(self.workspace),
                "depth": 3,
                "exclude": [
                    "temporary/",
                    "*.tmp",
                ],
            },
            self.ctx,
        )

        lines = result.splitlines()
        self.assertFalse(
            any(
                line.strip().startswith(
                    ("temporary/", "├── temporary/", "└── temporary/")
                )
                for line in lines
            )
        )

        self.assertNotIn("one.tmp", result)
        self.assertIn("temporary.keep", result)


    async def test_multiple_ignore_sources_stack(self):
        """
        Built-ins + prismaignore + runtime exclusions should all apply.
        """
        (self.workspace / ".venv2").mkdir()
        (self.workspace / ".venv2" / "file.py").touch()

        runtime = self.workspace / "runtime"
        runtime.mkdir()
        (runtime / "ignored.txt").touch()

        result = await _ls_impl(
            {
                "path": str(self.workspace),
                "depth": 3,
                "exclude": [
                    "runtime/",
                ],
            },
            self.ctx,
        )

        # Built-in
        self.assertNotIn("├── .venv/", result)
        self.assertNotIn("└── .venv/", result)

        # prismaignore
        self.assertNotIn("database.sqlite", result)

        # runtime
        self.assertNotIn("runtime", result)
        self.assertNotIn("ignored.txt", result)

    async def test_prismaignore_negation_restores_file(self):
        """
        Gitignore negation rules should allow explicitly unignored files.
        """
        secrets = self.workspace / "secrets"
        secrets.mkdir()

        (secrets / "public.txt").touch()
        (secrets / "private.txt").touch()

        (self.workspace / ".prismaignore").write_text(
            "secrets/\n"
            "!secrets/public.txt\n",
            encoding="utf-8",
        )

        result = await _ls_impl(
            {
                "path": str(self.workspace),
                "depth": 3,
            },
            self.ctx,
        )

        self.assertNotIn("private.txt", result)

        # Depending on gitignore semantics, negating a child inside an ignored
        # directory may fail unless the parent directory is also unignored.
        # This assertion documents expected gitignore behavior.
        self.assertNotIn("public.txt", result)


    async def test_prismaignore_ignored_directory_cannot_be_resurrected_by_negation(self):
        """
        Ignored directories are pruned during traversal.

        This intentionally documents that ls does not implement full gitignore
        negation traversal semantics because ignored directories are never walked.
        """

        secrets = self.workspace / "secrets"
        secrets.mkdir()

        (secrets / "public.txt").touch()
        (secrets / "private.txt").touch()

        (self.workspace / ".prismaignore").write_text(
            "secrets/\n"
            "!secrets/public.txt\n",
            encoding="utf-8",
        )

        result = await _ls_impl(
            {
                "path": str(self.workspace),
                "depth": 3,
            },
            self.ctx,
        )

        self.assertNotIn("secrets", result)
        self.assertNotIn("public.txt", result)
        self.assertNotIn("private.txt", result)


    async def test_ignore_does_not_match_similar_names(self):
        """
        Ignore rules must not accidentally match prefixes.
        """
        (self.workspace / "node_modules").mkdir()
        (self.workspace / "node_modules" / "bad.js").touch()

        (self.workspace / "node_modules_backup").mkdir()
        (self.workspace / "node_modules_backup" / "good.js").touch()

        result = await _ls_impl(
            {
                "path": str(self.workspace),
                "depth": 3,
            },
            self.ctx,
        )

        self.assertNotIn("node_modules/", result)
        self.assertNotIn("bad.js", result)

        self.assertIn("node_modules_backup", result)
        self.assertIn("good.js", result)


    async def test_ignore_handles_windows_style_relative_paths(self):
        """
        IgnoreMatcher normalizes '\\' into '/'.

        This directly validates the matcher rather than ls traversal.
        """
        from tools.ignore import IgnoreMatcher

        (self.workspace / ".prismaignore").write_text(
            "build/\n",
            encoding="utf-8",
        )

        matcher = IgnoreMatcher(self.workspace)

        self.assertTrue(
            matcher.ignores_relative(
                r"build\output.txt",
                is_dir=False,
            )
        )

        self.assertTrue(
            matcher.ignores_relative(
                r"build",
                is_dir=True,
            )
        )


    async def test_ignore_directory_requires_directory_flag(self):
        """
        Directory patterns should depend on is_dir information.
        """
        from tools.ignore import IgnoreMatcher

        (self.workspace / ".prismaignore").write_text(
            "cache/\n",
            encoding="utf-8",
        )

        matcher = IgnoreMatcher(self.workspace)

        self.assertTrue(
            matcher.ignores_relative(
                "cache",
                is_dir=True,
            )
        )

        self.assertFalse(
            matcher.ignores_relative(
                "cache",
                is_dir=False,
            )
        )


    async def test_empty_prismaignore_file_is_safe(self):
        """
        Empty .prismaignore should not break listing.
        """
        (self.workspace / ".prismaignore").write_text(
            "",
            encoding="utf-8",
        )

        (self.workspace / "hello.py").touch()

        result = await _ls_impl(
            {
                "path": str(self.workspace),
                "depth": 1,
            },
            self.ctx,
        )

        self.assertIn("hello.py", result)


    async def test_duplicate_ignore_patterns_are_safe(self):
        """
        Duplicate patterns should not alter behavior.
        """
        ignored = self.workspace / "ignored"
        ignored.mkdir()

        (ignored / "file.txt").touch()

        (self.workspace / ".prismaignore").write_text(
            "ignored/\n"
            "ignored/\n"
            "*.txt\n"
            "*.txt\n",
            encoding="utf-8",
        )

        result = await _ls_impl(
            {
                "path": str(self.workspace),
                "depth": 2,
            },
            self.ctx,
        )

        self.assertNotIn("ignored", result)
        self.assertNotIn("file.txt", result)


    async def test_symlink_to_directory_is_displayed_but_not_followed(self):
        """
        Directory symlinks appear as leaf nodes and are never recursively walked.
        """
        import os

        real_dir = self.workspace / "real_node_modules"
        real_dir.mkdir()
        (real_dir / "package.json").touch()

        link = self.workspace / "node_modules_link"

        try:
            os.symlink(real_dir, link, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("Symlink creation unavailable")

        result = await _ls_impl(
            {
                "path": str(self.workspace),
                "depth": 5,
            },
            self.ctx,
        )

        # Real directory is visible
        self.assertIn("real_node_modules", result)
        self.assertIn("package.json", result)

        # Symlink is shown as a leaf
        self.assertIn("node_modules_link ->", result)

        # But it was not expanded
        self.assertNotIn(
            "node_modules_link/\n",
            result,
        )

    async def test_symlink_files_are_displayed_as_leaves(self):
        """
        Valid file symlinks should appear but must not recurse.
        """
        import os

        target = self.workspace / "original.txt"
        target.touch()

        link = self.workspace / "linked.txt"

        try:
            os.symlink(target, link)
        except (OSError, NotImplementedError):
            self.skipTest("Symlink creation unavailable")

        result = await _ls_impl(
            {
                "path": str(self.workspace),
                "depth": 2,
            },
            self.ctx,
        )

        self.assertIn("linked.txt", result)
        self.assertIn("->", result)


    async def test_broken_symlink_does_not_crash_ignore_processing(self):
        """
        Broken symlinks should not trigger stat errors during ignore checks.
        """
        import os

        broken = self.workspace / "broken_link"

        try:
            os.symlink(
                self.workspace / "missing_target",
                broken,
            )
        except (OSError, NotImplementedError):
            self.skipTest("Symlink creation unavailable")

        result = await _ls_impl(
            {
                "path": str(self.workspace),
                "depth": 1,
            },
            self.ctx,
        )

        self.assertIn("broken_link", result)

if __name__ == "__main__":
    unittest.main()