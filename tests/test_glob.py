import os
import time
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Adjust import paths depending on your exact project structure
from sessioncontext import InvocationContext
import tools.filesearch as fs
from tools.filesearch import _glob_impl
from typedefs import ToolFailure


class TestGlobTool(unittest.IsolatedAsyncioTestCase):
    """
    Test Suite for the Filesearch (Glob) Tool.
    Covers brace expansion, recursive matching, mtime sorting, deduplication, 
    and proper error handling.
    """

    def setUp(self):
        # Create a real temporary directory for safe file I/O testing
        self.test_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.test_dir.name).resolve()

        self.ctx = InvocationContext(
            workspace=self.base_path,
            cwd=self.base_path,  # Or a subfolder if you want to test relative paths
            resume_file=None
        )

    def tearDown(self):
        self.test_dir.cleanup()

    def _create_file(self, relative_path: str, age_seconds: int = 0) -> Path:
        """Helper to create a file and optionally backdate its modification time."""        
        file_path = self.base_path / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(f"content of {relative_path}")
        
        if age_seconds > 0:
            past_time = time.time() - age_seconds
            os.utime(file_path, (past_time, past_time))
            
        return file_path.name

    # ---------------------------------------------------------
    # VALIDATION & ERROR HANDLING
    # ---------------------------------------------------------

    async def test_missing_pattern(self):
        result = await _glob_impl({"path": str(self.base_path)}, self.ctx)
        self.assertIsInstance(result, ToolFailure)
        self.assertIn("pattern is required", result.error_message)

    async def test_invalid_path(self):
        bad_path = self.base_path / "does_not_exist"
        result = await _glob_impl({"pattern": "*.txt", "path": str(bad_path)}, self.ctx)
        self.assertIsInstance(result, ToolFailure)
        self.assertIn("Directory does not exist", result.error_message)

    async def test_file_instead_of_directory_path(self):
        file_path = self._create_file("target.txt")
        result = await _glob_impl({"pattern": "*.txt", "path": str(file_path)}, self.ctx)
        self.assertIsInstance(result, ToolFailure)
        self.assertIn("is not a valid directory", result.error_message)

    async def test_no_files_found(self):
        self._create_file("script.py")
        result = await _glob_impl({"pattern": "*.txt", "path": str(self.base_path)}, self.ctx)
        self.assertIsInstance(result, str)
        self.assertEqual(result, "No files found.")

    # ---------------------------------------------------------
    # MATCHING & EXPANSION
    # ---------------------------------------------------------

    async def test_basic_glob(self):
        file1 = self._create_file("a.txt")
        file2 = self._create_file("b.txt")
        self._create_file("c.py")

        result = await _glob_impl({"pattern": "*.txt", "path": str(self.base_path)}, self.ctx)
        self.assertIsInstance(result, str)
        self.assertIn(str(file1), result)
        self.assertIn(str(file2), result)
        self.assertNotIn("c.py", result)

    async def test_recursive_glob(self):
        file1 = self._create_file("root.txt")
        file2 = self._create_file("nested/level1.txt")
        file3 = self._create_file("nested/deep/level2.txt")
        self._create_file("nested/deep/ignore.py")

        result = await _glob_impl({"pattern": "**/*.txt", "path": str(self.base_path)}, self.ctx)
        self.assertIsInstance(result, str)
        self.assertIn(str(file1), result)
        self.assertIn(str(file2), result)
        self.assertIn(str(file3), result)
        self.assertNotIn("ignore.py", result)

    async def test_brace_expansion(self):
        file_js = self._create_file("app.js")
        file_ts = self._create_file("app.ts")
        file_py = self._create_file("app.py")

        result = await _glob_impl({"pattern": "*.{js,ts}", "path": str(self.base_path)}, self.ctx)
        self.assertIsInstance(result, str)
        self.assertIn(str(file_js), result)
        self.assertIn(str(file_ts), result)
        self.assertNotIn(str(file_py), result)

    async def test_complex_brace_expansion(self):
        file1 = self._create_file("src/a/main.js")
        file2 = self._create_file("src/b/main.ts")
        self._create_file("src/c/main.js")

        result = await _glob_impl({"pattern": "src/{a,b}/*.{js,ts}", "path": str(self.base_path)}, self.ctx)
        self.assertIsInstance(result, str)
        self.assertIn(str(file1), result)
        self.assertIn(str(file2), result)
        self.assertNotIn("src/c/main.js", result)

    async def test_deduplication(self):
        # A pattern like {*,*.txt} could match a.txt twice.
        # The set() should deduplicate it.
        file1 = self._create_file("a.txt")
        
        result = await _glob_impl({"pattern": "{*,*.txt}", "path": str(self.base_path)}, self.ctx)
        self.assertIsInstance(result, str)
        
        # Count occurrences of the file path in the result string
        occurrences = result.split("\n").count(str(file1))
        self.assertEqual(occurrences, 1)

    # ---------------------------------------------------------
    # SORTING & TRUNCATION
    # ---------------------------------------------------------

    async def test_sorting_by_mtime(self):
        # Create files with specific ages
        oldest = self._create_file("3_oldest.txt", age_seconds=100)
        middle = self._create_file("2_middle.txt", age_seconds=50)
        newest = self._create_file("1_newest.txt", age_seconds=0)

        result = await _glob_impl({"pattern": "*.txt", "path": str(self.base_path)}, self.ctx)
        self.assertIsInstance(result, str)
        
        lines = result.split("\n")
        self.assertEqual(len(lines), 3)
        # Should be sorted newest to oldest
        self.assertEqual(lines[0], str(newest))
        self.assertEqual(lines[1], str(middle))
        self.assertEqual(lines[2], str(oldest))

    @patch("tools.filesearch.MAX_GLOB_RESULTS", 2)
    async def test_truncation(self):
        # Create 3 files, but limit is patched to 2
        newest1 = self._create_file("a.txt", age_seconds=10)
        newest2 = self._create_file("b.txt", age_seconds=20)
        oldest3 = self._create_file("c.txt", age_seconds=30)

        result = await _glob_impl({"pattern": "*.txt", "path": str(self.base_path)}, self.ctx)
        self.assertIsInstance(result, str)
        
        lines = result.split("\n")
        # 2 files + 1 truncation warning message = 3 lines
        self.assertEqual(len(lines), 3)
        self.assertEqual(lines[0], str(newest1))
        self.assertEqual(lines[1], str(newest2))
        self.assertNotIn(str(oldest3), result)
        self.assertIn("Results are truncated", lines[2])

    async def test_unstatable_file_graceful_handling(self):
        import inspect

        # 1. Create the files normally
        file1 = self._create_file("a.txt")
        file2 = self._create_file("b.txt")
        
        original_stat = Path.stat

        # 2. Define a selective stat that only raises an error when called from get_mtime
        def selective_stat(path_inst, *args, **kwargs):
            # Check the stack frames to see if this call originated inside "get_mtime"
            frame = inspect.currentframe()
            is_from_get_mtime = False
            while frame:
                if frame.f_code.co_name == "walk_iterative":
                    is_from_get_mtime = True
                    break
                frame = frame.f_back
            
            if is_from_get_mtime and path_inst.name in ("a.txt", "b.txt"):
                raise OSError("Permission denied")
                
            return original_stat(path_inst, *args, **kwargs)

        # 3. Patch Path.stat with our intelligent selective function
        with patch("pathlib.Path.stat", autospec=True, side_effect=selective_stat):
            result = await _glob_impl({"pattern": "*.txt", "path": str(self.base_path)}, self.ctx)
            
        self.assertIsInstance(result, str)
        
        # It shouldn't crash, both files should still be returned (falling back to mtime 0.0)
        self.assertIn(str(file1), result)
        self.assertIn(str(file2), result)


    # ---------------------------------------------------------
    # IGNORE & EXCLUDE STRATEGIES
    # ---------------------------------------------------------

    async def test_builtin_ignores(self):
        """Ensure built-in patterns like .git, __pycache__, venv, and *.pyc are ignored automatically."""
        # Create files that should be caught by BUILTIN_IGNORE_PATTERNS
        self._create_file(".git/config")
        self._create_file("__pycache__/app.cpython-310.pyc")
        self._create_file("venv/bin/activate")
        self._create_file("app.pyc")
        
        # Create a valid file
        valid_file = self._create_file("src/main.py")

        result = await _glob_impl({"pattern": "**/*", "path": str(self.base_path)}, self.ctx)
        
        self.assertIsInstance(result, str)
        self.assertIn(str(valid_file), result)
        
        # None of the ignored files should appear in the results
        self.assertNotIn(".git", result)
        self.assertNotIn("__pycache__", result)
        self.assertNotIn("venv", result)
        self.assertNotIn("app.pyc", result)

    async def test_prismaignore_file(self):
        """Ensure patterns inside <workspace>/.prismaignore are respected."""
        # Create the .prismaignore file in the workspace root
        ignore_file = self.base_path / ".prismaignore"
        ignore_file.write_text("secret.txt\nbuild/")

        self._create_file("secret.txt")
        self._create_file("build/output.bin")
        self._create_file("build/temp.log")
        valid_file = self._create_file("public.txt")

        result = await _glob_impl({"pattern": "**/*", "path": str(self.base_path)}, self.ctx)
        
        self.assertIsInstance(result, str)
        self.assertIn(str(valid_file), result)
        
        # The .prismaignore file itself might be found depending on pattern, 
        # but the filtered files should NOT be found
        self.assertNotIn("secret.txt", result)
        self.assertNotIn("build/output.bin", result)
        self.assertNotIn("build/temp.log", result)

    async def test_runtime_exclude_list(self):
        """Ensure the 'exclude' parameter works when passed as a list of strings."""
        self._create_file("node_modules/lib.js")
        self._create_file("temp_file.log")
        valid_file = self._create_file("index.js")

        result = await _glob_impl({
            "pattern": "**/*", 
            "path": str(self.base_path),
            "exclude": ["node_modules/", "*.log"]
        }, self.ctx)
        
        self.assertIsInstance(result, str)
        self.assertIn(str(valid_file), result)
        self.assertNotIn("node_modules", result)
        self.assertNotIn("temp_file.log", result)

    async def test_runtime_exclude_string_defensive_check(self):
        """Ensure the tool gracefully handles when the LLM passes a single string for 'exclude'."""
        self._create_file("ignored.bak")
        valid_file = self._create_file("kept.txt")

        # Intentionally passing a string instead of a list
        result = await _glob_impl({
            "pattern": "**/*", 
            "path": str(self.base_path),
            "exclude": "*.bak"
        }, self.ctx)
        
        self.assertIsInstance(result, str)
        self.assertIn(str(valid_file), result)
        self.assertNotIn("ignored.bak", result)

    async def test_directory_ignore_prevents_deep_traversal(self):
        """Ensure that if a directory is ignored, its children are not traversed or returned."""
        self._create_file("vendor/lib1/core.py")
        self._create_file("vendor/lib2/utils.py")
        valid_file = self._create_file("app/main.py")

        # Exclude the vendor directory entirely
        result = await _glob_impl({
            "pattern": "**/*.py", 
            "path": str(self.base_path),
            "exclude": ["vendor/"]
        }, self.ctx)
        
        self.assertIsInstance(result, str)
        self.assertIn(str(valid_file), result)
        self.assertNotIn("vendor/lib1/core.py", result)
        self.assertNotIn("vendor/lib2/utils.py", result)
        
    async def test_out_of_workspace_fallback(self):
        """Ensure paths entirely outside the workspace gracefully fall back to not being ignored."""
        # Create a temp dir outside of our current workspace context
        with tempfile.TemporaryDirectory() as external_dir:
            ext_path = Path(external_dir).resolve()
            
            # Create a file inside this external directory
            ext_file = ext_path / "external.txt"
            ext_file.write_text("data")
            
            # Create a directory that WOULD be ignored if it were in the workspace (.git)
            git_dir = ext_path / ".git"
            git_dir.mkdir()
            git_file = git_dir / "config"
            git_file.write_text("core")

            # Run glob on the external directory
            result = await _glob_impl({
                "pattern": "**/*",
                "path": str(ext_path)
            }, self.ctx)
            
            self.assertIsInstance(result, str)
            # Because it is out of the workspace, IgnoreMatcher throws ValueError internally 
            # and gracefully defaults to returning False (i.e. DO NOT ignore).
            self.assertIn("external.txt", result)
            
            # The out-of-workspace .git folder is actually scanned because it bypassed the workspace IgnoreMatcher
            self.assertIn(os.path.join(".git", "config").replace("\\", "/"), result.replace("\\", "/"))


    # ---------------------------------------------------------
    # EDGE CASES (Symlinks, Dotfiles, Dir vs File)
    # ---------------------------------------------------------

    async def test_symlinks_are_skipped(self):
        """Ensure symlinks to files and directories are completely skipped by glob."""
        target_file = self._create_file("real_target.txt")
        target_dir = self.base_path / "real_dir"
        target_dir.mkdir()
        (target_dir / "child.txt").write_text("child")
        
        # Create symlinks
        symlink_file = self.base_path / "sym_file.txt"
        symlink_dir = self.base_path / "sym_dir"
        
        try:
            os.symlink(self.base_path / "real_target.txt", symlink_file)
            os.symlink(target_dir, symlink_dir)
        except OSError:
            # On Windows, creating symlinks requires admin privileges or Developer Mode.
            # If it fails, skip the test gracefully rather than failing the suite.
            self.skipTest("OS does not permit creating symlinks without elevated privileges")

        result = await _glob_impl({"pattern": "**/*", "path": str(self.base_path)}, self.ctx)
        
        self.assertIsInstance(result, str)
        # Real files should be found
        self.assertIn("real_target.txt", result)
        self.assertIn(os.path.join("real_dir", "child.txt").replace("\\", "/"), result.replace("\\", "/"))
        
        # Symlinks should be strictly ignored
        self.assertNotIn("sym_file.txt", result)
        self.assertNotIn("sym_dir", result)

    async def test_dotglob_enabled(self):
        """Ensure hidden files/folders (starting with .) are found because DOTGLOB is enabled."""
        # Create hidden file and a file inside a hidden directory
        self._create_file(".env")
        self._create_file(".config/settings.json")
        
        result = await _glob_impl({"pattern": "**/*", "path": str(self.base_path)}, self.ctx)
        
        self.assertIsInstance(result, str)
        self.assertIn(".env", result)
        self.assertIn(os.path.join(".config", "settings.json").replace("\\", "/"), result.replace("\\", "/"))

    async def test_directories_are_not_returned_as_matches(self):
        """Ensure that if a pattern matches a directory name, the directory itself isn't returned."""
        # Create a directory named "test_dir" and a file named "test_file.txt"
        dir_path = self.base_path / "test_dir"
        dir_path.mkdir()
        
        self._create_file("test_file.txt")

        # Pattern matches anything starting with "test_"
        result = await _glob_impl({"pattern": "test_*", "path": str(self.base_path)}, self.ctx)
        
        self.assertIsInstance(result, str)
        self.assertIn("test_file.txt", result)
        # The directory should not be in the output, because entry.is_file() is checked
        self.assertNotIn("test_dir\n", result + "\n")
        
        # Another explicit test: ask for the exact directory name
        result_exact = await _glob_impl({"pattern": "test_dir", "path": str(self.base_path)}, self.ctx)
        self.assertEqual(result_exact, "No files found.")

    # ---------------------------------------------------------
    # STANDARD GLOB FEATURES (?, []) - NO EXTRA FLAGS NEEDED
    # ---------------------------------------------------------

    async def test_single_character_wildcard(self):
        """Ensure '?' matches exactly one character (standard glob behavior)."""
        self._create_file("script_a.py")
        self._create_file("script_b.py")
        self._create_file("script_12.py") # Two chars after _, should not match

        result = await _glob_impl({"pattern": "script_?.py", "path": str(self.base_path)}, self.ctx)
        
        self.assertIsInstance(result, str)
        self.assertIn("script_a.py", result)
        self.assertIn("script_b.py", result)
        self.assertNotIn("script_12.py", result)

    async def test_character_classes(self):
        """Ensure '[a-z]' and '[0-9]' matching works correctly (standard glob behavior)."""
        self._create_file("image_01.png")
        self._create_file("image_02.png")
        self._create_file("image_xx.png")

        result = await _glob_impl({"pattern": "image_[0-9][0-9].png", "path": str(self.base_path)}, self.ctx)
        
        self.assertIsInstance(result, str)
        self.assertIn("image_01.png", result)
        self.assertIn("image_02.png", result)
        self.assertNotIn("image_xx.png", result)

    async def test_character_class_negation(self):
        """Ensure '[!a-z]' negation works (standard glob behavior)."""
        self._create_file("item_1.txt")
        self._create_file("item_a.txt")

        # [!a-z] means any character that is NOT a lowercase letter
        result = await _glob_impl({"pattern": "item_[!a-z].txt", "path": str(self.base_path)}, self.ctx)
        
        self.assertIsInstance(result, str)
        self.assertIn("item_1.txt", result)
        self.assertNotIn("item_a.txt", result)

if __name__ == "__main__":
    unittest.main()