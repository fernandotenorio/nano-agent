import os
import time
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Adjust import paths depending on your exact project structure
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
        result = await _glob_impl({"path": str(self.base_path)})
        self.assertIsInstance(result, ToolFailure)
        self.assertIn("pattern is required", result.error_message)

    async def test_invalid_path(self):
        bad_path = self.base_path / "does_not_exist"
        result = await _glob_impl({"pattern": "*.txt", "path": str(bad_path)})
        self.assertIsInstance(result, ToolFailure)
        self.assertIn("Directory does not exist", result.error_message)

    async def test_file_instead_of_directory_path(self):
        file_path = self._create_file("target.txt")
        result = await _glob_impl({"pattern": "*.txt", "path": str(file_path)})
        self.assertIsInstance(result, ToolFailure)
        self.assertIn("is not a valid directory", result.error_message)

    async def test_no_files_found(self):
        self._create_file("script.py")
        result = await _glob_impl({"pattern": "*.txt", "path": str(self.base_path)})
        self.assertIsInstance(result, str)
        self.assertEqual(result, "No files found.")

    # ---------------------------------------------------------
    # MATCHING & EXPANSION
    # ---------------------------------------------------------

    async def test_basic_glob(self):
        file1 = self._create_file("a.txt")
        file2 = self._create_file("b.txt")
        self._create_file("c.py")

        result = await _glob_impl({"pattern": "*.txt", "path": str(self.base_path)})
        self.assertIsInstance(result, str)
        self.assertIn(str(file1), result)
        self.assertIn(str(file2), result)
        self.assertNotIn("c.py", result)

    async def test_recursive_glob(self):
        file1 = self._create_file("root.txt")
        file2 = self._create_file("nested/level1.txt")
        file3 = self._create_file("nested/deep/level2.txt")
        self._create_file("nested/deep/ignore.py")

        result = await _glob_impl({"pattern": "**/*.txt", "path": str(self.base_path)})
        self.assertIsInstance(result, str)
        self.assertIn(str(file1), result)
        self.assertIn(str(file2), result)
        self.assertIn(str(file3), result)
        self.assertNotIn("ignore.py", result)

    async def test_brace_expansion(self):
        file_js = self._create_file("app.js")
        file_ts = self._create_file("app.ts")
        file_py = self._create_file("app.py")

        result = await _glob_impl({"pattern": "*.{js,ts}", "path": str(self.base_path)})
        self.assertIsInstance(result, str)
        self.assertIn(str(file_js), result)
        self.assertIn(str(file_ts), result)
        self.assertNotIn(str(file_py), result)

    async def test_complex_brace_expansion(self):
        file1 = self._create_file("src/a/main.js")
        file2 = self._create_file("src/b/main.ts")
        self._create_file("src/c/main.js")

        result = await _glob_impl({"pattern": "src/{a,b}/*.{js,ts}", "path": str(self.base_path)})
        self.assertIsInstance(result, str)
        self.assertIn(str(file1), result)
        self.assertIn(str(file2), result)
        self.assertNotIn("src/c/main.js", result)

    async def test_deduplication(self):
        # A pattern like {*,*.txt} could match a.txt twice.
        # The set() should deduplicate it.
        file1 = self._create_file("a.txt")
        
        result = await _glob_impl({"pattern": "{*,*.txt}", "path": str(self.base_path)})
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

        result = await _glob_impl({"pattern": "*.txt", "path": str(self.base_path)})
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

        result = await _glob_impl({"pattern": "*.txt", "path": str(self.base_path)})
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
                if frame.f_code.co_name == "get_mtime":
                    is_from_get_mtime = True
                    break
                frame = frame.f_back
            
            if is_from_get_mtime and path_inst.name in ("a.txt", "b.txt"):
                raise OSError("Permission denied")
                
            return original_stat(path_inst, *args, **kwargs)

        # 3. Patch Path.stat with our intelligent selective function
        with patch("pathlib.Path.stat", autospec=True, side_effect=selective_stat):
            result = await _glob_impl({"pattern": "*.txt", "path": str(self.base_path)})
            
        self.assertIsInstance(result, str)
        
        # It shouldn't crash, both files should still be returned (falling back to mtime 0.0)
        self.assertIn(str(file1), result)
        self.assertIn(str(file2), result)


if __name__ == "__main__":
    unittest.main()