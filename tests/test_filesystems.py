import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the target module and its global state
import tools.filesystem as fs
from tools.filesystem import _read_impl, _write_impl, _edit_impl
from typedefs import ToolFailure


class TestFilesystemTools(unittest.IsolatedAsyncioTestCase):
    """
    Test Suite for Filesystem Tools (Read, Write, Edit)
    Covers edge cases, safeguards (Read-before-Write), string manipulation, 
    and output formatting.
    """

    def setUp(self):
        # Create a real temporary directory for safe file I/O testing
        self.test_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.test_dir.name)
        
        # VERY IMPORTANT: Reset the global state trackers before every test
        fs.known_content_files.clear()
        fs.stale_content_files.clear()

    def tearDown(self):
        self.test_dir.cleanup()

    # ---------------------------------------------------------
    # READ TOOL TESTS
    # ---------------------------------------------------------

    async def test_read_missing_args(self):
        result = await _read_impl({})
        self.assertIsInstance(result, ToolFailure)
        self.assertIn("file_path is required", result.error_message)

    async def test_read_not_found_with_heuristic(self):
        # Create a file named 'main.py'
        (self.base_path / "main.py").write_text("print('hello')")
        
        # Ask for 'main.txt' in the same directory to trigger the "wrong extension" heuristic
        # 'main.txt' and 'main.py' share the same stem: 'main'
        bad_path = str(self.base_path / "main.txt")
        result = await _read_impl({"file_path": bad_path})
        
        self.assertIsInstance(result, ToolFailure)
        self.assertIn("File does not exist", result.error_message)
        self.assertIn("Did you mean main.py?", result.error_message)  # Heuristic triggered

    async def test_read_success_and_formatting(self):
        file_path = self.base_path / "test.txt"
        file_path.write_text("line 1\nline 2\nline 3", encoding="utf-8")
        
        result = await _read_impl({"file_path": str(file_path)})
        
        self.assertIsInstance(result, str)
        self.assertIn("    1→line 1", result)
        self.assertIn("    2→line 2", result)
        self.assertIn("    3→line 3", result)
        
        # Verify it was added to the state tracker!
        self.assertIn(file_path.resolve(), fs.known_content_files)

    async def test_read_offset_out_of_bounds(self):
        file_path = self.base_path / "short.txt"
        file_path.write_text("only one line")
        
        result = await _read_impl({"file_path": str(file_path), "offset": 50})
        
        self.assertIsInstance(result, str)
        self.assertIn("Warning: the file only has 1 lines", result)

    async def test_read_size_and_token_limits(self):
        file_path = self.base_path / "huge.txt"
        
        # 1. Test MAX_FILE_BYTES limit
        with patch.object(fs, 'MAX_FILE_BYTES', 100):
            file_path.write_text("A" * 150)
            result1 = await _read_impl({"file_path": str(file_path)})
            self.assertIsInstance(result1, ToolFailure)
            self.assertIn("exceeds maximum allowed size", result1.error_message)
            
        # 2. Test MAX_TOKENS limit (1 token ~= 4 chars)
        with patch.object(fs, 'MAX_TOKENS', 10):
            file_path.write_text("A" * 50)  # ~12 tokens
            result2 = await _read_impl({"file_path": str(file_path)})
            self.assertIsInstance(result2, ToolFailure)
            self.assertIn("exceeds maximum allowed tokens", result2.error_message)


    # ---------------------------------------------------------
    # WRITE TOOL TESTS
    # ---------------------------------------------------------

    async def test_write_missing_args(self):
        res1 = await _write_impl({"content": "foo"})
        self.assertIsInstance(res1, ToolFailure)
        self.assertIn("file_path is required", res1.error_message)
        
        res2 = await _write_impl({"file_path": "foo.txt"})
        self.assertIsInstance(res2, ToolFailure)
        self.assertIn("content is required", res2.error_message)

    async def test_write_read_before_write_enforcement(self):
        file_path = self.base_path / "existing.txt"
        file_path.write_text("old content")
        
        # Try to write without reading first
        result = await _write_impl({"file_path": str(file_path), "content": "new content"})
        
        self.assertIsInstance(result, ToolFailure)
        self.assertIn("File has not been read yet", result.error_message)
        
        # Now read it, then write it
        await _read_impl({"file_path": str(file_path)})
        result2 = await _write_impl({"file_path": str(file_path), "content": "new content"})
        
        self.assertNotIsInstance(result2, ToolFailure)
        self.assertEqual(file_path.read_text(), "new content")

    async def test_write_new_file_success(self):
        file_path = self.base_path / "new.txt"
        
        result = await _write_impl({"file_path": str(file_path), "content": "brand new"})
        
        self.assertNotIsInstance(result, ToolFailure)
        self.assertIn("File created successfully", result)
        self.assertEqual(file_path.read_text(), "brand new")
        
        # Verify state tracker updated
        self.assertIn(file_path.resolve(), fs.known_content_files)


    # ---------------------------------------------------------
    # EDIT TOOL TESTS
    # ---------------------------------------------------------

    async def test_edit_missing_args(self):
        res1 = await _edit_impl({"old_string": "a", "new_string": "b"})
        self.assertIsInstance(res1, ToolFailure)
        self.assertIn("file_path is required", res1.error_message)
        
        res2 = await _edit_impl({"file_path": "a.txt", "old_string": "a"})
        self.assertIsInstance(res2, ToolFailure)
        self.assertIn("old_string and new_string are required", res2.error_message)

    async def test_edit_read_before_edit_enforcement(self):
        file_path = self.base_path / "code.py"
        file_path.write_text("x = 1")
        
        result = await _edit_impl({
            "file_path": str(file_path), "old_string": "x = 1", "new_string": "x = 2"
        })
        self.assertIsInstance(result, ToolFailure)
        self.assertIn("File has not been read yet", result.error_message)

    async def test_edit_fallback_to_write(self):
        # Target a file that does NOT exist yet
        file_path = self.base_path / "does_not_exist.py"
        
        # Provide an empty old_string (fallback trigger)
        result = await _edit_impl({
            "file_path": str(file_path), "old_string": "", "new_string": "def foo(): pass"
        })
        
        self.assertNotIsInstance(result, ToolFailure)
        self.assertIn("File created successfully", result)
        self.assertTrue(file_path.exists())
        self.assertEqual(file_path.read_text(), "def foo(): pass")

    async def test_edit_file_does_not_exist_no_fallback(self):
        file_path = self.base_path / "missing.txt"
        
        result = await _edit_impl({
            "file_path": str(file_path), "old_string": "something", "new_string": "else"
        })
        self.assertIsInstance(result, ToolFailure)
        self.assertIn("File does not exist", result.error_message)

    async def test_edit_exact_replacements(self):
        file_path = self.base_path / "target.txt"
        file_path.write_text("apple\nbanana\napple\ncherry")
        await _read_impl({"file_path": str(file_path)})  # Fulfill read requirement
        
        # Case 1: old == new
        res1 = await _edit_impl({
            "file_path": str(file_path), "old_string": "apple", "new_string": "apple"
        })
        self.assertIsInstance(res1, ToolFailure)
        self.assertIn("old_string and new_string are exactly the same", res1.error_message)
        
        # Case 2: old not found
        res2 = await _edit_impl({
            "file_path": str(file_path), "old_string": "grape", "new_string": "apple"
        })
        self.assertIsInstance(res2, ToolFailure)
        self.assertIn("String to replace not found", res2.error_message)

        # Case 3: ambiguous replacement (found twice, replace_all=False)
        res3 = await _edit_impl({
            "file_path": str(file_path), "old_string": "apple", "new_string": "orange"
        })
        self.assertIsInstance(res3, ToolFailure)
        self.assertIn("Found 2 matches", res3.error_message)
        self.assertIn("replace_all is false", res3.error_message)

        # Case 4: replace_all=True success
        res4 = await _edit_impl({
            "file_path": str(file_path), "old_string": "apple", "new_string": "orange", "replace_all": True
        })
        self.assertNotIsInstance(res4, ToolFailure)
        self.assertEqual(file_path.read_text(), "orange\nbanana\norange\ncherry")

        # Case 5: exact unique single replacement success
        res5 = await _edit_impl({
            "file_path": str(file_path), "old_string": "banana", "new_string": "mango", "replace_all": False
        })
        self.assertNotIsInstance(res5, ToolFailure)
        self.assertEqual(file_path.read_text(), "orange\nmango\norange\ncherry")
        
        # Verify formatted snippet response for single edit
        self.assertIn("Here's the result of running `cat -n`", res5)
        self.assertIn("mango", res5)

    async def test_edit_exception_handling(self):
        file_path = self.base_path / "fail.txt"
        file_path.write_text("data")
        await _read_impl({"file_path": str(file_path)})
        
        # Apply the patch ONLY during the tool execution, not during the file setup
        with patch("pathlib.Path.write_text", side_effect=PermissionError("Access denied")):
            result = await _edit_impl({
                "file_path": str(file_path), "old_string": "data", "new_string": "info"
            })
        
        self.assertIsInstance(result, ToolFailure)
        self.assertIn("Error writing to file", result.error_message)
        self.assertIn("Access denied", result.error_message)


if __name__ == "__main__":
    unittest.main()