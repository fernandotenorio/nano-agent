import os
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

# Adjust import paths depending on your exact project structure
from tools.filesearch import _ls_impl
from typedefs import ToolFailure
from sessioncontext import InvocationContext

# gemini 3.1-pro
class TestLsTool(unittest.IsolatedAsyncioTestCase):
    """
    Test Suite for the ls Tool.
    Covers formatting, sorting, depth control, ignore patterns, symlinks, 
    and OS-level permission error handling.
    """

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.test_dir.name).resolve()

        self.ctx = InvocationContext(
            workspace=self.base_path,
            cwd=self.base_path,  # Or a subfolder if you want to test relative paths
            resume_file=None
        )

    def tearDown(self):
        self.test_dir.cleanup()

    def _create_file(self, relative_path: str, content: str = "") -> Path:
        """Helper to create a file."""
        file_path = self.base_path / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return file_path

    def _create_dir(self, relative_path: str) -> Path:
        """Helper to create a directory."""
        dir_path = self.base_path / relative_path
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path

    def _create_symlink(self, link_relative: str, target_relative: str) -> Path:
        """Helper to create a symlink, skipping the test if OS doesn't support it."""
        link_path = self.base_path / link_relative
        target_path = self.base_path / target_relative
        link_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            link_path.symlink_to(target_path)
            return link_path
        except OSError:
            self.skipTest("Symlinks are not supported or permitted in this OS environment.")

    # ---------------------------------------------------------
    # 1. VALIDATION & ERROR HANDLING
    # ---------------------------------------------------------

    async def test_nonexistent_path(self):
        bad_path = self.base_path / "nowhere"
        result = await _ls_impl({"path": str(bad_path)}, self.ctx)
        self.assertIsInstance(result, ToolFailure)
        self.assertIn("Path does not exist", result.error_message)

    async def test_invalid_depth_type(self):
        self._create_file("test.txt")
        # Pass a bad depth string; it should gracefully fallback to depth=1
        result = await _ls_impl({"path": str(self.base_path), "depth": "not_an_int"}, self.ctx)
        self.assertIsInstance(result, str)
        self.assertIn("test.txt", result)

    async def test_invalid_ignore_type(self):
        self._create_file("test.txt")
        # Pass garbage into the ignore list
        result = await _ls_impl({"path": str(self.base_path), "exclude": [123, None, {"bad": "type"}]}, self.ctx)
        self.assertIsInstance(result, str)
        self.assertIn("test.txt", result)

    # ---------------------------------------------------------
    # 2. BASIC LISTING, SORTING & FORMATTING
    # ---------------------------------------------------------

    async def test_empty_directory(self):
        result = await _ls_impl({"path": str(self.base_path)}, self.ctx)
        self.assertIsInstance(result, str)
        self.assertIn("[Empty Directory]", result)

    async def test_sorting_and_counts(self):
        # Create directories
        self._create_file("B_dir/file1.txt")
        self._create_file("a_dir/file1.txt")
        self._create_file("a_dir/file2.txt")
        # Create files
        self._create_file("Z.txt")
        self._create_file("c.txt")

        result = await _ls_impl({"path": str(self.base_path)}, self.ctx)
        self.assertIsInstance(result, str)
        
        lines = result.split("\n")
        # Find where items appear in the output
        idx_a_dir = next(i for i, line in enumerate(lines) if "a_dir/" in line)
        idx_B_dir = next(i for i, line in enumerate(lines) if "B_dir/" in line)
        idx_c_txt = next(i for i, line in enumerate(lines) if "c.txt" in line)
        idx_Z_txt = next(i for i, line in enumerate(lines) if "Z.txt" in line)

        # Verify sorting: Dirs first, then files, both case-insensitive alphabetically
        self.assertTrue(idx_a_dir < idx_B_dir < idx_c_txt < idx_Z_txt)
        
        # Verify counts
        self.assertIn("a_dir/ (2 items)", result)
        self.assertIn("B_dir/ (1 item)", result)

    async def test_target_is_file(self):
        file_path = self._create_file("target.txt")
        result = await _ls_impl({"path": str(file_path)}, self.ctx)
        self.assertIsInstance(result, str)
        # self.assertIn("Listing:", result)
        self.assertIn("└── target.txt", result)

    # ---------------------------------------------------------
    # 3. DEPTH CONTROL
    # ---------------------------------------------------------

    async def test_depth_zero(self):
        self._create_file("child.txt")
        result = await _ls_impl({"path": str(self.base_path), "depth": 0}, self.ctx)
        self.assertIsInstance(result, str)
        self.assertIn("(depth limit reached)", result)
        self.assertNotIn("child.txt", result)

    async def test_depth_one_default(self):
        self._create_file("child/grandchild.txt")
        result = await _ls_impl({"path": str(self.base_path)}, self.ctx)
        self.assertIsInstance(result, str)
        self.assertIn("child/ (1 item)", result)
        self.assertNotIn("grandchild.txt", result) # Should not recurse to 2nd level

    async def test_depth_unlimited(self):
        self._create_file("child/grandchild/great_grandchild.txt")
        result = await _ls_impl({"path": str(self.base_path), "depth": -1}, self.ctx)
        self.assertIsInstance(result, str)
        self.assertIn("child/ (1 item)", result)
        self.assertIn("grandchild/ (1 item)", result)
        self.assertIn("great_grandchild.txt", result)

    # ---------------------------------------------------------
    # 4. IGNORE LOGIC
    # ---------------------------------------------------------

    async def test_default_ignore(self):
        self._create_file(".git/config")
        self._create_file("node_modules/lib.js")
        self._create_file("__pycache__/app.pyc")
        self._create_file("normal.txt")

        result = await _ls_impl({"path": str(self.base_path)}, self.ctx)
        self.assertIsInstance(result, str)
        self.assertIn("normal.txt", result)
        self.assertNotIn(".git", result)
        self.assertNotIn("node_modules", result)
        self.assertNotIn("__pycache__", result)

    async def test_custom_ignore(self):
        self._create_file("app.log")
        self._create_file("temp_file.txt")
        self._create_file("keep.txt")
        self._create_file(".git/config") # Should still ignore defaults

        result = await _ls_impl({
            "path": str(self.base_path), 
            "exclude": ["*.log", "temp_*"]
        }, self.ctx)
        self.assertIsInstance(result, str)
        self.assertIn("keep.txt", result)
        self.assertNotIn("app.log", result)
        self.assertNotIn("temp_file.txt", result)
        self.assertNotIn(".git", result)

    # ---------------------------------------------------------
    # 5. SYMLINKS & EDGE CASES
    # ---------------------------------------------------------

    async def test_target_is_symlink(self):
        self._create_file("real_file.txt")
        symlink = self._create_symlink("link.txt", "real_file.txt")
        
        result = await _ls_impl({"path": str(symlink)}, self.ctx)
        self.assertIsInstance(result, str)
        self.assertIn("link.txt ->", result)
        self.assertIn("real_file.txt", result)

    async def test_symlinks_in_directory(self):
        self._create_file("data/real_file.txt")
        self._create_dir("real_dir")
        
        self._create_symlink("link_to_file.txt", "data/real_file.txt")
        self._create_symlink("link_to_dir", "real_dir")

        result = await _ls_impl({"path": str(self.base_path)}, self.ctx)
        self.assertIsInstance(result, str)
        
        # Symlinks should both be treated as files (no trailing slash, has arrow)
        self.assertIn("link_to_file.txt ->", result)
        self.assertIn("link_to_dir ->", result)
        # Should not format it as link_to_dir/
        self.assertNotIn("link_to_dir/", result)

    async def test_broken_symlink(self):
        # Target does not exist
        self._create_symlink("broken_link.txt", "does_not_exist.txt")
        
        result = await _ls_impl({"path": str(self.base_path)}, self.ctx)
        self.assertIsInstance(result, str)
        self.assertIn("broken_link.txt ->", result)
        self.assertNotIn("ToolFailure", str(type(result)))

    # ---------------------------------------------------------
    # 6. OS & PERMISSION ERRORS (MOCKED)
    # ---------------------------------------------------------

    async def test_unreadable_directory(self):
        self._create_dir("restricted_dir")
        
        original_iterdir = Path.iterdir

        def selective_iterdir(path_inst):
            if path_inst.name == "restricted_dir":
                raise OSError(13, "Permission denied")
            return original_iterdir(path_inst)

        with patch("pathlib.Path.iterdir", autospec=True, side_effect=selective_iterdir):
            # Tell it to recurse to hit the restricted directory's iterdir
            result = await _ls_impl({"path": str(self.base_path), "depth": 2}, self.ctx)

        self.assertIsInstance(result, str)
        self.assertIn("restricted_dir/", result)
        self.assertIn("[Unable to read: Permission denied]", result)

    async def test_unstatable_item(self):
        self._create_file("locked_file.txt")
        self._create_file("normal_file.txt")

        original_is_symlink = Path.is_symlink

        def selective_is_symlink(path_inst, *args, **kwargs):
            if path_inst.name == "locked_file.txt":
                raise OSError("Permission denied")
            return original_is_symlink(path_inst, *args, **kwargs)

        # Patch `is_symlink` since it's the first stat check in `generate_tree`
        with patch("pathlib.Path.is_symlink", autospec=True, side_effect=selective_is_symlink):
            result = await _ls_impl({"path": str(self.base_path)}, self.ctx)

        self.assertIsInstance(result, str)
        self.assertIn("normal_file.txt", result)
        self.assertIn("locked_file.txt [unreadable]", result)

    # ---------------------------------------------------------
    # 7. TRUNCATION
    # ---------------------------------------------------------

    @patch("tools.filesearch.MAX_LS_ENTRIES", 3)
    async def test_truncation(self):
        # Create 5 files
        for i in range(5):
            self._create_file(f"file_{i}.txt")
            
        result = await _ls_impl({"path": str(self.base_path)}, self.ctx)
        self.assertIsInstance(result, str)
        
        lines = result.split("\n")
        
        # 1 line for "Listing: ...", 3 lines for the files, 1 empty line (due to \n 
        # prepended in the code), and 1 line for the truncation warning = 6 lines
        self.assertEqual(len(lines), 6)
        self.assertIn("Results truncated to 3 items", result)
        
        # Robustly verify that exactly 3 files made it into the output
        file_count = sum(1 for line in lines if "file_" in line)
        self.assertEqual(file_count, 3)


if __name__ == "__main__":
    unittest.main()