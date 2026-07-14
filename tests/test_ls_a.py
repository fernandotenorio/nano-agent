import os
import time
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Adjust import paths depending on your exact project structure
import tools.filesearch as fs
from tools.filesearch import _ls_impl
from typedefs import ToolFailure

# chatgpt
class TestLsTool(unittest.IsolatedAsyncioTestCase):
    """
    Production-focused test suite for _ls_impl.

    Covers:
    - validation
    - files vs directories
    - depth handling
    - sorting
    - exclude rules
    - directory counts
    - symlinks
    - truncation
    - permission errors
    - unreadable entries
    """

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.test_dir.name).resolve()

    def tearDown(self):
        self.test_dir.cleanup()

    def _create_file(self, relative_path: str, content: str = "x") -> Path:
        path = self.base_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    def _create_dir(self, relative_path: str) -> Path:
        path = self.base_path / relative_path
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ---------------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------------

    async def test_nonexistent_path_returns_toolfailure(self):
        result = await fs._ls_impl(
            {"path": str(self.base_path / "missing")}
        )

        self.assertIsInstance(result, ToolFailure)
        self.assertIn("Path does not exist", result.error_message)

    async def test_invalid_depth_string_defaults_to_one(self):
        self._create_file("root.txt")
        self._create_file("nested/child.txt")

        result = await fs._ls_impl(
            {
                "path": str(self.base_path),
                "depth": "abc",
            }
        )

        self.assertIsInstance(result, str)
        self.assertIn("root.txt", result)
        self.assertIn("nested/", result)
        self.assertNotIn("child.txt", result)

    async def test_invalid_depth_none_defaults_to_one(self):
        self._create_file("nested/child.txt")

        result = await fs._ls_impl(
            {
                "path": str(self.base_path),
                "depth": None,
            }
        )

        self.assertIsInstance(result, str)
        self.assertIn("nested/", result)
        self.assertNotIn("child.txt", result)

    async def test_ignore_not_list_is_ignored(self):
        self._create_file("a.txt")

        result = await fs._ls_impl(
            {
                "path": str(self.base_path),
                "exclude": "*.txt",
            }
        )

        self.assertIn("a.txt", result)

    async def test_non_string_ignore_entries_are_filtered(self):
        self._create_file("a.txt")
        self._create_file("b.py")

        result = await fs._ls_impl(
            {
                "path": str(self.base_path),
                "exclude": ["*.txt", 123, None, object()],
            }
        )

        self.assertNotIn("a.txt", result)
        self.assertIn("b.py", result)

    # ---------------------------------------------------------
    # FILES / DIRECTORIES
    # ---------------------------------------------------------

    async def test_empty_directory(self):
        result = await fs._ls_impl(
            {"path": str(self.base_path)}
        )

        self.assertIn("[Empty Directory]", result)

    async def test_root_file_returns_leaf(self):
        file_path = self._create_file("single.txt")

        result = await fs._ls_impl(
            {"path": str(file_path)}
        )

        self.assertIn(str(file_path), result)
        self.assertIn("single.txt", result)
        self.assertIn("└──", result)

    async def test_root_directory_lists_children(self):
        self._create_file("a.txt")
        self._create_file("b.txt")

        result = await fs._ls_impl(
            {"path": str(self.base_path)}
        )

        self.assertIn("a.txt", result)
        self.assertIn("b.txt", result)

    # ---------------------------------------------------------
    # DEPTH
    # ---------------------------------------------------------

    async def test_depth_zero(self):
        self._create_file("a.txt")

        result = await fs._ls_impl(
            {
                "path": str(self.base_path),
                "depth": 0,
            }
        )

        self.assertIn("(depth limit reached)", result)
        self.assertNotIn("a.txt", result)

    async def test_depth_one(self):
        self._create_file("nested/child.txt")

        result = await fs._ls_impl(
            {
                "path": str(self.base_path),
                "depth": 1,
            }
        )

        self.assertIn("nested/", result)
        self.assertNotIn("child.txt", result)

    async def test_depth_two(self):
        self._create_file("nested/child.txt")

        result = await fs._ls_impl(
            {
                "path": str(self.base_path),
                "depth": 2,
            }
        )

        self.assertIn("nested/", result)
        self.assertIn("child.txt", result)

    async def test_negative_depth_is_unlimited(self):
        self._create_file("a/b/c/d/file.txt")

        result = await fs._ls_impl(
            {
                "path": str(self.base_path),
                "depth": -1,
            }
        )

        self.assertIn("file.txt", result)

    # ---------------------------------------------------------
    # SORTING
    # ---------------------------------------------------------

    async def test_directories_sort_before_files(self):
        self._create_dir("z_dir")
        self._create_dir("a_dir")

        self._create_file("z.txt")
        self._create_file("a.txt")

        result = await fs._ls_impl(
            {
                "path": str(self.base_path),
                "depth": 1,
            }
        )

        lines = result.splitlines()

        idx_a_dir = next(i for i, l in enumerate(lines) if "a_dir/" in l)
        idx_z_dir = next(i for i, l in enumerate(lines) if "z_dir/" in l)
        idx_a_file = next(i for i, l in enumerate(lines) if "a.txt" in l)

        self.assertLess(idx_a_dir, idx_a_file)
        self.assertLess(idx_z_dir, idx_a_file)

    async def test_case_insensitive_sorting(self):
        self._create_file("Apple.txt")
        self._create_file("zebra.txt")

        result = await fs._ls_impl(
            {"path": str(self.base_path)}
        )

        lines = result.splitlines()

        apple = next(i for i, l in enumerate(lines) if "Apple.txt" in l)
        zebra = next(i for i, l in enumerate(lines) if "zebra.txt" in l)

        self.assertLess(apple, zebra)

    # ---------------------------------------------------------
    # EXCLUDES
    # ---------------------------------------------------------

    async def test_default_git_ignore(self):
        self._create_file(".git/config")
        self._create_file("visible.txt")

        result = await fs._ls_impl(
            {"path": str(self.base_path), "depth": 2}
        )

        self.assertIn("visible.txt", result)
        self.assertNotIn(".git", result)

    async def test_default_pycache_ignore(self):
        self._create_file("__pycache__/a.pyc")
        self._create_file("normal.py")

        result = await fs._ls_impl(
            {"path": str(self.base_path), "depth": 2}
        )

        self.assertIn("normal.py", result)
        self.assertNotIn("__pycache__", result)

    async def test_user_ignore_pattern(self):
        self._create_file("a.txt")
        self._create_file("b.py")

        result = await fs._ls_impl(
            {
                "path": str(self.base_path),
                "exclude": ["*.txt"],
            }
        )

        self.assertNotIn("a.txt", result)
        self.assertIn("b.py", result)

    # ---------------------------------------------------------
    # DIRECTORY COUNTS
    # ---------------------------------------------------------

    async def test_directory_count_plural(self):
        self._create_file("dir/a.txt")
        self._create_file("dir/b.txt")

        result = await fs._ls_impl(
            {"path": str(self.base_path)}
        )

        self.assertIn("(2 items)", result)

    async def test_directory_count_singular(self):
        self._create_file("dir/a.txt")

        result = await fs._ls_impl(
            {"path": str(self.base_path)}
        )

        self.assertIn("(1 item)", result)

    async def test_directory_count_respects_ignore(self):
        self._create_file("dir/a.py")
        self._create_file("dir/.gitkeep")

        result = await fs._ls_impl(
            {
                "path": str(self.base_path),
                "exclude": [".gitkeep"],
            }
        )

        self.assertIn("(1 item)", result)

    # ---------------------------------------------------------
    # SYMLINKS
    # ---------------------------------------------------------

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks unavailable")
    async def test_symlink_to_file(self):
        target = self._create_file("real.txt")

        link = self.base_path / "link.txt"
        os.symlink(target, link)

        result = await fs._ls_impl(
            {"path": str(link)}
        )

        self.assertIn("link.txt", result)
        self.assertIn("->", result)

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks unavailable")
    async def test_broken_symlink(self):
        target = self.base_path / "missing.txt"
        link = self.base_path / "broken.txt"

        os.symlink(target, link)

        result = await fs._ls_impl(
            {"path": str(link)}
        )

        self.assertIn("broken.txt", result)
        self.assertIn("->", result)

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks unavailable")
    async def test_directory_symlink_not_recursed(self):
        real_dir = self._create_dir("real")
        self._create_file("real/file.txt")

        link = self.base_path / "dirlink"
        os.symlink(real_dir, link)

        result = await fs._ls_impl(
            {
                "path": str(self.base_path),
                "depth": 3,
            }
        )

        self.assertIn("dirlink ->", result)

    # ---------------------------------------------------------
    # PERMISSION / ERROR PATHS
    # ---------------------------------------------------------

    async def test_root_iterdir_permission_error(self):
        with patch(
            "pathlib.Path.iterdir",
            side_effect=PermissionError("Permission denied")
        ):
            result = await fs._ls_impl(
                {"path": str(self.base_path)}
            )

        self.assertIn("Unable to read", result)

    async def test_directory_count_scandir_failure(self):
        self._create_file("dir/file.txt")

        with patch(
            "os.scandir",
            side_effect=PermissionError("denied")
        ):
            result = await fs._ls_impl(
                {"path": str(self.base_path)}
            )

        self.assertIn("dir/", result)
        self.assertNotIn("(1 item)", result)

    async def test_unreadable_entry_during_enumeration(self):
        self._create_file("a.txt")

        original_is_dir = Path.is_dir

        def selective_failure(path_obj):
            if path_obj.name == "a.txt":
                raise OSError("boom")
            return original_is_dir(path_obj)

        with patch(
            "pathlib.Path.is_dir",
            autospec=True,
            side_effect=selective_failure
        ):
            result = await fs._ls_impl(
                {"path": str(self.base_path)}
            )

        self.assertIn("[unreadable]", result)

    async def test_unreadable_symlink_target(self):
        self._create_file("target.txt")

        with patch(
            "pathlib.Path.readlink",
            side_effect=OSError("fail")
        ):
            link = self.base_path / "link"

            with patch(
                "pathlib.Path.is_symlink",
                return_value=True
            ):
                result = await fs._ls_impl(
                    {"path": str(link)}
                )

        self.assertIn("[unreadable link]", result)

    # ---------------------------------------------------------
    # TRUNCATION
    # ---------------------------------------------------------

    @patch("tools.filesearch.MAX_LS_ENTRIES", 3)
    async def test_truncation(self):
        for i in range(10):
            self._create_file(f"file_{i}.txt")

        result = await fs._ls_impl(
            {
                "path": str(self.base_path),
                "depth": 1,
            }
        )

        self.assertIn("Results truncated", result)

    # ---------------------------------------------------------
    # FORMATTING
    # ---------------------------------------------------------

    async def test_tree_connectors_present(self):
        self._create_file("dir/file.txt")

        result = await fs._ls_impl(
            {
                "path": str(self.base_path),
                "depth": 2,
            }
        )

        # Correct way to assert an "OR" condition in unittest
        self.assertTrue("├──" in result or "└──" in result)
        
        # Or even more explicitly, we can just assert the one we know is there:
        self.assertIn("└──", result)

    async def test_header_contains_absolute_path(self):
        result = await fs._ls_impl(
            {"path": str(self.base_path)}
        )

        expected = f"{self.base_path}/"
        self.assertTrue(result.startswith(expected))


