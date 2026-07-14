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
    Test suite for the ls tool.

    Covers:

    * validation
    * files vs directories
    * depth behavior
    * sorting
    * formatting
    """

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.test_dir.name).resolve()

    def tearDown(self):
        self.test_dir.cleanup()

    def _create_file(self, relative_path: str, text="hello") -> Path:
        p = self.base_path / relative_path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        return p

    def _create_dir(self, relative_path: str) -> Path:
        p = self.base_path / relative_path
        p.mkdir(parents=True, exist_ok=True)
        return p

    #
    # ---------------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------------
    #

    async def test_nonexistent_path(self):
        result = await fs._ls_impl({
            "path": str(self.base_path / "missing")
        })

        self.assertIsInstance(result, ToolFailure)
        self.assertIn("Path does not exist", result.error_message)

    async def test_invalid_depth_string_defaults_to_one(self):
        self._create_dir("src")
        self._create_file("src/main.py")
        self._create_file("src/deep/file.py")

        result = await fs._ls_impl({
            "path": str(self.base_path),
            "depth": "abc"
        })

        self.assertIsInstance(result, str)
        self.assertIn("src/", result)
        self.assertNotIn("main.py", result)
        self.assertNotIn("file.py", result)

    async def test_invalid_depth_none_defaults_to_one(self):
        self._create_dir("src")
        self._create_file("src/main.py")
        self._create_file("src/deep/file.py")

        result = await fs._ls_impl({
            "path": str(self.base_path),
            "depth": None
        })

        self.assertIn("src/", result)
        self.assertNotIn("main.py", result)
        self.assertNotIn("file.py", result)

    async def test_ignore_not_list_is_ignored(self):
        self._create_file("a.txt")

        result = await fs._ls_impl({
            "path": str(self.base_path),
            "ignore": "*.txt"
        })

        self.assertIn("a.txt", result)

    async def test_ignore_non_string_members_are_ignored(self):
        self._create_file("keep.txt")
        self._create_file("remove.py")

        result = await fs._ls_impl({
            "path": str(self.base_path),
            "ignore": [
                "*.py",
                123,
                None,
                object()
            ]
        })

        self.assertIn("keep.txt", result)
        self.assertNotIn("remove.py", result)

    #
    # ---------------------------------------------------------
    # FILES / DIRECTORIES
    # ---------------------------------------------------------
    #

    async def test_empty_directory(self):
        result = await fs._ls_impl({
            "path": str(self.base_path)
        })

        self.assertEqual(
            result.splitlines()[-1],
            "└── [Empty Directory]"
        )

    async def test_root_file(self):
        f = self._create_file("hello.txt")

        result = await fs._ls_impl({
            "path": str(f)
        })

        self.assertIn(str(self.base_path), result)
        self.assertIn("hello.txt", result)
        self.assertEqual(len(result.splitlines()), 2)

    async def test_lists_directory_children(self):
        self._create_file("a.txt")
        self._create_file("b.txt")

        result = await fs._ls_impl({
            "path": str(self.base_path)
        })

        self.assertIn("a.txt", result)
        self.assertIn("b.txt", result)

    async def test_absolute_path_header(self):
        self._create_file("a.txt")

        result = await fs._ls_impl({
            "path": str(self.base_path)
        })

        first = result.splitlines()[0]

        self.assertTrue(first.startswith(str(self.base_path))) # we removed the "Listing:" header
        self.assertIn(str(self.base_path), first)

    async def test_root_directory_has_trailing_slash(self):
        result = await fs._ls_impl({
            "path": str(self.base_path)
        })

        self.assertTrue(
            result.splitlines()[0].endswith("/")
        )

    #
    # ---------------------------------------------------------
    # DEPTH
    # ---------------------------------------------------------
    #

    async def test_depth_zero(self):
        self._create_file("a.txt")

        result = await fs._ls_impl({
            "path": str(self.base_path),
            "depth": 0
        })

        lines = result.splitlines()

        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[1], "└── (depth limit reached)")

    async def test_depth_one(self):
        self._create_dir("src")
        self._create_file("src/main.py")
        self._create_file("src/pkg/file.py")

        result = await fs._ls_impl({
            "path": str(self.base_path),
            "depth": 1
        })

        self.assertIn("src/", result)
        self.assertNotIn("main.py", result)
        self.assertNotIn("file.py", result)

    async def test_depth_two(self):
        self._create_dir("src")
        self._create_file("src/main.py")
        self._create_file("src/pkg/file.py")

        result = await fs._ls_impl({
            "path": str(self.base_path),
            "depth": 2
        })

        self.assertIn("main.py", result)
        self.assertIn("pkg/", result)
        self.assertNotIn("file.py", result)

    async def test_unlimited_depth(self):
        self._create_file("a/b/c/d/e/file.txt")

        result = await fs._ls_impl({
            "path": str(self.base_path),
            "depth": -1
        })

        self.assertIn("file.txt", result)

    #
    # ---------------------------------------------------------
    # SORTING
    # ---------------------------------------------------------
    #

    async def test_directories_before_files(self):
        self._create_file("z.txt")
        self._create_dir("dir")

        result = await fs._ls_impl({
            "path": str(self.base_path)
        })

        lines = result.splitlines()[1:]

        self.assertTrue(lines[0].startswith("├── dir/"))
        self.assertTrue(lines[-1].endswith("z.txt"))

    async def test_case_insensitive_sorting(self):
        self._create_dir("beta")
        self._create_dir("Alpha")

        result = await fs._ls_impl({
            "path": str(self.base_path)
        })

        lines = result.splitlines()[1:]

        self.assertIn("Alpha/", lines[0])
        self.assertIn("beta/", lines[1])

    async def test_files_sorted_case_insensitive(self):
        self._create_file("zebra.txt")
        self._create_file("Apple.txt")
        self._create_file("banana.txt")

        result = await fs._ls_impl({
            "path": str(self.base_path)
        })

        files = [
            l for l in result.splitlines()[1:]
            if ".txt" in l
        ]

        self.assertIn("Apple.txt", files[0])
        self.assertIn("banana.txt", files[1])
        self.assertIn("zebra.txt", files[2])

    async def test_directory_counts_are_displayed(self):
        self._create_dir("src")
        self._create_file("src/a.py")
        self._create_file("src/b.py")

        result = await fs._ls_impl({
            "path": str(self.base_path)
        })

        self.assertIn("src/ (2 items)", result)

    async def test_directory_count_singular(self):
        self._create_dir("src")
        self._create_file("src/a.py")

        result = await fs._ls_impl({
            "path": str(self.base_path)
        })

        self.assertIn("src/ (1 item)", result)

    #
    # ---------------------------------------------------------
    # IGNORE RULES
    # ---------------------------------------------------------
    #

    async def test_default_ignore_git(self):
        self._create_dir(".git")
        self._create_file(".git/config")
        self._create_file("visible.txt")

        result = await fs._ls_impl({"path": str(self.base_path)})

        self.assertIn("visible.txt", result)
        self.assertNotIn(".git", result)

    async def test_default_ignore_node_modules(self):
        self._create_dir("node_modules")
        self._create_file("node_modules/pkg.json")
        self._create_file("main.py")

        result = await fs._ls_impl({"path": str(self.base_path)})

        self.assertIn("main.py", result)
        self.assertNotIn("node_modules", result)

    async def test_default_ignore_pycache(self):
        self._create_dir("__pycache__")
        self._create_file("__pycache__/mod.pyc")
        self._create_file("main.py")

        result = await fs._ls_impl({"path": str(self.base_path)})

        self.assertIn("main.py", result)
        self.assertNotIn("__pycache__", result)

    async def test_default_ignore_pyc_files(self):
        self._create_file("good.py")
        self._create_file("bad.pyc")

        result = await fs._ls_impl({"path": str(self.base_path)})

        self.assertIn("good.py", result)
        self.assertNotIn("bad.pyc", result)

    async def test_custom_ignore_pattern(self):
        self._create_file("keep.py")
        self._create_file("remove.log")

        result = await fs._ls_impl({
            "path": str(self.base_path),
            "ignore": ["*.log"]
        })

        self.assertIn("keep.py", result)
        self.assertNotIn("remove.log", result)

    async def test_custom_ignore_directory(self):
        self._create_dir("build")
        self._create_file("build/a.txt")
        self._create_file("main.py")

        result = await fs._ls_impl({
            "path": str(self.base_path),
            "ignore": ["build"]
        })

        self.assertIn("main.py", result)
        self.assertNotIn("build", result)

    async def test_directory_count_respects_ignore(self):
        self._create_dir("src")
        self._create_file("src/a.py")
        self._create_file("src/debug.log")

        result = await fs._ls_impl({
            "path": str(self.base_path),
            "ignore": ["*.log"]
        })

        self.assertIn("src/ (1 item)", result)

    #
    # ---------------------------------------------------------
    # TREE FORMATTING
    # ---------------------------------------------------------
    #

    async def test_tree_contains_connectors(self):
        # Create multiple files in 'src/' to get '├──'
        self._create_file("src/main.py")
        self._create_file("src/helper.py")
        
        # Create a sibling at the root level to force the vertical '│' 
        self._create_file("z_root_sibling.txt")

        result = await fs._ls_impl({
            "path": str(self.base_path),
            "depth": 2
        })

        self.assertIn("├──", result)
        self.assertIn("│", result) # This will now pass successfully!

    async def test_last_child_uses_corner_connector(self):
        self._create_file("only.txt")

        result = await fs._ls_impl({
            "path": str(self.base_path)
        })

        self.assertTrue(result.splitlines()[1].startswith("└──"))

    async def test_nested_prefixes_are_correct(self):
        # Create a sibling structure: 'b/' and 'z_sibling.txt' inside 'a/'
        self._create_file("a/b/file.txt")
        self._create_file("a/z_sibling.txt") # This forces 'b/' to use '│' because it's not the last item in 'a/'

        result = await fs._ls_impl({
            "path": str(self.base_path),
            "depth": -1
        })

        self.assertIn("file.txt", result)
        self.assertIn("│", result)

    #
    # ---------------------------------------------------------
    # SYMLINKS
    # ---------------------------------------------------------
    #

    def _symlinks_supported(self):
        probe = self.base_path / "probe"
        target = self.base_path / "target"

        try:
            target.write_text("x")
            probe.symlink_to(target)
            return True
        except (OSError, NotImplementedError):
            return False
        finally:
            try:
                probe.unlink()
            except Exception:
                pass
            try:
                target.unlink()
            except Exception:
                pass

    async def test_root_file_symlink(self):
        if not self._symlinks_supported():
            self.skipTest("Symlinks unavailable")

        target = self._create_file("real.txt")
        link = self.base_path / "link.txt"
        link.symlink_to(target)

        result = await fs._ls_impl({
            "path": str(link)
        })

        self.assertIn("link.txt ->", result)
        self.assertIn("real.txt", result)

    async def test_broken_symlink(self):
        if not self._symlinks_supported():
            self.skipTest("Symlinks unavailable")

        target = self.base_path / "missing.txt"
        link = self.base_path / "broken.txt"
        link.symlink_to(target)

        result = await fs._ls_impl({
            "path": str(link)
        })

        self.assertIn("broken.txt ->", result)

    async def test_directory_symlink_not_recursed(self):
        if not self._symlinks_supported():
            self.skipTest("Symlinks unavailable")

        real = self._create_dir("real")
        self._create_file("real/file.txt")

        link = self.base_path / "linked"
        link.symlink_to(real, target_is_directory=True)

        result = await fs._ls_impl({
            "path": str(self.base_path),
            "depth": -1
        })

        self.assertIn("linked ->", result)

        linked_index = result.index("linked ->")
        self.assertNotIn("file.txt", result[linked_index:])

    async def test_unreadable_readlink(self):
        if not self._symlinks_supported():
            self.skipTest("Symlinks unavailable")

        target = self._create_file("real.txt")
        link = self.base_path / "link.txt"
        link.symlink_to(target)

        original = Path.readlink

        def boom(self):
            if self == link:
                raise OSError("boom")
            return original(self)

        with patch.object(Path, "readlink", autospec=True, side_effect=boom):
            result = await fs._ls_impl({
                "path": str(link)
            })

        self.assertIn("[unreadable link]", result)

    #
    # ---------------------------------------------------------
    # DIRECTORY COUNT FAILURES
    # ---------------------------------------------------------
    #

    async def test_directory_count_scandir_failure(self):
        self._create_dir("src")

        with patch("os.scandir", side_effect=PermissionError):
            result = await fs._ls_impl({
                "path": str(self.base_path)
            })

        self.assertIn("src/", result)
        self.assertNotIn("(1 item)", result)
        self.assertNotIn("(0 items)", result)

    #
    # ---------------------------------------------------------
    # PERMISSION FAILURES
    # ---------------------------------------------------------
    #

    async def test_unreadable_directory(self):
        original = Path.iterdir

        def boom(self):
            if self == self_test.base_path:
                raise PermissionError("Permission denied")
            return original(self)

        self_test = self

        with patch.object(Path, "iterdir", autospec=True, side_effect=boom):
            result = await fs._ls_impl({
                "path": str(self.base_path)
            })

        self.assertIsInstance(result, str)
        self.assertIn("Unable to read", result)

    async def test_is_dir_failure_marks_unreadable(self):
        self._create_file("a.txt")

        original = Path.is_dir

        def selective(self):
            if self.name == "a.txt":
                raise OSError("boom")
            return original(self)

        with patch.object(Path, "is_dir", autospec=True, side_effect=selective):
            result = await fs._ls_impl({
                "path": str(self.base_path)
            })

        self.assertIn("[unreadable]", result)

    async def test_readlink_failure_inside_directory(self):
        if not self._symlinks_supported():
            self.skipTest("Symlinks unavailable")

        target = self._create_file("real.txt")
        link = self.base_path / "link.txt"
        link.symlink_to(target)

        original = Path.readlink

        def selective(self):
            if self == link:
                raise OSError("boom")
            return original(self)

        with patch.object(Path, "readlink", autospec=True, side_effect=selective):
            result = await fs._ls_impl({
                "path": str(self.base_path)
            })

        self.assertIn("[unreadable link]", result)

    #
    # ---------------------------------------------------------
    # RACE CONDITIONS
    # ---------------------------------------------------------
    #

    async def test_file_removed_before_is_dir(self):
        self._create_file("gone.txt")

        original = Path.is_dir

        def selective(self):
            if self.name == "gone.txt":
                raise FileNotFoundError()
            return original(self)

        with patch.object(Path, "is_dir", autospec=True, side_effect=selective):
            result = await fs._ls_impl({
                "path": str(self.base_path)
            })

        self.assertIn("gone.txt [unreadable]", result)

    async def test_directory_removed_before_iteration(self):
        d = self._create_dir("src")

        original = Path.iterdir

        def selective(self):
            if self == d:
                raise FileNotFoundError()
            return original(self)

        with patch.object(Path, "iterdir", autospec=True, side_effect=selective):
            result = await fs._ls_impl({
                "path": str(self.base_path),
                "depth": 2
            })

        self.assertIn("Unable to read", result)

    async def test_scandir_failure_does_not_abort_listing(self):
        self._create_dir("src")
        self._create_file("root.py")

        with patch("os.scandir", side_effect=PermissionError):
            result = await fs._ls_impl({
                "path": str(self.base_path)
            })

        self.assertIn("src/", result)
        self.assertIn("root.py", result)

    #
    # ---------------------------------------------------------
    # TRUNCATION
    # ---------------------------------------------------------
    #

    @patch("tools.filesearch.MAX_LS_ENTRIES", 5)
    async def test_truncation(self):
        for i in range(20):
            self._create_file(f"file{i}.txt")

        result = await fs._ls_impl({
            "path": str(self.base_path)
        })

        self.assertIn("Results truncated", result)

    @patch("tools.filesearch.MAX_LS_ENTRIES", 3)
    async def test_truncation_keeps_header(self):
        for i in range(10):
            self._create_file(f"{i}.txt")

        result = await fs._ls_impl({
            "path": str(self.base_path)
        })

        lines = result.splitlines()

        self.assertTrue(lines[0].startswith(str(self.base_path)))
        self.assertIn("Results truncated", result)

    #
    # ---------------------------------------------------------
    # REGRESSION TESTS
    # ---------------------------------------------------------
    #

    async def test_listing_single_directory(self):
        self._create_dir("src")

        result = await fs._ls_impl({
            "path": str(self.base_path)
        })

        self.assertEqual(len(result.splitlines()), 2)

    async def test_listing_single_file(self):
        self._create_file("main.py")

        result = await fs._ls_impl({
            "path": str(self.base_path)
        })

        self.assertIn("main.py", result)

    async def test_listing_multiple_directories(self):
        self._create_dir("a")
        self._create_dir("b")
        self._create_dir("c")

        result = await fs._ls_impl({
            "path": str(self.base_path)
        })

        self.assertIn("a/", result)
        self.assertIn("b/", result)
        self.assertIn("c/", result)

    async def test_nested_directory_structure(self):
        self._create_file("src/pkg/util/helpers.py")

        result = await fs._ls_impl({
            "path": str(self.base_path),
            "depth": -1
        })

        self.assertIn("src/", result)
        self.assertIn("pkg/", result)
        self.assertIn("util/", result)
        self.assertIn("helpers.py", result)

    async def test_directory_with_mixed_files(self):
        self._create_file("a.py")
        self._create_file("b.txt")
        self._create_file("c.md")

        result = await fs._ls_impl({
            "path": str(self.base_path)
        })

        self.assertIn("a.py", result)
        self.assertIn("b.txt", result)
        self.assertIn("c.md", result)

    async def test_default_ignored_items_not_counted(self):
        self._create_dir("src")
        self._create_file("src/main.py")
        self._create_dir("src/__pycache__")
        self._create_file("src/__pycache__/x.pyc")

        result = await fs._ls_impl({
            "path": str(self.base_path)
        })

        self.assertIn("src/ (1 item)", result)

    async def test_header_for_file_has_no_trailing_slash(self):
        f = self._create_file("hello.txt")

        result = await fs._ls_impl({
            "path": str(f)
        })

        self.assertFalse(result.splitlines()[0].endswith("/"))

    async def test_large_tree(self):
        for d in range(10):
            for f in range(10):
                self._create_file(f"dir{d}/file{f}.txt")

        result = await fs._ls_impl({
            "path": str(self.base_path),
            "depth": -1
        })

        self.assertIsInstance(result, str)
        self.assertIn("dir0/", result)
        self.assertIn("file0.txt", result)

    async def test_empty_subdirectory(self):
        self._create_dir("empty")

        result = await fs._ls_impl({
            "path": str(self.base_path),
            "depth": 2
        })

        self.assertIn("empty/", result)

    async def test_ignore_multiple_patterns(self):
        self._create_file("keep.py")
        self._create_file("drop.log")
        self._create_file("drop.tmp")

        result = await fs._ls_impl({
            "path": str(self.base_path),
            "ignore": ["*.log", "*.tmp"]
        })

        self.assertIn("keep.py", result)
        self.assertNotIn("drop.log", result)
        self.assertNotIn("drop.tmp", result)