import os
import unittest
import tempfile
from pathlib import Path

from sessioncontext import InvocationContext
from tools.paths import resolve_in_workspace
from tools.filesystem import _read_impl, _write_impl, _edit_impl, _multiedit_impl
from tools.filesearch import _glob_impl, _ls_impl
from typedefs import ToolFailure


def _supports_symlinks(base: Path) -> bool:
    """Symlink creation may need elevated privileges on Windows."""
    target = base / "_symlink_probe_target.txt"
    link = base / "_symlink_probe_link.txt"
    target.write_text("probe")
    try:
        os.symlink(target, link)
        return True
    except (OSError, NotImplementedError):
        return False
    finally:
        if link.exists() or link.is_symlink():
            link.unlink()
        target.unlink()


class TestWorkspaceBoundary(unittest.IsolatedAsyncioTestCase):
    """
    Test Suite for workspace boundary enforcement.

    Every file tool must refuse to operate on paths that resolve outside
    ctx.workspace: absolute paths elsewhere on disk, `..` traversal, and
    symlinks pointing out of the workspace.
    """

    def setUp(self):
        # Workspace directory (with a subdirectory to act as cwd)
        self.workspace_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.workspace_dir.name).resolve()
        self.subdir = self.workspace / "sub"
        self.subdir.mkdir()

        # A sibling directory OUTSIDE the workspace, containing a "secret"
        self.outside_dir = tempfile.TemporaryDirectory()
        self.outside = Path(self.outside_dir.name).resolve()
        self.secret_file = self.outside / "secret.txt"
        self.secret_file.write_text("TOP SECRET", encoding="utf-8")

        self.ctx = InvocationContext(
            workspace=self.workspace,
            cwd=self.workspace,
            workspace_is_git_repo=False,
            resume_file=None
        )

    def tearDown(self):
        self.workspace_dir.cleanup()
        self.outside_dir.cleanup()

    def assertDenied(self, result):
        self.assertIsInstance(result, ToolFailure)
        self.assertIn("outside", result.error_message)
        self.assertIn("Access denied", result.error_message)

    # ---------------------------------------------------------
    # RESOLVER HELPER
    # ---------------------------------------------------------

    async def test_resolver_allows_inside_and_denies_outside(self):
        inside = resolve_in_workspace(str(self.workspace / "a.txt"), self.ctx)
        self.assertIsInstance(inside, Path)

        outside = resolve_in_workspace(str(self.secret_file), self.ctx)
        self.assertDenied(outside)

    async def test_resolver_relative_paths_use_ctx_cwd(self):
        # ctx.cwd is a subdirectory of the workspace; a bare relative path
        # must resolve against it (NOT the process cwd, which is elsewhere).
        sub_ctx = InvocationContext(
            workspace=self.workspace,
            cwd=self.subdir,
            workspace_is_git_repo=False,
            resume_file=None
        )
        resolved = resolve_in_workspace("notes.txt", sub_ctx)
        self.assertEqual(resolved, (self.subdir / "notes.txt").resolve())

    async def test_resolver_denies_dotdot_traversal(self):
        # `..` climbing from inside the workspace to outside must be denied,
        # no matter how deep the prefix looks.
        escape = str(self.workspace / "sub" / ".." / ".." / self.outside.name / "secret.txt")
        result = resolve_in_workspace(escape, self.ctx)
        self.assertDenied(result)

    # ---------------------------------------------------------
    # FILESYSTEM TOOLS
    # ---------------------------------------------------------

    async def test_read_denied_outside_workspace(self):
        result = await _read_impl({"file_path": str(self.secret_file)}, self.ctx)
        self.assertDenied(result)
        # The denial must not leak whether/what files exist there
        self.assertNotIn("Did you mean", result.error_message)

    async def test_read_denied_dotdot_traversal(self):
        traversal = str(self.workspace) + os.sep + os.pardir + os.sep + self.outside.name + os.sep + "secret.txt"
        result = await _read_impl({"file_path": traversal}, self.ctx)
        self.assertDenied(result)

    async def test_write_denied_outside_workspace(self):
        target = self.outside / "evil.txt"
        result = await _write_impl({"file_path": str(target), "content": "pwned"}, self.ctx)
        self.assertDenied(result)
        self.assertFalse(target.exists())

    async def test_edit_denied_outside_workspace(self):
        result = await _edit_impl({
            "file_path": str(self.secret_file),
            "old_string": "TOP SECRET",
            "new_string": "changed"
        }, self.ctx)
        self.assertDenied(result)
        self.assertEqual(self.secret_file.read_text(), "TOP SECRET")

    async def test_multiedit_denied_outside_workspace(self):
        result = await _multiedit_impl({
            "file_path": str(self.secret_file),
            "edits": [{"old_string": "TOP SECRET", "new_string": "changed"}]
        }, self.ctx)
        self.assertDenied(result)
        self.assertEqual(self.secret_file.read_text(), "TOP SECRET")

    async def test_read_relative_path_resolves_inside_cwd(self):
        (self.subdir / "hello.txt").write_text("hi there", encoding="utf-8")
        sub_ctx = InvocationContext(
            workspace=self.workspace,
            cwd=self.subdir,
            workspace_is_git_repo=False,
            resume_file=None
        )
        result = await _read_impl({"file_path": "hello.txt"}, sub_ctx)
        self.assertIsInstance(result, str)
        self.assertIn("hi there", result)

    # ---------------------------------------------------------
    # SEARCH TOOLS
    # ---------------------------------------------------------

    async def test_glob_denied_outside_workspace(self):
        result = await _glob_impl({"pattern": "**/*", "path": str(self.outside)}, self.ctx)
        self.assertDenied(result)

    async def test_glob_allowed_inside_workspace(self):
        (self.workspace / "found.py").write_text("x = 1")
        result = await _glob_impl({"pattern": "**/*.py", "path": str(self.workspace)}, self.ctx)
        self.assertIsInstance(result, str)
        self.assertIn("found.py", result)

    async def test_ls_denied_outside_workspace(self):
        result = await _ls_impl({"path": str(self.outside)}, self.ctx)
        self.assertDenied(result)

    async def test_ls_allowed_inside_workspace(self):
        result = await _ls_impl({"path": str(self.workspace)}, self.ctx)
        self.assertIsInstance(result, str)
        self.assertIn("sub/", result)

    # ---------------------------------------------------------
    # SYMLINK ESCAPES
    # ---------------------------------------------------------

    async def test_symlink_escape_denied(self):
        if not _supports_symlinks(self.workspace):
            self.skipTest("symlinks not supported in this environment")

        # A symlink INSIDE the workspace pointing at a file OUTSIDE it.
        link = self.workspace / "innocent_link.txt"
        os.symlink(self.secret_file, link)

        read_res = await _read_impl({"file_path": str(link)}, self.ctx)
        self.assertDenied(read_res)

        write_res = await _write_impl({"file_path": str(link), "content": "pwned"}, self.ctx)
        self.assertDenied(write_res)
        self.assertEqual(self.secret_file.read_text(), "TOP SECRET")

    async def test_symlink_dir_escape_denied_for_ls(self):
        if not _supports_symlinks(self.workspace):
            self.skipTest("symlinks not supported in this environment")

        # A directory symlink INSIDE the workspace pointing OUTSIDE it.
        link_dir = self.workspace / "innocent_dir"
        os.symlink(self.outside, link_dir, target_is_directory=True)

        ls_res = await _ls_impl({"path": str(link_dir)}, self.ctx)
        self.assertDenied(ls_res)

        glob_res = await _glob_impl({"pattern": "**/*", "path": str(link_dir)}, self.ctx)
        self.assertDenied(glob_res)


if __name__ == "__main__":
    unittest.main()
