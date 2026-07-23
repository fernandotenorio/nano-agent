import os
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import tools.filesystem as fs
from tools.filesystem import _read_impl, _write_impl, _edit_impl, _multiedit_impl
from filestate import FileStateTracker, file_changes_hook
from hooks import UserPromptEvent
from sessioncontext import InvocationContext
from typedefs import ToolFailure


def _touch_with_new_mtime(path: Path) -> None:
    """Bumps the file's mtime without changing its content."""
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000))


def _modify_externally(path: Path, content: str) -> None:
    """Simulates the user (or another process) changing a file on disk.

    Also bumps mtime explicitly so the test never depends on filesystem
    timestamp granularity.
    """
    path.write_text(content, encoding="utf-8")
    _touch_with_new_mtime(path)


class TestFileStateTracking(unittest.IsolatedAsyncioTestCase):
    """
    Test Suite for on-demand file staleness detection:
    the write gate (Write/Edit/MultiEdit), the user-prompt diff hook,
    and per-agent state isolation.
    """

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.test_dir.name)

        self.ctx = InvocationContext(
            workspace=self.base_path,
            cwd=self.base_path,
            workspace_is_git_repo=False,
            resume_file=None
        )

    def tearDown(self):
        self.test_dir.cleanup()

    async def _read(self, path: Path):
        return await _read_impl({"file_path": str(path)}, self.ctx)

    # ---------------------------------------------------------
    # WRITE GATE: external modifications must block writes
    # ---------------------------------------------------------

    async def test_external_modification_blocks_write(self):
        path = self.base_path / "target.txt"
        path.write_text("original content", encoding="utf-8")

        await self._read(path)
        _modify_externally(path, "changed by the user meanwhile")

        result = await _write_impl({"file_path": str(path), "content": "agent overwrite"}, self.ctx)

        self.assertIsInstance(result, ToolFailure)
        self.assertIn("modified on disk since you last read it", result.error_message)
        # The external content must NOT have been overwritten
        self.assertEqual(path.read_text(), "changed by the user meanwhile")

    async def test_external_modification_blocks_edit(self):
        path = self.base_path / "code.py"
        path.write_text("x = 1", encoding="utf-8")

        await self._read(path)
        _modify_externally(path, "x = 1  # user comment")

        result = await _edit_impl({
            "file_path": str(path), "old_string": "x = 1", "new_string": "x = 2"
        }, self.ctx)

        self.assertIsInstance(result, ToolFailure)
        self.assertIn("modified on disk since you last read it", result.error_message)
        self.assertEqual(path.read_text(), "x = 1  # user comment")

    async def test_external_modification_blocks_multiedit(self):
        path = self.base_path / "multi.txt"
        path.write_text("alpha beta", encoding="utf-8")

        await self._read(path)
        _modify_externally(path, "alpha beta gamma")

        result = await _multiedit_impl({
            "file_path": str(path),
            "edits": [{"old_string": "alpha", "new_string": "delta"}]
        }, self.ctx)

        self.assertIsInstance(result, ToolFailure)
        self.assertIn("modified on disk since you last read it", result.error_message)
        self.assertEqual(path.read_text(), "alpha beta gamma")

    async def test_reread_unblocks_write(self):
        path = self.base_path / "target.txt"
        path.write_text("original", encoding="utf-8")

        await self._read(path)
        _modify_externally(path, "externally changed")

        blocked = await _write_impl({"file_path": str(path), "content": "new"}, self.ctx)
        self.assertIsInstance(blocked, ToolFailure)

        # Re-reading refreshes the tracker; the write must now succeed
        await self._read(path)
        result = await _write_impl({"file_path": str(path), "content": "new"}, self.ctx)

        self.assertNotIsInstance(result, ToolFailure)
        self.assertEqual(path.read_text(), "new")

    # ---------------------------------------------------------
    # FALSE ALARMS: timestamp-only changes must not block
    # ---------------------------------------------------------

    async def test_touch_without_content_change_does_not_block(self):
        path = self.base_path / "touched.txt"
        path.write_text("stable content", encoding="utf-8")

        await self._read(path)
        _touch_with_new_mtime(path)  # new mtime, identical content

        result = await _write_impl({"file_path": str(path), "content": "updated"}, self.ctx)

        self.assertNotIsInstance(result, ToolFailure)
        self.assertEqual(path.read_text(), "updated")

    async def test_touch_refreshes_signature(self):
        path = (self.base_path / "touched2.txt")
        path.write_text("stable", encoding="utf-8")
        resolved = path.resolve()

        await self._read(path)
        _touch_with_new_mtime(path)

        # The content-compare fallback should refresh the stored signature
        self.assertEqual(self.ctx.file_state.status(resolved), "fresh")
        self.assertEqual(self.ctx.file_state.known[resolved].mtime_ns, path.stat().st_mtime_ns)

    # ---------------------------------------------------------
    # SELF-MODIFICATIONS: the agent's own writes never look stale
    # ---------------------------------------------------------

    async def test_own_write_then_edit_is_not_blocked(self):
        path = self.base_path / "own.py"
        path.write_text("def foo(): pass", encoding="utf-8")

        await self._read(path)

        r1 = await _write_impl({"file_path": str(path), "content": "def foo():\n    return 1"}, self.ctx)
        self.assertNotIsInstance(r1, ToolFailure)

        # Immediately edit the file we just wrote (same turn, no re-Read)
        r2 = await _edit_impl({
            "file_path": str(path), "old_string": "return 1", "new_string": "return 2"
        }, self.ctx)
        self.assertNotIsInstance(r2, ToolFailure)

        # And multi-edit it again right after
        r3 = await _multiedit_impl({
            "file_path": str(path),
            "edits": [{"old_string": "return 2", "new_string": "return 3"}]
        }, self.ctx)
        self.assertNotIsInstance(r3, ToolFailure)
        self.assertIn("return 3", path.read_text())

    # ---------------------------------------------------------
    # PROMPT HOOK: diff reminders and per-item refresh
    # ---------------------------------------------------------

    async def test_prompt_hook_emits_diff_and_refreshes(self):
        path = (self.base_path / "watched.txt")
        path.write_text("line one\nline two\nline three", encoding="utf-8")

        await self._read(path)
        _modify_externally(path, "line one\nline 2 EDITED\nline three")

        event = UserPromptEvent(prompt="next question", is_first_prompt=False)
        event = await file_changes_hook(event, self.ctx)

        # A reminder with a real unified diff was injected
        self.assertEqual(len(event.context_post), 1)
        reminder = event.context_post[0].text
        self.assertIn("<system-reminder>", reminder)
        self.assertIn(str(path.resolve()), reminder)
        self.assertIn("-line two", reminder)
        self.assertIn("+line 2 EDITED", reminder)

        # The tracker was refreshed: the file is fresh again and a second
        # hook invocation reports nothing.
        self.assertEqual(self.ctx.file_state.status(path.resolve()), "fresh")
        event2 = await file_changes_hook(
            UserPromptEvent(prompt="another", is_first_prompt=False), self.ctx
        )
        self.assertEqual(event2.context_post, [])

    async def test_prompt_hook_shows_pure_deletions(self):
        path = (self.base_path / "deletion.txt")
        path.write_text("keep me\ndelete me\nkeep me too", encoding="utf-8")

        await self._read(path)
        _modify_externally(path, "keep me\nkeep me too")

        event = await file_changes_hook(
            UserPromptEvent(prompt="q", is_first_prompt=False), self.ctx
        )

        # Unlike the mini-agent line-range renderer, a pure deletion is visible
        self.assertEqual(len(event.context_post), 1)
        self.assertIn("-delete me", event.context_post[0].text)

    async def test_prompt_hook_ignores_unchanged_files(self):
        path = self.base_path / "unchanged.txt"
        path.write_text("nothing happens", encoding="utf-8")

        await self._read(path)

        event = await file_changes_hook(
            UserPromptEvent(prompt="q", is_first_prompt=False), self.ctx
        )
        self.assertEqual(event.context_post, [])

    # ---------------------------------------------------------
    # DELETED FILES
    # ---------------------------------------------------------

    async def test_deleted_file_is_forgotten_and_requires_reread(self):
        path = (self.base_path / "doomed.txt")
        path.write_text("about to be deleted", encoding="utf-8")
        resolved = path.resolve()

        await self._read(path)
        path.unlink()

        # The prompt hook silently forgets deleted files (no reminder)
        event = await file_changes_hook(
            UserPromptEvent(prompt="q", is_first_prompt=False), self.ctx
        )
        self.assertEqual(event.context_post, [])
        self.assertNotIn(resolved, self.ctx.file_state.known)

        # If the file reappears with different content, the old read must
        # not count: writing is only allowed because it's now a new file
        # creation path... so recreate it and verify the gate demands a Read.
        path.write_text("resurrected with new content", encoding="utf-8")
        result = await _write_impl({"file_path": str(path), "content": "overwrite"}, self.ctx)
        self.assertIsInstance(result, ToolFailure)
        self.assertIn("has not been read yet", result.error_message)

    # ---------------------------------------------------------
    # FAILED READS MUST NOT UNLOCK WRITES
    # ---------------------------------------------------------

    async def test_failed_oversized_read_does_not_unlock_write(self):
        path = self.base_path / "huge.txt"
        path.write_text("A" * 150, encoding="utf-8")

        with patch.object(fs, "MAX_FILE_BYTES", 100):
            read_result = await self._read(path)
            self.assertIsInstance(read_result, ToolFailure)

        # The failed read must not have registered the file as known
        self.assertNotIn(path.resolve(), self.ctx.file_state.known)

        result = await _write_impl({"file_path": str(path), "content": "sneaky overwrite"}, self.ctx)
        self.assertIsInstance(result, ToolFailure)
        self.assertIn("has not been read yet", result.error_message)

    async def test_failed_token_limit_read_does_not_unlock_write(self):
        path = self.base_path / "tokens.txt"
        path.write_text("B" * 50, encoding="utf-8")  # ~12 "tokens"

        with patch.object(fs, "MAX_TOKENS", 10):
            read_result = await self._read(path)
            self.assertIsInstance(read_result, ToolFailure)

        self.assertNotIn(path.resolve(), self.ctx.file_state.known)

    # ---------------------------------------------------------
    # SUB-AGENT ISOLATION
    # ---------------------------------------------------------

    async def test_clone_for_subagent_starts_empty(self):
        path = self.base_path / "parent_only.txt"
        path.write_text("read by the parent", encoding="utf-8")

        await self._read(path)
        self.assertIn(path.resolve(), self.ctx.file_state.known)

        sub_ctx = self.ctx.clone_for_subagent()

        # Fresh, empty, independent tracker; same invocation facts
        self.assertIsNot(sub_ctx.file_state, self.ctx.file_state)
        self.assertEqual(sub_ctx.file_state.known, {})
        self.assertEqual(sub_ctx.workspace, self.ctx.workspace)
        self.assertEqual(sub_ctx.cwd, self.ctx.cwd)

        # The sub-agent cannot write a file only the parent has read
        result = await _write_impl({"file_path": str(path), "content": "sub overwrite"}, sub_ctx)
        self.assertIsInstance(result, ToolFailure)
        self.assertIn("has not been read yet", result.error_message)

        # And the sub-agent's reads don't leak back into the parent
        await _read_impl({"file_path": str(path)}, sub_ctx)
        sub_only = self.base_path / "sub_only.txt"
        sub_only.write_text("read by the sub-agent", encoding="utf-8")
        await _read_impl({"file_path": str(sub_only)}, sub_ctx)
        self.assertNotIn(sub_only.resolve(), self.ctx.file_state.known)

    # ---------------------------------------------------------
    # TRACKER UNIT BEHAVIOR
    # ---------------------------------------------------------

    def test_status_unknown_for_untracked_path(self):
        tracker = FileStateTracker()
        self.assertEqual(tracker.status(self.base_path / "never_seen.txt"), "unknown")

    def test_status_stale_for_deleted_tracked_file(self):
        tracker = FileStateTracker()
        path = self.base_path / "gone.txt"
        path.write_text("here today", encoding="utf-8")
        tracker.record(path, ["here today"])

        path.unlink()
        self.assertEqual(tracker.status(path), "stale")

    def test_record_of_missing_file_forgets_it(self):
        tracker = FileStateTracker()
        path = self.base_path / "phantom.txt"
        path.write_text("x", encoding="utf-8")
        tracker.record(path, ["x"])

        path.unlink()
        tracker.record(path, ["x"])  # stat fails -> entry dropped
        self.assertNotIn(path, tracker.known)


if __name__ == "__main__":
    unittest.main()
