import json
import os
import re
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import PropertyMock, patch

from config import AppConfig
from sessions import (
    HASH_CHARS,
    SLUG_MAX_CHARS,
    SessionPaths,
    ensure_meta,
    latest_session,
    list_session_ids,
    new_session,
    new_session_id,
    project_slug,
    read_meta,
    session_for,
    sessions_dir,
    subagent_transcript_path,
)


class TestProjectSlug(unittest.TestCase):
    """
    Test Suite for turning a workspace path into a directory name (sessions.project_slug)

    The slug is browsable but lossy; the digest is what has to be trusted to
    keep two projects apart. These check that the readable part is a legal
    filename on both platforms and that the digest survives every way the
    readable part can be mangled.
    """

    def test_separators_become_dashes(self):
        slug = project_slug(Path(tempfile.gettempdir()))

        self.assertNotIn(os.sep, slug)
        self.assertNotIn("/", slug)

    def test_drive_colon_is_dropped(self):
        """A colon cannot appear in a filename at all."""
        slug = project_slug(Path.cwd())

        self.assertNotIn(":", slug)

    def test_ends_with_hash_of_resolved_path(self):
        base = Path(tempfile.gettempdir()).resolve()
        slug = project_slug(base)

        digest = slug.rsplit("-", 1)[-1]
        self.assertEqual(len(digest), HASH_CHARS)
        self.assertRegex(digest, r"^[0-9a-f]+$")

    def test_deterministic(self):
        base = Path(tempfile.gettempdir())

        self.assertEqual(project_slug(base), project_slug(base))

    def test_relative_path_resolves_to_same_slug_as_absolute(self):
        """Two spellings of one directory are one project."""
        with tempfile.TemporaryDirectory() as tmp:
            absolute = Path(tmp).resolve()
            roundabout = absolute / "sub" / ".."
            (absolute / "sub").mkdir()

            self.assertEqual(project_slug(absolute), project_slug(roundabout))

    def test_dash_ambiguity_is_broken_by_the_hash(self):
        """'/a/b-c' and '/a-b/c' both read as 'a-b-c'; they are still distinct."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            first = base / "a" / "b-c"
            second = base / "a-b" / "c"
            first.mkdir(parents=True)
            second.mkdir(parents=True)

            self.assertNotEqual(project_slug(first), project_slug(second))

    def test_unsafe_characters_collapse(self):
        noisy = Path(Path.cwd().anchor) / "my project (v2)"

        slug = project_slug(noisy)

        self.assertRegex(slug, r"^[A-Za-z0-9._-]+$")
        self.assertIn("my-project-v2", slug)

    def test_length_is_capped_but_hash_survives(self):
        """A deep workspace must not spend a whole path budget on one name."""
        deep = Path(Path.cwd().anchor)
        for _ in range(12):
            deep = deep / ("directory" + "x" * 10)

        slug = project_slug(deep)

        self.assertLessEqual(len(slug), SLUG_MAX_CHARS + 1 + HASH_CHARS)
        self.assertEqual(len(slug.rsplit("-", 1)[-1]), HASH_CHARS)

    def test_filesystem_root_still_gets_a_name(self):
        """The readable part can be empty; the directory name cannot."""
        root = Path(Path.cwd().anchor)

        slug = project_slug(root)

        self.assertTrue(slug)
        self.assertFalse(slug.startswith("-"))
        self.assertTrue(slug.startswith("root-") or re.match(r"^[A-Za-z0-9._]", slug))

    def test_no_leading_or_trailing_separator_artifacts(self):
        """A leading POSIX separator must not open the name with a dash, and a
        trailing dot is illegal on Windows."""
        slug = project_slug(Path(tempfile.gettempdir()))

        self.assertFalse(slug.startswith("-"))
        self.assertFalse(slug.startswith("."))
        self.assertFalse(slug.endswith("."))
        self.assertFalse(slug.endswith("-"))


class TestSessionId(unittest.TestCase):
    """
    Test Suite for session id generation (sessions.new_session_id)
    """

    def test_shape(self):
        session_id = new_session_id(datetime(2026, 8, 19, 16, 25, 3))

        self.assertRegex(session_id, r"^2026-08-19_16-25-03-[0-9a-f]{6}$")

    def test_two_ids_in_the_same_second_differ(self):
        """Otherwise two sessions started together would share a transcript."""
        moment = datetime(2026, 8, 19, 16, 25, 3)

        self.assertNotEqual(new_session_id(moment), new_session_id(moment))

    def test_name_order_is_creation_order(self):
        earlier = new_session_id(datetime(2026, 8, 19, 16, 25, 3))
        later = new_session_id(datetime(2026, 8, 19, 16, 25, 4))
        next_day = new_session_id(datetime(2026, 8, 20, 0, 0, 0))

        self.assertEqual(sorted([next_day, later, earlier]), [earlier, later, next_day])

    def test_id_is_a_legal_filename(self):
        self.assertRegex(new_session_id(), r"^[A-Za-z0-9._-]+$")


class SessionLayoutTestCase(unittest.TestCase):
    """Shared plumbing: a fake home so nothing touches the real ~/.prisma."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.temp_dir.name).resolve()

        self.fake_home = self.base_path / "fake_home" / ".prisma"
        self.fake_home.mkdir(parents=True)

        self.workspace = self.base_path / "project"
        self.workspace.mkdir()

        self.app_config = AppConfig(app_name="prisma", app_dir_name=".prisma")

        patcher = patch("config.AppConfig.home_config_dir", new_callable=PropertyMock)
        self.mock_home = patcher.start()
        self.mock_home.return_value = self.fake_home
        self.addCleanup(patcher.stop)

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_session(self, session_id: str, text: str = "{}\n") -> SessionPaths:
        paths = session_for(self.app_config, self.workspace, session_id)
        paths.directory.mkdir(parents=True, exist_ok=True)
        paths.transcript.write_text(text, encoding="utf-8")
        return paths


class TestSessionLayout(SessionLayoutTestCase):
    """
    Test Suite for the shape of a session directory (sessions.SessionPaths)
    """

    def test_paths_hang_off_the_projects_directory(self):
        paths = new_session(self.app_config, self.workspace)

        self.assertEqual(
            paths.directory.parent.parent,
            self.fake_home / "projects" / project_slug(self.workspace),
        )
        self.assertEqual(paths.directory.parent.name, "sessions")

    def test_transcript_is_named_after_the_session(self):
        paths = session_for(self.app_config, self.workspace, "abc")

        self.assertEqual(paths.transcript, paths.directory / "abc.jsonl")
        self.assertEqual(paths.meta_file, paths.directory / "meta.json")
        self.assertEqual(paths.subagents_dir, paths.directory / "subagents")

    def test_new_session_touches_no_disk(self):
        """The directory appears on the first transcript write, so a session
        that records nothing leaves nothing behind."""
        paths = new_session(self.app_config, self.workspace)

        self.assertFalse(paths.directory.exists())
        self.assertFalse((self.fake_home / "projects").exists())
        self.assertFalse(paths.exists)

    def test_workspace_decides_the_project_not_cwd(self):
        """Launching from a subdirectory must land in the same bucket."""
        subdir = self.workspace / "src" / "backend"
        subdir.mkdir(parents=True)

        self.assertNotEqual(
            sessions_dir(self.app_config, self.workspace),
            sessions_dir(self.app_config, subdir),
        )
        self.assertEqual(
            sessions_dir(self.app_config, self.workspace),
            sessions_dir(self.app_config, self.workspace / "src" / ".."),
        )


class TestSubagentTranscriptPath(SessionLayoutTestCase):
    """
    Test Suite for naming a sub-agent's transcript
    (sessions.subagent_transcript_path)
    """

    def test_lands_in_the_sessions_subagents_directory(self):
        paths = session_for(self.app_config, self.workspace, "abc")

        sub = subagent_transcript_path(paths.transcript, "code-reviewer", "123456")

        self.assertEqual(sub, paths.subagents_dir / "code-reviewer_123456.jsonl")

    def test_two_runs_of_one_type_do_not_collide(self):
        paths = session_for(self.app_config, self.workspace, "abc")

        first = subagent_transcript_path(paths.transcript, "explore")
        second = subagent_transcript_path(paths.transcript, "explore")

        self.assertNotEqual(first, second)
        self.assertEqual(first.parent, second.parent)

    def test_run_id_shape_is_what_rehydration_looks_for(self):
        sub = subagent_transcript_path(Path("/s/abc.jsonl"), "explore")

        self.assertRegex(sub.name, r"^explore_[0-9a-f]{6}\.jsonl$")

    def test_type_is_made_filename_safe(self):
        """A sub-agent type comes from the model, so it cannot be trusted to be
        a legal path component."""
        sub = subagent_transcript_path(Path("/s/abc.jsonl"), "../../evil type", "123456")

        self.assertEqual(sub.name, "evil-type_123456.jsonl")
        self.assertEqual(sub.parent, Path("/s/subagents"))


class TestSessionResolution(SessionLayoutTestCase):
    """
    Test Suite for finding an existing session
    (sessions.list_session_ids, sessions.latest_session)
    """

    def test_no_sessions_yet(self):
        self.assertEqual(list_session_ids(self.app_config, self.workspace), [])
        self.assertIsNone(latest_session(self.app_config, self.workspace))

    def test_lists_sessions_oldest_first(self):
        self.write_session("2026-08-19_16-25-03-aaaaaa")
        self.write_session("2026-08-18_09-00-00-bbbbbb")

        self.assertEqual(
            list_session_ids(self.app_config, self.workspace),
            ["2026-08-18_09-00-00-bbbbbb", "2026-08-19_16-25-03-aaaaaa"],
        )

    def test_directory_without_its_transcript_is_not_resumable(self):
        """A half-created session must not be offered as one to continue."""
        empty = session_for(self.app_config, self.workspace, "halfmade")
        empty.subagents_dir.mkdir(parents=True)

        self.assertEqual(list_session_ids(self.app_config, self.workspace), [])
        self.assertIsNone(latest_session(self.app_config, self.workspace))

    def test_stray_file_in_the_sessions_directory_is_ignored(self):
        root = sessions_dir(self.app_config, self.workspace)
        root.mkdir(parents=True)
        (root / "notes.txt").write_text("hello", encoding="utf-8")

        self.assertEqual(list_session_ids(self.app_config, self.workspace), [])

    def test_latest_follows_last_use_not_creation(self):
        """Resuming an old conversation makes it the one --continue finds."""
        old = self.write_session("2026-08-01_09-00-00-aaaaaa")
        new = self.write_session("2026-08-19_16-25-03-bbbbbb")

        os.utime(new.transcript, (1_000_000, 1_000_000))
        os.utime(old.transcript, (2_000_000, 2_000_000))

        latest = latest_session(self.app_config, self.workspace)

        self.assertIsNotNone(latest)
        self.assertEqual(latest.session_id, old.session_id)

    def test_sessions_of_another_workspace_are_invisible(self):
        other = self.base_path / "other_project"
        other.mkdir()
        self.write_session("2026-08-19_16-25-03-aaaaaa")

        self.assertEqual(list_session_ids(self.app_config, other), [])
        self.assertIsNone(latest_session(self.app_config, other))

    def test_session_for_reports_a_missing_session_rather_than_inventing_one(self):
        paths = session_for(self.app_config, self.workspace, "nope")

        self.assertFalse(paths.exists)


class TestSessionMeta(SessionLayoutTestCase):
    """
    Test Suite for session metadata (sessions.ensure_meta)
    """

    def test_written_with_a_null_title(self):
        paths = new_session(self.app_config, self.workspace)

        warning = ensure_meta(paths, self.workspace)

        self.assertIsNone(warning)
        meta = json.loads(paths.meta_file.read_text(encoding="utf-8"))
        self.assertEqual(meta["session_id"], paths.session_id)
        self.assertEqual(meta["workspace"], str(self.workspace))
        self.assertIsNone(meta["title"])
        self.assertTrue(meta["created_at"])

    def test_records_the_real_workspace_path_the_slug_cannot_reverse(self):
        paths = new_session(self.app_config, self.workspace)

        ensure_meta(paths, self.workspace)

        self.assertEqual(read_meta(paths)["workspace"], str(self.workspace))

    def test_an_existing_title_is_never_clobbered(self):
        """Resuming must not undo a rename."""
        paths = self.write_session("2026-08-19_16-25-03-aaaaaa")
        ensure_meta(paths, self.workspace)
        meta = read_meta(paths)
        meta["title"] = "Refactor the session layout"
        paths.meta_file.write_text(json.dumps(meta), encoding="utf-8")

        warning = ensure_meta(paths, self.workspace)

        self.assertIsNone(warning)
        self.assertEqual(read_meta(paths)["title"], "Refactor the session layout")

    def test_unwritable_metadata_is_reported_not_raised(self):
        """A session is usable without its metadata."""
        paths = new_session(self.app_config, self.workspace)

        with self.assertLogs(level="WARNING"), patch(
            "pathlib.Path.write_text", side_effect=OSError("read-only filesystem")
        ):
            warning = ensure_meta(paths, self.workspace)

        self.assertIsNotNone(warning)
        self.assertIn("read-only filesystem", warning)

    def test_unreadable_metadata_reads_as_absent(self):
        paths = self.write_session("2026-08-19_16-25-03-aaaaaa")
        paths.meta_file.write_text("{not json", encoding="utf-8")

        self.assertEqual(read_meta(paths), {})


class TestProjectsDir(SessionLayoutTestCase):
    """
    Test Suite for the global projects directory (config.AppConfig.projects_dir)
    """

    def test_sits_under_the_home_config_dir(self):
        self.assertEqual(self.app_config.projects_dir, self.fake_home / "projects")

    def test_transcripts_are_outside_every_workspace(self):
        """Which is what keeps them out of reach of the agent's own tools."""
        paths = new_session(self.app_config, self.workspace)

        self.assertFalse(paths.transcript.is_relative_to(self.workspace))


if __name__ == "__main__":
    unittest.main()
