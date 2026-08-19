# sessions.py
"""
Where a session's files live on disk.

Every session gets a directory of its own, under a directory named after the
project it belongs to:

    <home>/.prisma/projects/<slug>/sessions/<session_id>/
        <session_id>.jsonl              the main transcript
        meta.json                       who this session is, and what it is called
        subagents/<type>_<runid>.jsonl  one per sub-agent run

Two decisions shape this module.

First, the project is identified by the *workspace root*, not the working
directory: the workspace is the project boundary, so every session belongs to
one bucket no matter which subdirectory it was launched from. The working
directory is not lost — it is recorded inside the transcript's system message.

Second, a session directory holds exactly one session. That is what lets
`--resume` find a sub-agent transcript by looking in a directory rather than by
matching filenames against the main transcript's name, which is how sub-agents
used to be told apart from every other session sharing one folder.

The session id is immutable. Anything a user may later want to change (a title)
belongs in meta.json, so that renaming a session never moves a directory and
never invalidates an id someone wrote down.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from config import AppConfig

SESSIONS_DIR_NAME = "sessions"
SUBAGENTS_DIR_NAME = "subagents"
META_FILENAME = "meta.json"
TRANSCRIPT_SUFFIX = ".jsonl"

# Everything a filename cannot safely hold, collapsed to a single dash. '+' so
# a run of separators or spaces becomes one dash rather than several.
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")

SLUG_MAX_CHARS: int = 60
"""Cap on the readable part of a project slug. A single path component is
limited to 255 bytes on most filesystems, and the whole path matters on Windows,
so a deep workspace must not spend it all on one directory name."""

HASH_CHARS: int = 8
"""Length of the digest appended to a project slug."""

RUN_ID_CHARS: int = 6
"""Length of the random suffix distinguishing one run from another."""

TIMESTAMP_FORMAT = "%Y-%m-%d_%H-%M-%S"


def project_slug(workspace: Path) -> str:
    """Builds the directory name holding every session of one workspace.

    The readable part is the path with its separators turned into dashes, which
    is what makes a projects directory browsable. It cannot be the whole answer:
    dash-replacement is lossy (both '/a/b-c' and '/a-b/c' read as 'a-b-c'), and
    a long or non-ASCII path can leave nothing usable behind. A digest of the
    resolved path is therefore appended, and it — not the readable part — is
    what actually keeps two projects apart.
    """
    resolved = workspace.expanduser().resolve()
    digest = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:HASH_CHARS]

    # A drive letter's colon cannot appear in a filename at all. Leading
    # separators would otherwise open the name with a dash, and a trailing dot
    # is illegal on Windows.
    readable = _UNSAFE.sub("-", str(resolved).replace(":", "")).strip("-.")
    readable = readable[:SLUG_MAX_CHARS].rstrip("-.")

    # A workspace at the filesystem root leaves no readable part to use.
    return f"{readable}-{digest}" if readable else f"root-{digest}"


def new_run_id() -> str:
    return uuid.uuid4().hex[:RUN_ID_CHARS]


def new_session_id(now: datetime | None = None) -> str:
    """Builds an id for a session starting now.

    The timestamp leads so that ordering by name is ordering by creation, and
    the random suffix is what makes the id unique: two sessions started in the
    same second would otherwise share a directory, and their two conversations
    would interleave into one transcript.
    """
    stamp = (now or datetime.now()).strftime(TIMESTAMP_FORMAT)
    return f"{stamp}-{new_run_id()}"


@dataclass(frozen=True)
class SessionPaths:
    """Every file belonging to one session. Pure path math: nothing on disk."""

    session_id: str
    directory: Path

    @property
    def transcript(self) -> Path:
        return self.directory / f"{self.session_id}{TRANSCRIPT_SUFFIX}"

    @property
    def meta_file(self) -> Path:
        return self.directory / META_FILENAME

    @property
    def subagents_dir(self) -> Path:
        return self.directory / SUBAGENTS_DIR_NAME

    @property
    def exists(self) -> bool:
        """True once the session has a transcript to resume."""
        return self.transcript.is_file()


def project_dir(app_config: AppConfig, workspace: Path) -> Path:
    return app_config.projects_dir / project_slug(workspace)


def sessions_dir(app_config: AppConfig, workspace: Path) -> Path:
    return project_dir(app_config, workspace) / SESSIONS_DIR_NAME


def session_for(app_config: AppConfig, workspace: Path, session_id: str) -> SessionPaths:
    """Locates a named session. Says nothing about whether it exists."""
    return SessionPaths(
        session_id=session_id,
        directory=sessions_dir(app_config, workspace) / session_id,
    )


def new_session(
    app_config: AppConfig,
    workspace: Path,
    now: datetime | None = None,
) -> SessionPaths:
    """Allocates a fresh session.

    Creates nothing: the directory appears when the transcript is first written
    (see `Transcript.append`), so a session that never records anything leaves
    no empty directory behind.
    """
    return session_for(app_config, workspace, new_session_id(now))


def list_sessions(app_config: AppConfig, workspace: Path) -> list[SessionPaths]:
    """Every resumable session of this workspace, oldest first.

    A directory only counts once it holds the transcript its name promises.
    That skips a half-created session, and anything else that found its way
    into the sessions directory.
    """
    root = sessions_dir(app_config, workspace)

    try:
        entries = sorted(root.iterdir())
    except OSError:
        return []

    found = [
        SessionPaths(session_id=entry.name, directory=entry)
        for entry in entries
        if entry.is_dir()
    ]

    return [paths for paths in found if paths.exists]


def list_session_ids(app_config: AppConfig, workspace: Path) -> list[str]:
    return [paths.session_id for paths in list_sessions(app_config, workspace)]


def latest_session(app_config: AppConfig, workspace: Path) -> SessionPaths | None:
    """The session to pick up when the user does not name one.

    Ordered by when the transcript was last written rather than by the
    timestamp in the id: resuming a week-old conversation makes it the one you
    are working in, and that is the one `--continue` should find next time.
    """
    def last_used(paths: SessionPaths) -> tuple[float, str]:
        try:
            mtime = paths.transcript.stat().st_mtime
        except OSError:
            mtime = 0.0
        return mtime, paths.session_id

    return max(list_sessions(app_config, workspace), key=last_used, default=None)


def subagent_transcript_path(
    main_transcript: Path,
    subagent_type: str,
    run_id: str | None = None,
) -> Path:
    """Names the transcript for one sub-agent run of a session.

    Derived from the main transcript because that is what the agent loop has in
    hand, and its parent is the session directory. The type travels in the
    filename so a resumed session can tell which sub-agent spent what, and the
    run id keeps two runs of one type apart.
    """
    safe_type = _UNSAFE.sub("-", subagent_type).strip("-.") or "subagent"
    run = run_id or new_run_id()

    return (
        main_transcript.parent
        / SUBAGENTS_DIR_NAME
        / f"{safe_type}_{run}{TRANSCRIPT_SUFFIX}"
    )


def ensure_meta(paths: SessionPaths, workspace: Path) -> str | None:
    """Records what this session is, if it has not been recorded already.

    Written once, at creation. Resuming must leave it alone: it is where a
    user-supplied title will live, and the whole point of keeping the title out
    of the directory name is that nothing else may overwrite it.

    The slug is lossy, so the real workspace path is kept here too — it is the
    only way back from a directory name to the project it stands for.

    Returns a warning to show the user if the file could not be written. A
    session is perfectly usable without it, so this never raises.
    """
    if paths.meta_file.exists():
        return None

    meta = {
        "session_id": paths.session_id,
        "workspace": str(workspace),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "title": None,
    }

    try:
        paths.directory.mkdir(parents=True, exist_ok=True)
        paths.meta_file.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except OSError as e:
        logging.warning("Could not write session metadata %s: %s", paths.meta_file, e)
        return f"Could not write session metadata to {paths.meta_file}: {e}"

    return None


def read_meta(paths: SessionPaths) -> dict:
    """Reads a session's metadata, treating anything unusable as absent."""
    try:
        data = json.loads(paths.meta_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}

    return data if isinstance(data, dict) else {}
