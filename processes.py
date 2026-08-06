# processes.py
"""
Signalling child processes we have stopped waiting for.

Both places that spawn a subprocess (the Shell tool and the ripgrep backend)
give up on it the same way: stop reading, then make sure it is gone. The race
is the same in both, too. A process can exit on its own between the moment we
decide to kill it and the moment we do, and once asyncio has reaped it and
closed the transport, `terminate()` and `kill()` raise ProcessLookupError.

That exception means the process is already dead, which is exactly what the
caller was asking for, so it is swallowed here rather than handled at every
call site.
"""

from __future__ import annotations

import asyncio


def terminate_quietly(process: asyncio.subprocess.Process) -> None:
    """Asks a process to stop, tolerating one that already has."""
    try:
        process.terminate()
    except ProcessLookupError:
        pass


def kill_quietly(process: asyncio.subprocess.Process) -> None:
    """Kills a process we no longer want, tolerating one that already exited."""
    try:
        process.kill()
    except ProcessLookupError:
        pass
