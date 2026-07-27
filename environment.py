from datetime import datetime
import platform
import os
from textwrap import dedent
from capabilities import model_warnings
from sessioncontext import InvocationContext


def _git_repo_line(ctx: InvocationContext) -> str:
    """Describes the repo state, including whether git can actually be asked.

    An unusable git widens what every search tool returns, so claiming a plain
    'Yes' here would be worse than saying nothing: it invites the agent to trust
    listings that are quietly including build output.

    Kept to a single line: the caller interpolates this into an indented block
    before dedenting it, and an unindented continuation line would flatten the
    common prefix and defeat the dedent.
    """
    answer = "Yes" if ctx.workspace_is_git_repo else "No"

    warnings = model_warnings(ctx.capabilities) if ctx.capabilities else []
    if not warnings:
        return f"Workspace root is a Git repo: {answer}"

    return f"Workspace root is a Git repo: {answer} (WARNING: {' '.join(warnings)})"


def get_environment_details(ctx: InvocationContext) -> str:
    """Returns a formatted summary of the current execution environment."""
    
    try:
        import psutil
    except ImportError:
        psutil = None

    try:
        now = datetime.now().astimezone()
        cpu_count = os.cpu_count() or "Unknown"

        if psutil:
            total_ram_gb = psutil.virtual_memory().total / (1024**3)
            ram = f"{total_ram_gb:.1f} GB"
        else:
            ram = "Unknown"

        e = dedent(f'''
        <workspace>
        Root: {ctx.workspace}
        Current directory: {ctx.cwd}
        {_git_repo_line(ctx)}

        The workspace root defines the project boundary. Relative paths
        are resolved from the current directory. When using tools, treat
        the workspace root as the top-level location unless the user
        explicitly instructs otherwise.
        </workspace>

        <environment>
        OS: {platform.system()} {platform.release()}
        Architecture: {platform.machine()}        
        Current time: {now.strftime("%Y-%m-%d %H:%M:%S %z")}
        Timezone: {now.tzname() or "Unknown"}
        CPUs: {cpu_count}
        Memory (RAM): {ram}
        </environment>''')
        return e
    except Exception:
        return ''