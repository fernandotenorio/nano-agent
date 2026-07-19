from datetime import datetime
import platform
import os
from textwrap import dedent
from sessioncontext import InvocationContext

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
        Workspace root is a Git repo: {'Yes' if ctx.workspace_is_git_repo else 'No'}

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