from datetime import datetime
import platform
import os
from pathlib import Path
from textwrap import dedent

def get_environment_details() -> str:
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
        <environment>
        Current directory: {Path.cwd()}
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