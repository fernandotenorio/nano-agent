from tools.registry import ToolRegistry
from tools.filesystem import register_fs_tools
from tools.tasks import register_tasks_tools
from tools.bash import register_bash_tools

def create_core_registry() -> ToolRegistry:
    registry = ToolRegistry()

    register_fs_tools(registry)
    register_tasks_tools(registry)
    register_bash_tools(registry)

    return registry