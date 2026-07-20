from tools.registry import ToolRegistry
from tools.filesystem import register_fsystem_tools
from tools.filesearch import register_fsearch_tools
from tools.tasks import register_tasks_tools
from tools.shell import register_shell_tools
from tools.plan import register_plan_tools
from sessioncontext import InvocationContext

def create_core_registry(ctx: InvocationContext) -> ToolRegistry:
    registry = ToolRegistry()

    register_fsystem_tools(registry, ctx)
    register_fsearch_tools(registry, ctx)
    register_tasks_tools(registry, ctx)
    register_shell_tools(registry, ctx)
    register_plan_tools(registry, ctx)

    return registry