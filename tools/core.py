from capabilities import find_ripgrep
from tools.registry import ToolRegistry
from tools.filesystem import register_fsystem_tools
from tools.filesearch import register_fsearch_tools
from tools.grep import register_grep_tools
from tools.grep_fallback import register_fallback_grep_tools
from tools.tasks import register_tasks_tools
from tools.shell import register_shell_tools
from tools.plan import register_plan_tools
from sessioncontext import InvocationContext

def create_core_registry(ctx: InvocationContext) -> ToolRegistry:
    registry = ToolRegistry()

    register_fsystem_tools(registry, ctx)
    register_fsearch_tools(registry, ctx)
    register_grep_backend(registry, ctx)
    register_tasks_tools(registry, ctx)
    register_shell_tools(registry, ctx)
    register_plan_tools(registry, ctx)

    return registry


def register_grep_backend(registry: ToolRegistry, ctx: InvocationContext) -> None:
    """Registers whichever Grep engine this machine can run.

    There is always a Grep tool. Which engine backs it is decided once, here,
    from the startup probe; each engine describes its own capabilities, so the
    model never has to reason about what is installed. A context with no probe
    (unit tests, throwaway contexts) looks for ripgrep itself.
    """
    capabilities = ctx.capabilities
    rg = capabilities.ripgrep if capabilities else find_ripgrep()

    if rg:
        register_grep_tools(registry, ctx, rg)
    else:
        register_fallback_grep_tools(registry, ctx)
