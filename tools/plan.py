# tools/plan.py
from tools.registry import ToolRegistry
from typedefs import PlanApprovalCallback
from sessioncontext import InvocationContext

async def _plan_impl(kwargs: dict[str, Any]) -> ToolReturnType:
    """Explicitly submits a plan for user approval."""
    plan_summary = kwargs.get("plan_summary", "No plan provided.")
    return PlanApprovalCallback(plan_summary=plan_summary)

def register_plan_tools(registry: ToolRegistry, ctx: InvocationContext):
    registry.register(
        name="SubmitPlan",
        description="Call this tool when you have finished investigating and have a clear, step-by-step plan to propose to the user.",
        input_schema={
            "type": "object",
            "properties": {
                "plan_summary": {
                    "type": "string",
                    "description": "The detailed step-by-step plan for the user to review."
                }
            },
            "required": ["plan_summary"]
        },
        func=lambda kwargs: _plan_impl(kwargs, ctx),
        is_readonly=True  # MUST be True so it's available in Plan Mode!
    )