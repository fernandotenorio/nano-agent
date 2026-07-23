# tools/plan.py
from tools.registry import ToolRegistry, ToolReturnType
from typedefs import PlanApprovalCallback, ToolFailure
from typing import Any
from sessioncontext import InvocationContext

async def _plan_impl(kwargs: dict[str, Any], ctx: InvocationContext) -> ToolReturnType:
    """Explicitly submits a plan for user approval."""
    plan_summary = kwargs.get("plan_summary")

    # Check truthiness, not just key presence: an explicit empty/whitespace
    # string must not reach the user as an empty plan.
    if not isinstance(plan_summary, str) or not plan_summary.strip():
        return ToolFailure(error_message="Error: plan_summary is required and cannot be empty.")

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