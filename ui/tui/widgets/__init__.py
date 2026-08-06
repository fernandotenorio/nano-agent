# Widgets making up the Textual front-end.

from ui.tui.widgets.approval import PlanApprovalBlock, ShellApprovalBlock
from ui.tui.widgets.blocks import (
    MessageBlock,
    NoticeBlock,
    ReasoningBlock,
    SessionBanner,
    ToolBlock,
)
from ui.tui.widgets.chrome import FooterBar, HeaderBar
from ui.tui.widgets.prompt import PromptArea, PromptInput
from ui.tui.widgets.spinner import SpinnerLine

__all__ = [
    "FooterBar",
    "HeaderBar",
    "MessageBlock",
    "NoticeBlock",
    "PlanApprovalBlock",
    "PromptArea",
    "PromptInput",
    "ReasoningBlock",
    "SessionBanner",
    "ShellApprovalBlock",
    "SpinnerLine",
    "ToolBlock",
]
