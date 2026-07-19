import logging
import os
from pathlib import Path

from config import AppConfig
from textwrap import dedent
from typedefs import SystemMessage
from sessioncontext import InvocationContext
from environment import get_environment_details


# _DEFAULT_USER_INSTRUCTIONS should contain only tone/style guidance.
# This is the only layer that can be replaced by the user via --system-prompt-file flag.
_DEFAULT_USER_INSTRUCTIONS = """\
- Be concise and direct in your responses.
"""

def _load_optional_text(path: Path) -> str | None:
    """
    Attempts to load a UTF-8 text file.

    Returns:
        - stripped contents if successful and non-empty
        - None if the file doesn't exist, is empty, or fails to load
    """
    if not path.exists():
        return None

    try:
        text = path.read_text(encoding="utf-8").strip()
        return text or None
    except Exception as e:
        logging.warning("Failed to read %s: %s", path, e)
        return None


def _load_core_instructions(app_config: AppConfig):
    """
    Contains anything related to tools, capabilities, or operational behavior.
    It is never removed, replaced, or skipped by any CLI flag or user file.
    """

    core_sys_prompt = dedent(f"""\
    You are {app_config.app_name.capitalize()}, an interactive coding agent designed to help users with their software engineering tasks.
    You can read files, execute shell commands, and orchestrate sub-agents.

    By default:

    - Always use tools when you need to gather information or make changes.
    - Prioritize correctness, robustness, maintainability, security, and clear architecture over brevity. Never intentionally simplify implementations or omit important logic.
    - Make reasonable assumptions and continue unless critical information is missing. If a complete implementation exceeds the response limit, continue in subsequent responses instead of reducing quality or completeness.
    """)
    return core_sys_prompt


def _load_user_instructions(args) -> str:
    """
    Loads the custom user instructions layer.

    If --system-prompt-file is supplied, it completely replaces this
    user-customizable system instruction. If loading fails, the default one is used.
    """
    if not args.system_prompt_file:
        return _DEFAULT_USER_INSTRUCTIONS

    path = Path(args.system_prompt_file)
    text = _load_optional_text(path)

    if text is None:
        logging.warning(
            "Could not load instructions '%s'. Falling back to default instructions.",
            path,
        )
        return _DEFAULT_USER_INSTRUCTIONS
    return text


def build_system_prompt(app_config: AppConfig, cwd: Path, ctx: InvocationContext, args) -> SystemMessage:
    parts: list[str] = []

    # Immutable core prompt
    parts.append(_load_core_instructions(app_config))

    # Custom user system instructions
    parts.append(_load_user_instructions(args))

    # Global SYSTEM.md
    if not args.no_global_system_prompt_file:
        global_system = app_config.global_system_prompt_file()

        text = _load_optional_text(global_system)
        if text:
            parts.append(text)

    # Project SYSTEM.md
    if not args.no_proj_system_prompt_file:
        project_system = app_config.project_system_prompt_file(cwd)

        text = _load_optional_text(project_system)
        if text:
            parts.append(text)

    # Environment information
    parts.append(get_environment_details(ctx))

    return SystemMessage(
        content="\n\n---\n\n".join(parts)
    )