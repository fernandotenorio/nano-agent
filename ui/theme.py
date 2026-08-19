# ui/theme.py
"""
User-customizable appearance, loaded from `~/.prisma/ui.json`.

The theme is deliberately a plain data structure with no rendering library
behind it: the Textual front-end turns it into CSS variables, and it stays
unit-testable on its own.

One optional file is consulted, and it is the user's own rather than a
project's: how the interface looks is a preference of whoever is sitting at it,
not a property of the code being worked on.

A missing, malformed, or partly nonsensical file must never take the UI down
with it: unusable values are logged and the default is kept.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import AppConfig

UI_CONFIG_FILENAME = "ui.json"


@dataclass(frozen=True)
class BarTheme:
    """The fixed header and footer strips."""
    background: str
    text: str


@dataclass(frozen=True)
class BlockTheme:
    """A transcript block: user, assistant, or reasoning.

    `accent` colors the icon, the caption, and the left border together. That
    repetition is the point: one glance at the left edge identifies the
    speaker, without every block needing its own palette.
    """
    icon: str
    caption: str
    accent: str
    background: str
    text: str


@dataclass(frozen=True)
class ToolTheme:
    """Tool call blocks, including their success/failure markers."""
    success_icon: str
    error_icon: str
    success: str
    error: str
    accent: str
    background: str
    text: str


@dataclass(frozen=True)
class InputTheme:
    """The prompt input, which reads as focused or not at a glance."""
    border: str
    border_blurred: str
    background: str
    text: str


@dataclass(frozen=True)
class UITheme:
    header: BarTheme
    footer: BarTheme
    user: BlockTheme
    assistant: BlockTheme
    reasoning: BlockTheme
    tool: ToolTheme
    input: InputTheme


# Translucent whites rather than fixed greys: they lighten whatever screen
# background is in effect, so a block stays distinguishable under any palette.
_LIFTED = "rgba(255,255,255,0.05)"
_BAR = "rgba(255,255,255,0.08)"

DEFAULT_THEME = UITheme(
    header=BarTheme(background=_BAR, text="#d0d0d0"),
    footer=BarTheme(background=_BAR, text="#d0d0d0"),
    user=BlockTheme(
        icon="\N{BUST IN SILHOUETTE}",
        caption="User",
        accent="cyan",
        background=_LIFTED,
        text="white",
    ),
    assistant=BlockTheme(
        icon="\N{ROBOT FACE}",
        caption="Assistant",
        accent="orange",
        background="transparent",
        text="white",
    ),
    reasoning=BlockTheme(
        icon="\N{BRAIN}",
        caption="AI Reasoning",
        accent="white",
        background=_LIFTED,
        text="white",
    ),
    tool=ToolTheme(
        success_icon="\N{WHITE HEAVY CHECK MARK}",
        error_icon="\N{CROSS MARK}",
        success="#4ade80",
        error="#f87171",
        accent="#6b7280",
        background="transparent",
        text="white",
    ),
    input=InputTheme(
        border="cyan",
        border_blurred="#3f3f46",
        background=_LIFTED,
        text="white",
    ),
)


# Colors are pasted straight into a stylesheet, so anything unrecognisable is
# rejected here rather than at render time, where it would abort the whole app.
_COLOR_PATTERN = re.compile(
    r"""^(
        transparent
        | \#[0-9a-fA-F]{3,8}
        | [a-zA-Z][a-zA-Z0-9_-]*          # CSS color name, or a Textual token
        | \$[a-zA-Z][a-zA-Z0-9_-]*
        | rgba?\([0-9.,\s%]+\)
        | [a-zA-Z]+\s+[0-9]{1,3}%         # e.g. "cyan 20%"
    )$""",
    re.VERBOSE,
)

# Fields holding text rather than a color: free-form, no validation.
_TEXT_FIELDS = frozenset({"icon", "caption", "success_icon", "error_icon"})


def _is_color(value: str) -> bool:
    return bool(_COLOR_PATTERN.match(value.strip()))


def _read_config(path: Path) -> dict[str, Any]:
    """Loads one ui.json, returning {} for anything unusable."""
    if not path.exists():
        return {}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logging.warning("Ignoring UI config %s: %s", path, e)
        return {}

    if not isinstance(data, dict):
        logging.warning("Ignoring UI config %s: expected a JSON object", path)
        return {}

    return data


def _apply_section(section: Any, overrides: Any, name: str) -> Any:
    """Returns `section` with the valid entries of `overrides` applied."""
    if not isinstance(overrides, dict):
        logging.warning("Ignoring UI config section %r: expected an object", name)
        return section

    known = {field.name for field in dataclasses.fields(section)}
    changes: dict[str, str] = {}

    for key, value in overrides.items():
        if key not in known:
            logging.warning("Ignoring unknown UI config key %r in section %r", key, name)
            continue
        if not isinstance(value, str):
            logging.warning("Ignoring UI config key %r in section %r: expected a string", key, name)
            continue
        if key not in _TEXT_FIELDS and not _is_color(value):
            logging.warning("Ignoring UI config key %r in section %r: %r is not a color", key, name, value)
            continue
        changes[key] = value

    return dataclasses.replace(section, **changes) if changes else section


def theme_from_dict(data: dict[str, Any], base: UITheme = DEFAULT_THEME) -> UITheme:
    """Builds a theme by laying `data` over `base`."""
    sections = {field.name for field in dataclasses.fields(base)}
    changes: dict[str, Any] = {}

    for name, overrides in data.items():
        if name not in sections:
            logging.warning("Ignoring unknown UI config section %r", name)
            continue
        changes[name] = _apply_section(getattr(base, name), overrides, name)

    return dataclasses.replace(base, **changes) if changes else base


def load_ui_theme(app_config: AppConfig) -> UITheme:
    """Loads the theme for this session: defaults, then the user's own file."""
    return theme_from_dict(_read_config(app_config.home_config_dir / UI_CONFIG_FILENAME))


def css_variables(theme: UITheme) -> dict[str, str]:
    """Exposes the theme's colors to the stylesheet as `$prisma-*` variables.

    Icons and captions are not here: they are content, and the widgets read
    them from the theme directly.
    """
    return {
        "prisma-header-bg": theme.header.background,
        "prisma-header-text": theme.header.text,
        "prisma-footer-bg": theme.footer.background,
        "prisma-footer-text": theme.footer.text,
        "prisma-user-accent": theme.user.accent,
        "prisma-user-bg": theme.user.background,
        "prisma-user-text": theme.user.text,
        "prisma-assistant-accent": theme.assistant.accent,
        "prisma-assistant-bg": theme.assistant.background,
        "prisma-assistant-text": theme.assistant.text,
        "prisma-reasoning-accent": theme.reasoning.accent,
        "prisma-reasoning-bg": theme.reasoning.background,
        "prisma-reasoning-text": theme.reasoning.text,
        "prisma-tool-accent": theme.tool.accent,
        "prisma-tool-bg": theme.tool.background,
        "prisma-tool-text": theme.tool.text,
        "prisma-tool-success": theme.tool.success,
        "prisma-tool-error": theme.tool.error,
        "prisma-input-border": theme.input.border,
        "prisma-input-border-blurred": theme.input.border_blurred,
        "prisma-input-bg": theme.input.background,
        "prisma-input-text": theme.input.text,
    }
