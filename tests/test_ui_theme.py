import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from config import AppConfig
from ui.theme import (
    DEFAULT_THEME,
    UI_CONFIG_FILENAME,
    css_variables,
    load_ui_theme,
    theme_from_dict,
)


class TestDefaultTheme(unittest.TestCase):
    """
    Test Suite for the shipped defaults (theme.DEFAULT_THEME).

    The point of the default palette is that one glance identifies a block:
    icon, caption, and left border share a single accent per speaker.
    """

    def test_user_and_assistant_are_distinguishable(self):
        self.assertEqual(DEFAULT_THEME.user.accent, "cyan")
        self.assertEqual(DEFAULT_THEME.assistant.accent, "orange")
        self.assertNotEqual(DEFAULT_THEME.user.background, DEFAULT_THEME.assistant.background)

    def test_captions_and_icons(self):
        self.assertEqual(DEFAULT_THEME.user.caption, "User")
        self.assertEqual(DEFAULT_THEME.assistant.caption, "Assistant")
        self.assertEqual(DEFAULT_THEME.reasoning.caption, "AI Reasoning")
        self.assertTrue(DEFAULT_THEME.user.icon)
        self.assertTrue(DEFAULT_THEME.assistant.icon)

    def test_reasoning_shares_the_user_background(self):
        self.assertEqual(DEFAULT_THEME.reasoning.background, DEFAULT_THEME.user.background)

    def test_input_is_cyan_when_focused(self):
        self.assertEqual(DEFAULT_THEME.input.border, "cyan")
        self.assertNotEqual(DEFAULT_THEME.input.border, DEFAULT_THEME.input.border_blurred)


class TestThemeFromDict(unittest.TestCase):
    """
    Test Suite for override handling (theme.theme_from_dict).
    A hand-edited config file is expected to be wrong sometimes; nothing in it
    may take the UI down.
    """

    def test_partial_override_keeps_the_rest(self):
        theme = theme_from_dict({"user": {"accent": "#ff00ff"}})

        self.assertEqual(theme.user.accent, "#ff00ff")
        self.assertEqual(theme.user.caption, DEFAULT_THEME.user.caption)
        self.assertEqual(theme.assistant, DEFAULT_THEME.assistant)

    def test_icons_and_captions_are_free_form(self):
        theme = theme_from_dict({"assistant": {"icon": ">>", "caption": "Prisma"}})

        self.assertEqual(theme.assistant.icon, ">>")
        self.assertEqual(theme.assistant.caption, "Prisma")

    def test_empty_config_returns_defaults(self):
        self.assertEqual(theme_from_dict({}), DEFAULT_THEME)

    def test_unknown_section_is_ignored(self):
        with self.assertLogs(level="WARNING"):
            theme = theme_from_dict({"sidebar": {"background": "red"}})
        self.assertEqual(theme, DEFAULT_THEME)

    def test_unknown_key_is_ignored(self):
        with self.assertLogs(level="WARNING"):
            theme = theme_from_dict({"user": {"glow": "red"}})
        self.assertEqual(theme, DEFAULT_THEME)

    def test_non_string_value_is_ignored(self):
        with self.assertLogs(level="WARNING"):
            theme = theme_from_dict({"user": {"accent": 42}})
        self.assertEqual(theme.user.accent, DEFAULT_THEME.user.accent)

    def test_section_that_is_not_an_object_is_ignored(self):
        with self.assertLogs(level="WARNING"):
            theme = theme_from_dict({"user": "cyan"})
        self.assertEqual(theme, DEFAULT_THEME)

    def test_invalid_color_is_rejected(self):
        with self.assertLogs(level="WARNING"):
            theme = theme_from_dict({"user": {"accent": "not a color; drop table"}})
        self.assertEqual(theme.user.accent, DEFAULT_THEME.user.accent)

    def test_accepted_color_forms(self):
        theme = theme_from_dict({
            "user": {"accent": "#abc", "background": "rgba(0,0,0,0.5)"},
            "assistant": {"accent": "$primary", "background": "transparent"},
            "tool": {"accent": "dodgerblue"},
        })

        self.assertEqual(theme.user.accent, "#abc")
        self.assertEqual(theme.user.background, "rgba(0,0,0,0.5)")
        self.assertEqual(theme.assistant.accent, "$primary")
        self.assertEqual(theme.assistant.background, "transparent")
        self.assertEqual(theme.tool.accent, "dodgerblue")


class TestLoadUITheme(unittest.TestCase):
    """
    Test Suite for config discovery (theme.load_ui_theme).
    Defaults, then ~/.prisma/ui.json, then <cwd>/.prisma/ui.json.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.home = self.root / "home"
        self.project = self.root / "project"
        (self.home / ".prisma").mkdir(parents=True)
        (self.project / ".prisma").mkdir(parents=True)

        self.app_config = AppConfig(app_name="prisma", app_dir_name=".prisma")
        self.home_patcher = patch.object(Path, "home", return_value=self.home)
        self.home_patcher.start()

    def tearDown(self):
        self.home_patcher.stop()
        self._tmp.cleanup()

    def _write(self, directory: Path, data) -> None:
        path = directory / ".prisma" / UI_CONFIG_FILENAME
        path.write_text(json.dumps(data) if not isinstance(data, str) else data, encoding="utf-8")

    def test_no_files_gives_defaults(self):
        self.assertEqual(load_ui_theme(self.app_config, self.project), DEFAULT_THEME)

    def test_global_config_is_applied(self):
        self._write(self.home, {"user": {"accent": "magenta"}})

        theme = load_ui_theme(self.app_config, self.project)
        self.assertEqual(theme.user.accent, "magenta")

    def test_project_overrides_global(self):
        self._write(self.home, {"user": {"accent": "magenta", "caption": "Me"}})
        self._write(self.project, {"user": {"accent": "green"}})

        theme = load_ui_theme(self.app_config, self.project)

        self.assertEqual(theme.user.accent, "green")
        # Untouched global keys survive the merge.
        self.assertEqual(theme.user.caption, "Me")

    def test_malformed_json_falls_back_to_defaults(self):
        self._write(self.project, "{ not json")

        with self.assertLogs(level="WARNING"):
            theme = load_ui_theme(self.app_config, self.project)

        self.assertEqual(theme, DEFAULT_THEME)

    def test_non_object_json_falls_back_to_defaults(self):
        self._write(self.project, [1, 2, 3])

        with self.assertLogs(level="WARNING"):
            theme = load_ui_theme(self.app_config, self.project)

        self.assertEqual(theme, DEFAULT_THEME)


class TestCSSVariables(unittest.TestCase):
    """Test Suite for the stylesheet bridge (theme.css_variables)."""

    def test_covers_every_section(self):
        variables = css_variables(DEFAULT_THEME)

        for prefix in ("header", "footer", "user", "assistant", "reasoning", "tool", "input"):
            self.assertTrue(
                any(name.startswith(f"prisma-{prefix}-") for name in variables),
                f"no variables for {prefix}",
            )

    def test_reflects_overrides(self):
        theme = theme_from_dict({"user": {"accent": "#123456"}})
        self.assertEqual(css_variables(theme)["prisma-user-accent"], "#123456")

    def test_values_are_all_strings(self):
        for name, value in css_variables(DEFAULT_THEME).items():
            self.assertIsInstance(value, str, name)


if __name__ == "__main__":
    unittest.main()
