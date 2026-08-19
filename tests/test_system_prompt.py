# test_system_prompt.py

import unittest
import tempfile
import logging
from pathlib import Path
from unittest.mock import patch

from capabilities import Capabilities
from config import load_app_config
from environment import get_environment_details
from prompts import build_system_prompt, _DEFAULT_USER_INSTRUCTIONS, _load_core_instructions
from sessioncontext import InvocationContext
from tools.ignore import GitStatus


class MockArgs:
    """Helper to simulate argparse namespace for the system prompt builder."""
    def __init__(
        self,
        system_prompt_file: str | None = None,
        no_global_system_prompt_file: bool = False,
    ):
        self.system_prompt_file = system_prompt_file
        self.no_global_system_prompt_file = no_global_system_prompt_file


class TestSystemPromptBuilder(unittest.TestCase):
    """
    Test Suite for System Prompt Loading & Discovery (prompts.py)
    Validates the layered assembly of core instructions, user overrides,
    and the global SYSTEM.md.

    There is no project layer here: project instructions live in AGENTS.md
    (see tests/test_context.py), so nothing inside a workspace reaches the
    system message.
    """

    def setUp(self):
        # 1. Create a sandboxed file system environment
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.temp_dir.name)
        
        # 2. Setup mock Home and Project directories
        self.mock_home = self.base_path / "home"
        self.mock_cwd = self.base_path / "project"
        self.mock_home.mkdir()
        self.mock_cwd.mkdir()

        # 3. Patch config's Path.home() to route global lookups to our mock home
        self.home_patcher = patch("config.Path.home", return_value=self.mock_home)
        self.home_patcher.start()

        # 4. Patch environment details for consistent test outputs
        self.env_patcher = patch("prompts.get_environment_details", return_value="<mocked_env>")
        self.env_patcher.start()

        # 5. Load App Config (will respect the patched Path.home)
        self.app_config = load_app_config()
        self.core_instructions = _load_core_instructions(self.app_config)

        self.ctx = InvocationContext(
            workspace="proj/dummy/",
            cwd="proj/dummy/",
            workspace_is_git_repo=False,
            resume_file=None
        )

    def tearDown(self):
        self.home_patcher.stop()
        self.env_patcher.stop()
        self.temp_dir.cleanup()

    # ---------------------------------------------------------
    # GROUP 1: Default Behavior
    # ---------------------------------------------------------

    def test_default_fallback_behavior(self):
        """Test 1.1: With no files and no flags, it falls back to 3 distinct parts."""
        args = MockArgs()
        sys_msg = build_system_prompt(self.app_config, self.ctx, args)
        
        # The prompt should be joined by \n\n---\n\n
        parts = sys_msg.content.split("\n\n---\n\n")
        
        self.assertEqual(len(parts), 3)
        self.assertEqual(parts[0], self.core_instructions)
        self.assertEqual(parts[1], _DEFAULT_USER_INSTRUCTIONS)
        self.assertEqual(parts[2], "<mocked_env>")

    # ---------------------------------------------------------
    # GROUP 2: User Instructions (--system-prompt-file)
    # ---------------------------------------------------------

    def test_user_file_successful_load(self):
        """Test 2.1: Custom user prompt file entirely replaces defaults."""
        custom_file = self.base_path / "custom_prompt.md"
        custom_file.write_text("Act as a strict linter.", encoding="utf-8")
        
        args = MockArgs(system_prompt_file=str(custom_file))
        sys_msg = build_system_prompt(self.app_config, self.ctx, args)
        
        self.assertIn("Act as a strict linter.", sys_msg.content)
        self.assertNotIn(_DEFAULT_USER_INSTRUCTIONS, sys_msg.content)

    def test_user_file_missing_fallback(self):
        """Test 2.2: Missing custom file logs a warning and falls back to default."""
        missing_file = self.base_path / "does_not_exist.md"
        args = MockArgs(system_prompt_file=str(missing_file))
        
        with self.assertLogs(level='WARNING') as cm:
            sys_msg = build_system_prompt(self.app_config, self.ctx, args)
            
        self.assertIn(_DEFAULT_USER_INSTRUCTIONS, sys_msg.content)
        self.assertTrue(any("Could not load instructions" in log for log in cm.output))

    def test_user_file_empty_fallback(self):
        """Test 2.3: Empty custom file (or whitespace only) falls back to default."""
        empty_file = self.base_path / "empty.md"
        empty_file.write_text("   \n  \n", encoding="utf-8")
        
        args = MockArgs(system_prompt_file=str(empty_file))
        sys_msg = build_system_prompt(self.app_config, self.ctx, args)
        
        self.assertIn(_DEFAULT_USER_INSTRUCTIONS, sys_msg.content)

    def test_user_file_read_error_fallback(self):
        """Test 2.4: Unreadable custom file logs a warning and falls back safely."""
        error_file = self.base_path / "error.md"
        error_file.write_text("content", encoding="utf-8")
        
        args = MockArgs(system_prompt_file=str(error_file))
        
        # Patch read_text to simulate a permissions error
        with patch("pathlib.Path.read_text", side_effect=PermissionError("Access denied")):
            with self.assertLogs(level='WARNING') as cm:
                sys_msg = build_system_prompt(self.app_config, self.ctx, args)
                
        self.assertIn(_DEFAULT_USER_INSTRUCTIONS, sys_msg.content)
        self.assertTrue(any("Failed to read" in log for log in cm.output))

    # ---------------------------------------------------------
    # GROUP 3: Global SYSTEM.md Discovery
    # ---------------------------------------------------------

    def _setup_global_system_file(self, content: str) -> Path:
        global_file = self.app_config.global_system_prompt_file()
        global_file.parent.mkdir(parents=True, exist_ok=True)
        global_file.write_text(content, encoding="utf-8")
        return global_file

    def test_global_system_file_success(self):
        """Test 3.1: Global SYSTEM.md is dynamically discovered and appended."""
        self._setup_global_system_file("Global company guidelines.")
        
        args = MockArgs()
        sys_msg = build_system_prompt(self.app_config, self.ctx, args)
        
        self.assertIn("Global company guidelines.", sys_msg.content)

    def test_global_system_file_flag_override(self):
        """Test 3.2: --no-global-system-prompt-file flag overrides discovery."""
        self._setup_global_system_file("Global company guidelines.")
        
        args = MockArgs(no_global_system_prompt_file=True)
        sys_msg = build_system_prompt(self.app_config, self.ctx, args)
        
        self.assertNotIn("Global company guidelines.", sys_msg.content)

    # ---------------------------------------------------------
    # GROUP 4: Nothing Inside The Workspace Is Read
    # ---------------------------------------------------------

    def test_a_workspace_local_system_file_is_not_read(self):
        """The project layer is gone: instructions committed to a repository
        belong in AGENTS.md, and a SYSTEM.md left over from the old layout must
        not quietly keep working."""
        stale = self.mock_cwd / ".prisma" / "SYSTEM.md"
        stale.parent.mkdir(parents=True, exist_ok=True)
        stale.write_text("Project specific architecture rules.", encoding="utf-8")

        args = MockArgs()
        sys_msg = build_system_prompt(self.app_config, self.ctx, args)

        self.assertNotIn("Project specific architecture rules.", sys_msg.content)

    # ---------------------------------------------------------
    # GROUP 5: Integration / Order of Assembly
    # ---------------------------------------------------------

    def test_full_stack_assembly_order(self):
        """
        Test 5.1: Verifies the strict assembly order when all layers are present.
        Must be: Core -> User -> Global -> Environment
        """
        # 1. Setup Custom User File
        custom_file = self.base_path / "custom.md"
        custom_file.write_text("User Custom Layer", encoding="utf-8")
        
        # 2. Setup Global File
        self._setup_global_system_file("Global Config Layer")
        
        # Action
        args = MockArgs(system_prompt_file=str(custom_file))
        sys_msg = build_system_prompt(self.app_config, self.ctx, args)
        
        # Assertions
        parts = sys_msg.content.split("\n\n---\n\n")
        
        self.assertEqual(len(parts), 4)
        self.assertEqual(parts[0], self.core_instructions)
        self.assertEqual(parts[1], "User Custom Layer")
        self.assertEqual(parts[2], "Global Config Layer")
        self.assertEqual(parts[3], "<mocked_env>")


class TestEnvironmentCapabilityWarnings(unittest.TestCase):
    """
    The environment block must not overstate what the tools can see.

    Reporting a plain 'Git repo: Yes' while git is unusable is worse than
    silence: it invites the agent to trust listings that quietly include build
    output and caches.
    """

    def _context(self, capabilities) -> InvocationContext:
        return InvocationContext(
            workspace=Path("/repo"),
            cwd=Path("/repo"),
            workspace_is_git_repo=True,
            resume_file=None,
            capabilities=capabilities,
        )

    def test_healthy_environment_says_nothing_extra(self):
        details = get_environment_details(
            self._context(Capabilities(ripgrep="/usr/bin/rg", git_status=GitStatus.OK))
        )

        self.assertIn("Workspace root is a Git repo: Yes", details)
        self.assertNotIn("WARNING", details)

    def test_unprobed_context_says_nothing_extra(self):
        details = get_environment_details(self._context(None))

        self.assertIn("Workspace root is a Git repo: Yes", details)
        self.assertNotIn("WARNING", details)

    def test_degraded_git_is_flagged(self):
        details = get_environment_details(
            self._context(
                Capabilities(git_status=GitStatus.UNAVAILABLE, git_error="no git")
            )
        )

        self.assertIn("WARNING", details)
        self.assertIn(".gitignore", details)

    def test_missing_ripgrep_is_not_mentioned(self):
        """Which engine backs Grep is the tool's business, not the prompt's."""
        details = get_environment_details(
            self._context(Capabilities(ripgrep=None, git_status=GitStatus.OK))
        )

        self.assertNotIn("ripgrep", details.lower())
        self.assertNotIn("WARNING", details)

    def test_warning_does_not_break_the_block_layout(self):
        """The warning is interpolated into an indented block before dedent().

        An unindented continuation line would flatten the common prefix and
        leave the whole environment block indented.
        """
        details = get_environment_details(
            self._context(
                Capabilities(git_status=GitStatus.FAILED, git_error="fatal: refused")
            )
        )

        self.assertIn("\n<workspace>\n", details)
        self.assertIn("\nOS: ", details)


if __name__ == "__main__":
    unittest.main()