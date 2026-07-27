import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from capabilities import (
    Capabilities,
    find_ripgrep,
    model_warnings,
    probe_capabilities,
    user_warnings,
)
from tools.ignore import GitStatus

GIT = shutil.which("git")


class TestCapabilityProbe(unittest.TestCase):
    """
    The probe decides which Grep engine runs and what the session warns about.

    Its git check deliberately runs the real `git ls-files`, because the failure
    that matters most in practice is a repository git refuses to read: a lighter
    check such as `git --version` would report success there.
    """

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.workspace = Path(self.test_dir.name).resolve()

    def tearDown(self):
        self.test_dir.cleanup()

    def test_ripgrep_is_reported_as_a_path(self):
        with patch("capabilities.shutil.which", return_value="/usr/bin/rg"):
            self.assertEqual(find_ripgrep(), "/usr/bin/rg")

    def test_missing_ripgrep_is_reported_as_none(self):
        with patch("capabilities.shutil.which", return_value=None):
            capabilities = probe_capabilities(self.workspace)

        self.assertIsNone(capabilities.ripgrep)

    def test_plain_directory_is_not_degraded(self):
        capabilities = probe_capabilities(self.workspace)

        self.assertIs(capabilities.git_status, GitStatus.UNUSED)
        self.assertFalse(capabilities.git_ignores_degraded)

    @unittest.skipUnless(GIT, "git is not installed")
    def test_healthy_repository_is_not_degraded(self):
        subprocess.run([GIT, "init"], cwd=self.workspace, capture_output=True, check=True)

        capabilities = probe_capabilities(self.workspace)

        self.assertIs(capabilities.git_status, GitStatus.OK)
        self.assertFalse(capabilities.git_ignores_degraded)

    @unittest.skipUnless(GIT, "git is not installed")
    def test_repository_without_usable_git_is_degraded(self):
        subprocess.run([GIT, "init"], cwd=self.workspace, capture_output=True, check=True)

        with patch("tools.ignore.subprocess.run", side_effect=OSError("no git")):
            capabilities = probe_capabilities(self.workspace)

        self.assertIs(capabilities.git_status, GitStatus.UNAVAILABLE)
        self.assertTrue(capabilities.git_ignores_degraded)


class TestWarningAudiences(unittest.TestCase):
    """
    The two audiences need different things.

    A missing ripgrep changes nothing the model can observe: whichever engine is
    registered describes its own capabilities. Telling the model would only
    invite it to apologise for the machine. The user, by contrast, is the only
    one who can install it.
    """

    HEALTHY = Capabilities(ripgrep="/usr/bin/rg", git_status=GitStatus.OK)
    NO_RIPGREP = Capabilities(ripgrep=None, git_status=GitStatus.OK)
    NO_GIT = Capabilities(
        ripgrep="/usr/bin/rg",
        git_status=GitStatus.UNAVAILABLE,
        git_error="no git",
    )
    REFUSED_GIT = Capabilities(
        ripgrep="/usr/bin/rg",
        git_status=GitStatus.FAILED,
        git_error="fatal: detected dubious ownership",
    )

    def test_healthy_environment_warns_nobody(self):
        self.assertEqual(model_warnings(self.HEALTHY), [])
        self.assertEqual(user_warnings(self.HEALTHY), [])

    def test_missing_ripgrep_is_hidden_from_the_model(self):
        self.assertEqual(model_warnings(self.NO_RIPGREP), [])

    def test_missing_ripgrep_is_shown_to_the_user(self):
        warnings = user_warnings(self.NO_RIPGREP)

        self.assertEqual(len(warnings), 1)
        self.assertIn("ripgrep", warnings[0])

    def test_degraded_git_reaches_both_audiences(self):
        self.assertEqual(len(model_warnings(self.NO_GIT)), 1)
        self.assertEqual(len(user_warnings(self.NO_GIT)), 1)

        for warning in model_warnings(self.NO_GIT) + user_warnings(self.NO_GIT):
            self.assertIn(".gitignore", warning)

    def test_the_model_is_never_told_about_ripgrep(self):
        both_broken = Capabilities(ripgrep=None, git_status=GitStatus.UNAVAILABLE)

        for warning in model_warnings(both_broken):
            self.assertNotIn("ripgrep", warning.lower())

    def test_git_refusal_is_quoted_with_a_way_out(self):
        """'dubious ownership' is fixable, but only if the user is told."""
        warning = user_warnings(self.REFUSED_GIT)[0]

        self.assertIn("dubious ownership", warning)
        self.assertIn("safe.directory", warning)

    def test_both_failures_produce_two_user_warnings(self):
        both_broken = Capabilities(ripgrep=None, git_status=GitStatus.FAILED)

        self.assertEqual(len(user_warnings(both_broken)), 2)


if __name__ == "__main__":
    unittest.main()
