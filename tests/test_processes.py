import unittest
from unittest.mock import MagicMock

from processes import kill_quietly, terminate_quietly


class TestProcessSignals(unittest.TestCase):
    """
    Test Suite for Process Signalling (processes.py)
    A child can exit between the moment we decide to stop it and the moment we
    signal it. Once asyncio has reaped it, terminate()/kill() raise
    ProcessLookupError, which means "already dead" and must be swallowed.
    """

    def test_kill_signals_a_live_process(self):
        process = MagicMock()

        kill_quietly(process)

        process.kill.assert_called_once_with()

    def test_terminate_signals_a_live_process(self):
        process = MagicMock()

        terminate_quietly(process)

        process.terminate.assert_called_once_with()

    def test_kill_tolerates_a_reaped_process(self):
        process = MagicMock()
        process.kill.side_effect = ProcessLookupError()

        kill_quietly(process)  # Must not raise

        process.kill.assert_called_once_with()

    def test_terminate_tolerates_a_reaped_process(self):
        process = MagicMock()
        process.terminate.side_effect = ProcessLookupError()

        terminate_quietly(process)  # Must not raise

        process.terminate.assert_called_once_with()

    def test_other_errors_still_propagate(self):
        """Only 'already dead' is expected. A permission error is a real bug."""
        process = MagicMock()
        process.kill.side_effect = PermissionError("not your process")

        with self.assertRaises(PermissionError):
            kill_quietly(process)


if __name__ == "__main__":
    unittest.main()
