import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch, PropertyMock

from config import AppConfig
from context import gather_context_files

class TestContextGathering(unittest.TestCase):
    """
    Test Suite for AGENTS.md Context Gathering (context.py)
    Validates global/local discovery, the strictly descending path lineage,
    and boundary constraints (root_dir vs cwd).
    """

    def setUp(self):
        # Create a real temporary directory to test accurate Pathlib behaviors
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.temp_dir.name).resolve()
        
        # Setup fake Global config directory
        self.fake_home_dir = self.base_path / "fake_home" / ".prisma"
        self.fake_home_dir.mkdir(parents=True)
        
        # Base AppConfig
        self.app_config = AppConfig(app_name="prisma", app_dir_name=".prisma")

    def tearDown(self):
        self.temp_dir.cleanup()

    @patch("config.AppConfig.home_config_dir", new_callable=PropertyMock)
    def test_cwd_outside_root_raises_error(self, mock_home_dir):
        """Test 1: Value error is raised if cwd is not a child of root."""
        mock_home_dir.return_value = self.fake_home_dir

        root = self.base_path / "project_a"
        cwd = self.base_path / "project_b"  # Completely outside root
        
        with self.assertRaises(ValueError) as context:
            gather_context_files(self.app_config, root, cwd)
            
        self.assertIn("not within the specified root", str(context.exception))

    @patch("config.AppConfig.home_config_dir", new_callable=PropertyMock)
    def test_global_agents_only(self, mock_home_dir):
        """Test 2: Correctly loads only the global AGENTS.md if no locals exist."""
        mock_home_dir.return_value = self.fake_home_dir
        
        # Create global file
        global_file = self.fake_home_dir / "AGENTS.md"
        global_file.write_text("Global Agent Context", encoding="utf-8")
        
        root = self.base_path / "project"
        cwd = root
        
        result = gather_context_files(self.app_config, root, cwd)
        
        self.assertIn("--- From", result)
        self.assertIn(str(global_file), result)
        self.assertIn("Global Agent Context", result)

    @patch("config.AppConfig.home_config_dir", new_callable=PropertyMock)
    def test_descending_lineage_gathering(self, mock_home_dir):
        """Test 3: Gathers files in precise order: Global -> Root -> Src -> Backend."""
        mock_home_dir.return_value = self.fake_home_dir
        
        # Create global file
        (self.fake_home_dir / "AGENTS.md").write_text("Global", encoding="utf-8")
        
        # Create project structure
        root = self.base_path / "project"
        src = root / "src"
        backend = src / "backend"
        backend.mkdir(parents=True)  # Creates all folders in path
        
        # Write AGENTS.md at each tier
        (root / "AGENTS.md").write_text("Root Agent", encoding="utf-8")
        (src / "AGENTS.md").write_text("Src Agent", encoding="utf-8")
        (backend / "AGENTS.md").write_text("Backend Agent", encoding="utf-8")
        
        # CWD is at the deepest level
        cwd = backend
        
        result = gather_context_files(self.app_config, root, cwd)
        
        # Verify all elements are present
        self.assertIn("Global", result)
        self.assertIn("Root Agent", result)
        self.assertIn("Src Agent", result)
        self.assertIn("Backend Agent", result)
        
        # Verify Order (Global -> Root -> Src -> Backend)
        global_idx = result.find("Global")
        root_idx = result.find("Root Agent")
        src_idx = result.find("Src Agent")
        backend_idx = result.find("Backend Agent")
        
        self.assertTrue(global_idx < root_idx < src_idx < backend_idx, "Context files were assembled in the wrong order.")

    @patch("config.AppConfig.home_config_dir", new_callable=PropertyMock)
    def test_missing_and_empty_files_skipped(self, mock_home_dir):
        """Test 4: Gracefully skips missing files and completely empty files."""
        mock_home_dir.return_value = self.fake_home_dir
        
        root = self.base_path / "project"
        src = root / "src"
        backend = src / "backend"
        backend.mkdir(parents=True)
        
        # Root has an EMPTY file
        (root / "AGENTS.md").write_text("   \n  \n", encoding="utf-8") 
        
        # Src is MISSING its file entirely
        
        # Backend has a VALID file
        (backend / "AGENTS.md").write_text("Backend Only", encoding="utf-8")
        
        cwd = backend
        
        result = gather_context_files(self.app_config, root, cwd)
        
        # We should ONLY see the backend file
        self.assertIn("Backend Only", result)
        
        # Should not have any empty headers or crashes
        self.assertEqual(result.count("--- From"), 1, "Should only have one header injected.")
        self.assertNotIn(str(root / "AGENTS.md"), result)

    @patch("config.AppConfig.home_config_dir", new_callable=PropertyMock)
    def test_root_is_cwd(self, mock_home_dir):
        """Test 5: If cwd == root, it only checks the global and root files."""
        mock_home_dir.return_value = self.fake_home_dir
        
        root = self.base_path / "project"
        root.mkdir()
        (root / "AGENTS.md").write_text("Root Info", encoding="utf-8")
        
        cwd = root
        
        result = gather_context_files(self.app_config, root, cwd)
        
        self.assertIn("Root Info", result)
        self.assertEqual(result.count("--- From"), 1)

if __name__ == '__main__':
    unittest.main()