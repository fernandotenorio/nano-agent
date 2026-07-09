import unittest
import tempfile
import json
from pathlib import Path

from transcript import Transcript
from typedefs import (
    SystemMessage, UserMessage, AssistantMessage, 
    TextMessageContent, ThinkingMessageContent, ToolUseMessageContent
)

class TestTranscript(unittest.TestCase):
    """
    Test Suite for Transcript State & Disk I/O (transcript.py)
    Validates JSONL serialization/deserialization, automatic directory management, 
    and robust error handling.
    """

    def setUp(self):
        # Create a real temporary directory for safe disk I/O
        self.test_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.test_dir.name)

    def tearDown(self):
        self.test_dir.cleanup()

    # ---------------------------------------------------------
    # GROUP 1: Loading from Disk
    # ---------------------------------------------------------

    def test_load_file_does_not_exist(self):
        """Test 1.1: Transcript handles missing files gracefully without crashing."""
        missing_path = self.base_path / "missing.jsonl"
        transcript = Transcript(missing_path)
        
        self.assertEqual(transcript.messages, [])
        # The file shouldn't be created just by initializing
        self.assertFalse(missing_path.exists())

    def test_load_valid_parsing(self):
        """Test 1.2: Transcript perfectly routes 'role' strings to Pydantic models."""
        target_path = self.base_path / "valid.jsonl"
        
        # Manually construct raw JSON strings exactly as they would appear on disk
        sys_json = '{"role": "system", "content": "You are AI."}'
        usr_json = '{"role": "user", "content": "Hello"}'
        ast_json = '{"role": "assistant", "id": "123", "type": "message", "content": [{"type": "text", "text": "Hi"}]}'
        
        target_path.write_text(f"{sys_json}\n{usr_json}\n{ast_json}", encoding="utf-8")
        
        transcript = Transcript(target_path)
        
        self.assertEqual(len(transcript.messages), 3)
        self.assertIsInstance(transcript.messages[0], SystemMessage)
        self.assertIsInstance(transcript.messages[1], UserMessage)
        self.assertIsInstance(transcript.messages[2], AssistantMessage)
        
        # Verify nested data populated correctly
        self.assertEqual(transcript.messages[2].content[0].text, "Hi")

    def test_load_ignoring_blank_lines(self):
        """Test 1.3: Transcript safely skips blank lines and whitespace."""
        target_path = self.base_path / "blanks.jsonl"
        
        sys_json = '{"role": "system", "content": "You are AI."}'
        usr_json = '{"role": "user", "content": "Hello"}'
        
        # Inject blank lines and trailing spaces between JSON objects
        dirty_content = f"   \n{sys_json}\n\n\n  \n{usr_json}\n"
        target_path.write_text(dirty_content, encoding="utf-8")
        
        transcript = Transcript(target_path)
        
        self.assertEqual(len(transcript.messages), 2)
        self.assertIsInstance(transcript.messages[0], SystemMessage)
        self.assertIsInstance(transcript.messages[1], UserMessage)

    def test_load_unrecognized_role(self):
        """Test 1.4: Unrecognized roles fall through without breaking the loop."""
        target_path = self.base_path / "alien.jsonl"
        
        valid_json = '{"role": "system", "content": "Valid"}'
        alien_json = '{"role": "alien", "content": "Take me to your leader."}'
        
        target_path.write_text(f"{valid_json}\n{alien_json}", encoding="utf-8")
        
        transcript = Transcript(target_path)
        
        # It should process the valid one, and silently skip the alien one
        self.assertEqual(len(transcript.messages), 1)
        self.assertIsInstance(transcript.messages[0], SystemMessage)


    # ---------------------------------------------------------
    # GROUP 2: Writing to Disk
    # ---------------------------------------------------------

    def test_append_automatic_directory_creation(self):
        """Test 2.1: Transcript creates deeply nested, missing folders automatically."""
        # This path does not exist yet!
        deep_path = self.base_path / "nested" / "very" / "deep" / "chat.jsonl"
        transcript = Transcript(deep_path)
        
        sys_msg = SystemMessage(content="Initialize.")
        transcript.append(sys_msg)
        
        # Assert directory was built
        self.assertTrue(deep_path.parent.exists())
        self.assertTrue(deep_path.exists())
        
        # Assert file contents
        file_contents = deep_path.read_text(encoding="utf-8")
        self.assertIn('"role":"system"', file_contents)
        self.assertIn('"content":"Initialize."', file_contents)

    def test_append_atomic_line_appending(self):
        """Test 2.2: Transcript writes distinct lines atomically."""
        target_path = self.base_path / "append.jsonl"
        transcript = Transcript(target_path)
        
        transcript.append(SystemMessage(content="A"))
        transcript.append(UserMessage(content="B"))
        transcript.append(SystemMessage(content="C"))
        
        self.assertEqual(len(transcript.messages), 3)
        
        # Read the raw file directly
        lines = target_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 3)
        self.assertIn('"content":"A"', lines[0])
        self.assertIn('"content":"B"', lines[1])
        self.assertIn('"content":"C"', lines[2])


    # ---------------------------------------------------------
    # GROUP 3: End-to-End Data Integrity
    # ---------------------------------------------------------

    def test_complex_serialization_round_trip(self):
        """
        Test 3.1: Complex structures (nested arrays, dicts, thinking blocks) 
        survive the memory -> disk -> memory round-trip perfectly.
        """
        target_path = self.base_path / "complex.jsonl"
        
        # 1. Create a highly complex message
        complex_msg = AssistantMessage(
            id="req_999",
            model="test-model",
            content=[
                TextMessageContent(text="Standard text."),
                ThinkingMessageContent(thinking="Deep thoughts here.", signature="sig123"),
                ToolUseMessageContent(id="call_x", name="Write", input={"path": "main.py", "lines": [1, 2]})
            ]
        )
        
        # 2. Append it to Transcript A (saves to disk)
        transcript_a = Transcript(target_path)
        transcript_a.append(complex_msg)
        
        # 3. Load it perfectly into Transcript B
        transcript_b = Transcript(target_path)
        self.assertEqual(len(transcript_b.messages), 1)
        
        loaded_msg = transcript_b.messages[0]
        self.assertIsInstance(loaded_msg, AssistantMessage)
        
        # 4. Deep structural assertions
        self.assertEqual(loaded_msg.id, "req_999")
        self.assertEqual(loaded_msg.model, "test-model")
        self.assertEqual(len(loaded_msg.content), 3)
        
        # Text Block
        self.assertIsInstance(loaded_msg.content[0], TextMessageContent)
        self.assertEqual(loaded_msg.content[0].text, "Standard text.")
        
        # Thinking Block
        self.assertIsInstance(loaded_msg.content[1], ThinkingMessageContent)
        self.assertEqual(loaded_msg.content[1].thinking, "Deep thoughts here.")
        self.assertEqual(loaded_msg.content[1].signature, "sig123")
        
        # Tool Use Block
        self.assertIsInstance(loaded_msg.content[2], ToolUseMessageContent)
        self.assertEqual(loaded_msg.content[2].name, "Write")
        
        # Verify the dict args translated cleanly
        expected_input = {"path": "main.py", "lines": [1, 2]}
        self.assertEqual(loaded_msg.content[2].input, expected_input)


if __name__ == "__main__":
    unittest.main()