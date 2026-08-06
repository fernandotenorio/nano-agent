import unittest

from ui.truncate import (
    HEAD_LINES,
    MAX_LINE_CHARS,
    MAX_OUTPUT_CHARS,
    MAX_VALUE_CHARS,
    TAIL_LINES,
    TruncatedText,
    as_text,
    truncate_call,
    truncate_output,
)


class _Block:
    """Stands in for a TextMessageContent block, which only `.text` matters for."""
    def __init__(self, text: str):
        self.text = text


class TestAsText(unittest.TestCase):
    """
    Test Suite for payload flattening (truncate.as_text).
    Tool results arrive as strings, block lists, or nothing at all.
    """

    def test_none_is_empty(self):
        self.assertEqual(as_text(None), "")

    def test_string_passthrough(self):
        self.assertEqual(as_text("hello"), "hello")

    def test_blocks_are_joined(self):
        self.assertEqual(as_text([_Block("one"), _Block("two")]), "one\ntwo")

    def test_other_types_are_stringified(self):
        self.assertEqual(as_text(42), "42")


class TestTruncateCall(unittest.TestCase):
    """
    Test Suite for call rendering (truncate.truncate_call).

    The pane exists for debugging, so a short call must survive verbatim and a
    long one must say what it dropped.
    """

    def test_small_call_is_untouched(self):
        result = truncate_call("Grep", {"pattern": "foo", "path": "src"})

        self.assertFalse(result.truncated)
        self.assertIn("Grep(", result.text)
        self.assertIn('pattern = "foo"', result.text)
        self.assertIn('path = "src"', result.text)

    def test_empty_args(self):
        result = truncate_call("ls", {})

        self.assertEqual(result.text, "ls()")
        self.assertFalse(result.truncated)

    def test_non_dict_args(self):
        result = truncate_call("Weird", "just a string")

        self.assertFalse(result.truncated)
        self.assertIn("Weird(", result.text)
        self.assertIn("just a string", result.text)

    def test_long_value_keeps_both_ends(self):
        content = "A" * 400 + "MIDDLE" + "Z" * 400
        result = truncate_call("Write", {"file_path": "a.txt", "content": content})

        self.assertTrue(result.truncated)
        self.assertIn("characters omitted", result.text)
        # Both ends survive; the middle does not.
        self.assertIn("A" * 50, result.text)
        self.assertIn("Z" * 50, result.text)
        self.assertNotIn("MIDDLE", result.text)
        # The file path is short and must not be collateral damage.
        self.assertIn("a.txt", result.text)

    def test_value_at_the_limit_is_not_truncated(self):
        result = truncate_call("Write", {"content": "A" * MAX_VALUE_CHARS})
        self.assertFalse(result.truncated)

    def test_nested_values_are_elided(self):
        edits = [{"old_string": "B" * 2000, "new_string": "short"}]
        result = truncate_call("MultiEdit", {"file_path": "a.txt", "edits": edits})

        self.assertTrue(result.truncated)
        self.assertIn("characters omitted", result.text)
        self.assertIn("short", result.text)

    def test_many_arguments_cap_total_lines(self):
        args = {f"key_{i}": f"value {i}" for i in range(200)}
        result = truncate_call("Big", args)

        self.assertTrue(result.truncated)
        self.assertIn("lines omitted", result.text)
        self.assertIn("lines", result.detail)
        # The tail survives, so the closing paren is still visible.
        self.assertTrue(result.text.rstrip().endswith(")"))

    def test_multiline_string_is_indented_not_escaped(self):
        result = truncate_call("Write", {"content": "line one\nline two"})

        self.assertIn("    line one", result.text)
        self.assertIn("    line two", result.text)
        self.assertNotIn("\\n", result.text)

    def test_unserializable_value_does_not_raise(self):
        result = truncate_call("Odd", {"thing": object()})
        self.assertIn("thing =", result.text)


class TestTruncateOutput(unittest.TestCase):
    """
    Test Suite for output rendering (truncate.truncate_output).
    Grep results and build logs must be bounded without hiding the tail.
    """

    def test_empty_output(self):
        result = truncate_output("")

        self.assertEqual(result.text, "")
        self.assertFalse(result.truncated)

    def test_short_output_is_untouched(self):
        result = truncate_output("one\ntwo\nthree")

        self.assertEqual(result.text, "one\ntwo\nthree")
        self.assertFalse(result.truncated)

    def test_long_output_keeps_head_and_tail(self):
        lines = [f"line {i}" for i in range(500)]
        result = truncate_output("\n".join(lines))

        self.assertTrue(result.truncated)
        self.assertIn("line 0", result.text)
        self.assertIn("line 499", result.text)
        self.assertIn("lines omitted", result.text)
        self.assertEqual(result.detail, "500 lines")
        # Head + marker + tail, nothing more.
        self.assertEqual(len(result.text.splitlines()), HEAD_LINES + 1 + TAIL_LINES)

    def test_output_at_the_line_limit_is_not_truncated(self):
        lines = [f"line {i}" for i in range(HEAD_LINES + TAIL_LINES + 1)]
        result = truncate_output("\n".join(lines))
        self.assertFalse(result.truncated)

    def test_single_enormous_line_is_capped(self):
        result = truncate_output("X" * 5000)

        self.assertTrue(result.truncated)
        self.assertIn("characters omitted", result.text)
        self.assertLess(len(result.text), MAX_LINE_CHARS + 100)

    def test_hard_character_cap(self):
        # Lines short enough to survive the line rules, numerous enough that
        # what remains would still be enormous.
        long_line = "Y" * (MAX_LINE_CHARS - 1)
        result = truncate_output("\n".join([long_line] * (HEAD_LINES + TAIL_LINES)))

        self.assertTrue(result.truncated)
        self.assertLessEqual(len(result.text), MAX_OUTPUT_CHARS + 40)

    def test_accepts_block_lists(self):
        result = truncate_output([_Block("alpha"), _Block("beta")])
        self.assertEqual(result.text, "alpha\nbeta")


class TestLabel(unittest.TestCase):
    """Test Suite for pane titles (TruncatedText.label)."""

    def test_untruncated_label(self):
        self.assertEqual(TruncatedText(text="x").label("Output"), "Output")

    def test_truncated_label_with_detail(self):
        label = TruncatedText(text="x", truncated=True, detail="1,204 lines").label("Output")
        self.assertEqual(label, "Output (truncated, 1,204 lines)")

    def test_truncated_label_without_detail(self):
        label = TruncatedText(text="x", truncated=True).label("Call")
        self.assertEqual(label, "Call (truncated)")


if __name__ == "__main__":
    unittest.main()
