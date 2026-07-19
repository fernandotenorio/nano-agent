import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import json
import sys

# Import litellm internals to create realistic mock responses
import litellm
from litellm.types.utils import ModelResponse, Choices, Message, Usage

from typedefs import (
    SystemMessage, AssistantMessage, UserMessage, 
    TextMessageContent, ToolUseMessageContent, ToolResultMessageContent, ThinkingMessageContent
)
from adapter import (
    format_tool_desc, to_openai_message, parse_assistant_response, acompletion, spinner
)


class TestAdapterLLM(unittest.IsolatedAsyncioTestCase):
    """
    Test Suite for the LLM Adapter (adapter.py)
    Validates API payload formatting, serialization/deserialization, 
    and orchestration rules for different AI providers (Anthropic, OpenAI, Ollama).
    """

    # ---------------------------------------------------------
    # GROUP 1: Tool Schema Formatting
    # ---------------------------------------------------------

    def test_format_tool_desc_unwrapped(self):
        raw_dict = {"name": "Read", "parameters": {"type": "object"}}
        result = format_tool_desc(raw_dict)
        
        self.assertEqual(result["type"], "function")
        self.assertIn("function", result)
        self.assertEqual(result["function"]["name"], "Read")
        self.assertEqual(result["function"]["parameters"], {"type": "object"})

    def test_format_tool_desc_already_wrapped(self):
        wrapped_dict = {"type": "function", "function": {"name": "Read", "parameters": {}}}
        result = format_tool_desc(wrapped_dict)
        
        # Idempotency check
        self.assertEqual(result, wrapped_dict)

    def test_format_tool_desc_mcp_object(self):
        mcp_mock = MagicMock()
        mcp_mock.name = "MCPTool"
        mcp_mock.description = "Does MCP stuff"
        mcp_mock.inputSchema = {"type": "object"}
        
        result = format_tool_desc(mcp_mock)
        self.assertEqual(result["type"], "function")
        self.assertEqual(result["function"]["name"], "MCPTool")
        self.assertEqual(result["function"]["description"], "Does MCP stuff")
        self.assertEqual(result["function"]["parameters"], {"type": "object"})


    # ---------------------------------------------------------
    # GROUP 2: Outbound Message Translation
    # ---------------------------------------------------------

    def test_to_openai_message_system(self):
        msg = SystemMessage(content="You are a helpful AI.")
        
        # Without Cache Control
        out = to_openai_message(msg, add_cache_control=False)
        self.assertEqual(out, [{"role": "system", "content": "You are a helpful AI."}])
        
        # With Anthropic Cache Control
        out_cached = to_openai_message(msg, add_cache_control=True)
        self.assertEqual(len(out_cached), 1)
        self.assertEqual(out_cached[0]["role"], "system")
        self.assertIsInstance(out_cached[0]["content"], list)
        self.assertEqual(out_cached[0]["content"][0]["cache_control"]["type"], "ephemeral")

    def test_to_openai_message_assistant_tool_use(self):
        msg = AssistantMessage(content=[
            TextMessageContent(text="Let me do that."),
            ToolUseMessageContent(id="call_123", name="Shell", input={"cmd": "ls"})
        ])
        
        out = to_openai_message(msg, add_cache_control=False)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["role"], "assistant")
        self.assertEqual(out[0]["content"], "Let me do that.")
        self.assertEqual(out[0]["tool_calls"][0]["id"], "call_123")
        self.assertEqual(out[0]["tool_calls"][0]["function"]["name"], "Shell")
        # Ensure arguments are serialized to JSON string
        self.assertEqual(out[0]["tool_calls"][0]["function"]["arguments"], '{"cmd": "ls"}')

    def test_to_openai_message_user_tools_and_text(self):
        msg = UserMessage(content=[
            ToolResultMessageContent(tool_use_id="call_1", content="Success 1"),
            ToolResultMessageContent(tool_use_id="call_2", content="Success 2"),
            TextMessageContent(text="What is next?")
        ])
        
        out = to_openai_message(msg, add_cache_control=False)
        
        # Should flatten into THREE distinct dictionaries
        self.assertEqual(len(out), 3)
        self.assertEqual(out[0]["role"], "tool")
        self.assertEqual(out[0]["tool_call_id"], "call_1")
        
        self.assertEqual(out[1]["role"], "tool")
        self.assertEqual(out[1]["tool_call_id"], "call_2")
        
        self.assertEqual(out[2]["role"], "user")
        self.assertEqual(out[2]["content"], "What is next?")


    # ---------------------------------------------------------
    # GROUP 3: Inbound Response Parsing
    # ---------------------------------------------------------

    def test_parse_text_only_response(self):
        # Create a real LiteLLM ModelResponse using pydantic/kwargs
        resp = ModelResponse(
            id="req_text", model="gpt-4o",
            choices=[Choices(finish_reason="stop", message=Message(content="Hello world!"))]
        )
        
        msg = parse_assistant_response(resp)
        self.assertIsInstance(msg, AssistantMessage)
        self.assertEqual(msg.id, "req_text")
        self.assertEqual(msg.model, "gpt-4o")
        self.assertEqual(msg.stop_reason, "stop")
        self.assertEqual(len(msg.content), 1)
        self.assertIsInstance(msg.content[0], TextMessageContent)
        self.assertEqual(msg.content[0].text, "Hello world!")

    def test_parse_tool_call_response_and_malformed_json(self):
        from litellm.types.utils import ChatCompletionMessageToolCall, Function
        
        # Use actual Pydantic models instead of MagicMock
        tc1 = ChatCompletionMessageToolCall(
            id="call_abc",
            type="function",
            function=Function(name="ValidTool", arguments='{"key": "value"}')
        )
        
        tc2 = ChatCompletionMessageToolCall(
            id="call_bad",
            type="function",
            function=Function(name="BrokenTool", arguments='{invalid_json: oops}') # Malformed JSON
        )
        
        resp = ModelResponse(
            id="req_tools",
            model="gpt-4o",
            choices=[Choices(finish_reason="tool_calls", message=Message(content=None, tool_calls=[tc1, tc2]))]
        )
        
        msg = parse_assistant_response(resp)
        self.assertEqual(msg.stop_reason, "tool_use")  # Automatically infers stop reason
        self.assertEqual(len(msg.content), 2)
        
        # 1. Valid tool call
        self.assertIsInstance(msg.content[0], ToolUseMessageContent)
        self.assertEqual(msg.content[0].input, {"key": "value"})
        
        # 2. Malformed tool call fallback
        self.assertIsInstance(msg.content[1], ToolUseMessageContent)
        self.assertEqual(msg.content[1].input, {"raw": '{invalid_json: oops}'})
    
    def test_parse_thinking_blocks_and_usage(self):
        resp = ModelResponse(
            choices=[Choices(message=Message(
                content="Final answer", 
                reasoning_content="I should output the final answer now."
            ))],
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        )
        
        msg = parse_assistant_response(resp)
        self.assertEqual(len(msg.content), 2)
        
        # Verify Text
        self.assertIsInstance(msg.content[0], TextMessageContent)
        self.assertEqual(msg.content[0].text, "Final answer")
        
        # Verify Thinking Block
        self.assertIsInstance(msg.content[1], ThinkingMessageContent)
        self.assertEqual(msg.content[1].thinking, "I should output the final answer now.")
        
        # Verify Usage
        self.assertIsNotNone(msg.usage)
        self.assertEqual(msg.usage["prompt_tokens"], 10)


    # ---------------------------------------------------------
    # GROUP 4: High-Level Orchestration (acompletion)
    # ---------------------------------------------------------

    @patch("adapter.to_openai_message")
    @patch("litellm.acompletion", new_callable=AsyncMock)
    async def test_anthropic_prompt_caching_logic(self, mock_litellm, mock_to_openai):
        mock_litellm.return_value = ModelResponse(choices=[Choices(message=Message(content=""))])
        mock_to_openai.return_value = [{"role": "user", "content": "..."}]
        
        messages = [SystemMessage(content="s1"), UserMessage(content="u1"), 
                    AssistantMessage(content=[]), UserMessage(content="u2"), UserMessage(content="u3")]
        
        # Target an anthropic model
        await acompletion("anthropic/claude-3", [], messages)
        
        # Verify `to_openai_message` calls
        self.assertEqual(mock_to_openai.call_count, 5)
        
        # Anthropic caching rule: indices 0, 1, and N-1 must have cache_control=True
        self.assertTrue(mock_to_openai.call_args_list[0][1]["add_cache_control"]) # idx 0
        self.assertTrue(mock_to_openai.call_args_list[1][1]["add_cache_control"]) # idx 1
        self.assertFalse(mock_to_openai.call_args_list[2][1]["add_cache_control"]) # idx 2
        self.assertFalse(mock_to_openai.call_args_list[3][1]["add_cache_control"]) # idx 3
        self.assertTrue(mock_to_openai.call_args_list[4][1]["add_cache_control"]) # idx 4

    @patch("litellm.acompletion", new_callable=AsyncMock)
    async def test_ollama_local_routing(self, mock_litellm):
        mock_litellm.return_value = ModelResponse(choices=[Choices(message=Message(content=""))])
        
        # Pass empty tools to an ollama model
        await acompletion("ollama/qwen2.5-coder", tools=[], messages=[SystemMessage(content="hello")])
        
        # Verify exactly what was passed to litellm
        called_kwargs = mock_litellm.call_args[1]
        
        # 1. api_base must be populated
        self.assertEqual(called_kwargs["api_base"], "http://localhost:11434")
        
        # 2. 'tools' key must NOT exist (because it was an empty array, prevents API crash)
        self.assertNotIn("tools", called_kwargs)
        
        # 3. 'user' cache key should NOT be present for Ollama
        self.assertNotIn("user", called_kwargs)

    @patch("litellm.acompletion", new_callable=AsyncMock)
    async def test_openai_default_behavior(self, mock_litellm):
        mock_litellm.return_value = ModelResponse(choices=[Choices(message=Message(content=""))])
        
        await acompletion("gpt-4o", tools=[{"name": "test"}], messages=[])
        
        called_kwargs = mock_litellm.call_args[1]
        self.assertIn("tools", called_kwargs)
        self.assertIn("user", called_kwargs) # OpenAI Prompt Cache Key


    # ---------------------------------------------------------
    # GROUP 5: Spinner UI
    # ---------------------------------------------------------

    @patch("sys.stdout.isatty", return_value=False)
    @patch("sys.stdout.write")
    async def test_spinner_no_tty(self, mock_write, mock_isatty):
        async def dummy_task(): return "done"
        
        res = await spinner(dummy_task())
        
        self.assertEqual(res, "done")
        mock_write.assert_not_called() # No prints in CI/CD mode

    @patch("sys.stdout.isatty", return_value=True)
    @patch("sys.stdout.write")
    @patch("sys.stdout.flush")
    async def test_spinner_tty(self, mock_flush, mock_write, mock_isatty):
        async def fast_task(): return "done"
        
        res = await spinner(fast_task())
        
        self.assertEqual(res, "done")
        mock_write.assert_called() # Spinner outputted correctly
        mock_flush.assert_called()


if __name__ == "__main__":
    unittest.main()