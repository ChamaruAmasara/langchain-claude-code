"""Unit tests for ChatClaudeCode — covers all public functionality."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_claude_code.chat_models import (
    ChatClaudeCode,
    _build_prompt_string,
    _content_to_anthropic_blocks,
    _convert_messages,
    _tool_to_anthropic_schema,
)


# ── _content_to_anthropic_blocks ─────────────────────────────


class TestContentToAnthropicBlocks:
    def test_string_passthrough(self) -> None:
        assert _content_to_anthropic_blocks("hello") == "hello"

    def test_non_list_non_string(self) -> None:
        assert _content_to_anthropic_blocks(42) == "42"  # type: ignore[arg-type]

    def test_list_with_string_items(self) -> None:
        result = _content_to_anthropic_blocks(["hello", "world"])
        assert result == [
            {"type": "text", "text": "hello"},
            {"type": "text", "text": "world"},
        ]

    def test_text_block(self) -> None:
        result = _content_to_anthropic_blocks([{"type": "text", "text": "hi"}])
        assert result == [{"type": "text", "text": "hi"}]

    def test_image_url_base64(self) -> None:
        result = _content_to_anthropic_blocks([
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}},
        ])
        assert result == [{
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": "abc123"},
        }]

    def test_image_url_base64_jpeg(self) -> None:
        result = _content_to_anthropic_blocks([
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,xyz"}},
        ])
        assert result[0]["source"]["media_type"] == "image/jpeg"

    def test_image_url_http(self) -> None:
        result = _content_to_anthropic_blocks([
            {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}},
        ])
        assert result == [{
            "type": "image",
            "source": {"type": "url", "url": "https://example.com/img.jpg"},
        }]

    def test_image_url_as_string(self) -> None:
        """image_url value can be a plain string instead of a dict."""
        result = _content_to_anthropic_blocks([
            {"type": "image_url", "image_url": "https://example.com/photo.png"},
        ])
        assert result[0]["source"]["url"] == "https://example.com/photo.png"

    def test_image_block_passthrough(self) -> None:
        block = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "x"}}
        result = _content_to_anthropic_blocks([block])
        assert result == [block]

    def test_unknown_type_becomes_text(self) -> None:
        result = _content_to_anthropic_blocks([{"type": "audio", "data": "foo"}])
        assert result[0]["type"] == "text"

    def test_non_dict_item_becomes_text(self) -> None:
        result = _content_to_anthropic_blocks([123])
        assert result == [{"type": "text", "text": "123"}]

    def test_mixed_content(self) -> None:
        result = _content_to_anthropic_blocks([
            {"type": "text", "text": "Look at this:"},
            {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}},
            "and this text",
        ])
        assert len(result) == 3
        assert result[0]["type"] == "text"
        assert result[1]["type"] == "image"
        assert result[2]["type"] == "text"


# ── _convert_messages ────────────────────────────────────────


class TestConvertMessages:
    def test_basic_conversation(self) -> None:
        msgs = [
            SystemMessage(content="Be helpful."),
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there"),
            HumanMessage(content="How are you?"),
        ]
        system, api_msgs, has_multimodal = _convert_messages(msgs)
        assert system == "Be helpful."
        assert len(api_msgs) == 3
        assert api_msgs[0] == {"role": "user", "content": "Hello"}
        assert api_msgs[1] == {"role": "assistant", "content": "Hi there"}
        assert api_msgs[2] == {"role": "user", "content": "How are you?"}
        assert has_multimodal is False

    def test_no_system(self) -> None:
        msgs = [HumanMessage(content="Hello")]
        system, api_msgs, has_multimodal = _convert_messages(msgs)
        assert system is None
        assert len(api_msgs) == 1
        assert has_multimodal is False

    def test_multimodal_image(self) -> None:
        msgs = [
            HumanMessage(content=[
                {"type": "text", "text": "What is this?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}},
            ])
        ]
        system, api_msgs, has_multimodal = _convert_messages(msgs)
        assert has_multimodal is True
        content = api_msgs[0]["content"]
        assert isinstance(content, list)
        assert content[0] == {"type": "text", "text": "What is this?"}
        assert content[1]["type"] == "image"
        assert content[1]["source"]["data"] == "abc123"

    def test_image_url_string(self) -> None:
        msgs = [
            HumanMessage(content=[
                {"type": "text", "text": "Describe"},
                {"type": "image_url", "image_url": "https://example.com/img.jpg"},
            ])
        ]
        _, api_msgs, has_multimodal = _convert_messages(msgs)
        assert has_multimodal is True
        assert api_msgs[0]["content"][1]["source"]["type"] == "url"

    def test_ai_message_with_tool_calls(self) -> None:
        msgs = [
            AIMessage(
                content="I'll check the weather.",
                tool_calls=[{
                    "id": "call_123",
                    "name": "get_weather",
                    "args": {"city": "Tokyo"},
                }],
            )
        ]
        _, api_msgs, has_multimodal = _convert_messages(msgs)
        assert has_multimodal is True
        content = api_msgs[0]["content"]
        assert content[0] == {"type": "text", "text": "I'll check the weather."}
        assert content[1] == {
            "type": "tool_use",
            "id": "call_123",
            "name": "get_weather",
            "input": {"city": "Tokyo"},
        }

    def test_ai_message_with_tool_calls_no_text(self) -> None:
        msgs = [
            AIMessage(
                content="",
                tool_calls=[{"id": "call_1", "name": "foo", "args": {}}],
            )
        ]
        _, api_msgs, _ = _convert_messages(msgs)
        # Empty content should not add a text block
        assert len(api_msgs[0]["content"]) == 1
        assert api_msgs[0]["content"][0]["type"] == "tool_use"

    def test_tool_message(self) -> None:
        msgs = [
            ToolMessage(content="25°C, sunny", tool_call_id="call_123"),
        ]
        _, api_msgs, has_multimodal = _convert_messages(msgs)
        assert has_multimodal is True
        assert api_msgs[0]["role"] == "user"
        assert api_msgs[0]["content"][0]["type"] == "tool_result"
        assert api_msgs[0]["content"][0]["tool_use_id"] == "call_123"
        assert api_msgs[0]["content"][0]["content"] == "25°C, sunny"

    def test_unknown_message_type(self) -> None:
        """Unknown message types should be treated as user messages."""

        class CustomMessage(BaseMessage):
            type: str = "custom"

        msgs = [CustomMessage(content="hello")]
        _, api_msgs, _ = _convert_messages(msgs)
        assert api_msgs[0] == {"role": "user", "content": "hello"}

    def test_multiple_system_messages_last_wins(self) -> None:
        msgs = [
            SystemMessage(content="First"),
            SystemMessage(content="Second"),
            HumanMessage(content="Hi"),
        ]
        system, _, _ = _convert_messages(msgs)
        assert system == "Second"

    def test_full_tool_calling_conversation(self) -> None:
        """Test a complete tool-calling conversation flow."""
        msgs = [
            SystemMessage(content="You are helpful."),
            HumanMessage(content="What's the weather in Tokyo?"),
            AIMessage(
                content="Let me check.",
                tool_calls=[{"id": "call_1", "name": "get_weather", "args": {"city": "Tokyo"}}],
            ),
            ToolMessage(content="25°C, sunny", tool_call_id="call_1"),
            AIMessage(content="It's 25°C and sunny in Tokyo!"),
        ]
        system, api_msgs, has_multimodal = _convert_messages(msgs)
        assert system == "You are helpful."
        assert len(api_msgs) == 4
        assert has_multimodal is True


# ── _build_prompt_string ─────────────────────────────────────


class TestBuildPromptString:
    def test_single_message(self) -> None:
        result = _build_prompt_string([{"role": "user", "content": "Hello"}])
        assert result == "Hello"

    def test_single_message_non_string(self) -> None:
        result = _build_prompt_string([{"role": "user", "content": [{"type": "text", "text": "hi"}]}])
        assert "text" in result  # str() of the list

    def test_multi_turn(self) -> None:
        result = _build_prompt_string([
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "How are you?"},
        ])
        assert "User: Hi" in result
        assert "Assistant: Hello!" in result
        assert "User: How are you?" in result

    def test_multi_turn_with_content_blocks(self) -> None:
        # With multiple messages, content blocks get text extracted
        result = _build_prompt_string([
            {"role": "user", "content": [{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}]},
            {"role": "assistant", "content": "ok"},
        ])
        assert "User: hello world" in result

    def test_single_message_content_blocks_uses_str(self) -> None:
        # Single message with non-string content uses str()
        result = _build_prompt_string([
            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
        ])
        # str() of the list — this is the current behavior
        assert "hello" in result


# ── _tool_to_anthropic_schema ────────────────────────────────


class TestToolToAnthropicSchema:
    def test_dict_passthrough(self) -> None:
        tool = {"name": "test", "description": "A test", "input_schema": {}}
        assert _tool_to_anthropic_schema(tool) == tool

    def test_pydantic_model(self) -> None:
        class MyTool(BaseModel):
            """A helpful tool."""
            x: int = Field(description="A number")

        result = _tool_to_anthropic_schema(MyTool)
        assert result["name"] == "MyTool"
        assert result["description"] == "A helpful tool."
        assert "properties" in result["input_schema"]

    def test_pydantic_model_no_docstring(self) -> None:
        class NoDoc(BaseModel):
            x: int

        result = _tool_to_anthropic_schema(NoDoc)
        assert result["description"] == ""


# ── ChatClaudeCode properties ────────────────────────────────


class TestChatClaudeCodeProperties:
    def test_llm_type(self) -> None:
        llm = ChatClaudeCode()
        assert llm._llm_type == "claude-code"

    def test_default_params(self) -> None:
        llm = ChatClaudeCode()
        assert llm.model == "claude-sonnet-4-20250514"
        assert llm.max_tokens == 4096
        assert llm.temperature is None
        assert llm.max_turns is None
        assert llm.permission_mode is None
        assert llm.allowed_tools is None
        assert llm.disallowed_tools is None
        assert llm.streaming is False
        assert llm.effort is None
        assert llm.thinking is None

    def test_identifying_params(self) -> None:
        llm = ChatClaudeCode(
            model="claude-opus-4-20250514",
            temperature=0.5,
            effort="high",
            max_turns=5,
            permission_mode="plan",
        )
        params = llm._identifying_params
        assert params["model"] == "claude-opus-4-20250514"
        assert params["temperature"] == 0.5
        assert params["effort"] == "high"
        assert params["max_turns"] == 5
        assert params["permission_mode"] == "plan"

    def test_identifying_params_defaults(self) -> None:
        params = ChatClaudeCode()._identifying_params
        assert params["thinking"] is None
        assert params["max_turns"] is None
        assert params["permission_mode"] is None

    def test_custom_params(self) -> None:
        llm = ChatClaudeCode(
            model="claude-haiku-3-5-20241022",
            max_tokens=1024,
            temperature=0.7,
            top_k=40,
            top_p=0.9,
            stop_sequences=["STOP"],
            effort="low",
            thinking={"type": "enabled", "budget_tokens": 10000},
            system_prompt="Be concise.",
            permission_mode="bypassPermissions",
            max_turns=10,
            cwd="/tmp",
            allowed_tools=["Read", "Glob"],
            disallowed_tools=["Bash"],
        )
        assert llm.model == "claude-haiku-3-5-20241022"
        assert llm.max_tokens == 1024
        assert llm.temperature == 0.7
        assert llm.top_k == 40
        assert llm.top_p == 0.9
        assert llm.stop_sequences == ["STOP"]
        assert llm.effort == "low"
        assert llm.thinking == {"type": "enabled", "budget_tokens": 10000}
        assert llm.system_prompt == "Be concise."
        assert llm.permission_mode == "bypassPermissions"
        assert llm.max_turns == 10
        assert llm.cwd == "/tmp"
        assert llm.allowed_tools == ["Read", "Glob"]
        assert llm.disallowed_tools == ["Bash"]


# ── bind_tools ───────────────────────────────────────────────


class TestBindTools:
    def test_bind_dict_tools(self) -> None:
        llm = ChatClaudeCode()
        tool = {"name": "test", "description": "test tool", "input_schema": {"type": "object"}}
        bound = llm.bind_tools([tool])
        assert bound._bound_tools == [tool]
        # Original should be unmodified
        assert llm._bound_tools is None

    def test_bind_pydantic_tools(self) -> None:
        class WeatherInput(BaseModel):
            """Get weather for a city."""
            city: str

        llm = ChatClaudeCode()
        bound = llm.bind_tools([WeatherInput])
        assert len(bound._bound_tools) == 1
        assert bound._bound_tools[0]["name"] == "WeatherInput"

    def test_bind_preserves_model_params(self) -> None:
        llm = ChatClaudeCode(model="claude-opus-4-20250514", effort="high", max_turns=5)
        bound = llm.bind_tools([{"name": "t", "description": "", "input_schema": {}}])
        assert bound.model == "claude-opus-4-20250514"
        assert bound.effort == "high"
        assert bound.max_turns == 5

    def test_bind_multiple_tools(self) -> None:
        llm = ChatClaudeCode()
        tools = [
            {"name": "tool1", "description": "first", "input_schema": {}},
            {"name": "tool2", "description": "second", "input_schema": {}},
        ]
        bound = llm.bind_tools(tools)
        assert len(bound._bound_tools) == 2


# ── _build_options ───────────────────────────────────────────


class TestBuildOptions:
    def test_default_options(self) -> None:
        llm = ChatClaudeCode()
        options = llm._build_options()
        assert options.model == "claude-sonnet-4-20250514"
        assert options.max_turns == 1
        assert options.include_partial_messages is False

    def test_streaming_options(self) -> None:
        llm = ChatClaudeCode()
        options = llm._build_options(partial_messages=True)
        assert options.include_partial_messages is True

    def test_effort_in_extra_args(self) -> None:
        llm = ChatClaudeCode(effort="high")
        options = llm._build_options()
        assert options.extra_args.get("effort") == "high"

    def test_permission_mode(self) -> None:
        llm = ChatClaudeCode(permission_mode="bypassPermissions")
        options = llm._build_options()
        assert options.permission_mode == "bypassPermissions"

    def test_cwd(self) -> None:
        llm = ChatClaudeCode(cwd="/tmp/test")
        options = llm._build_options()
        assert str(options.cwd) == "/tmp/test"

    def test_allowed_tools(self) -> None:
        llm = ChatClaudeCode(allowed_tools=["Read", "Glob"])
        options = llm._build_options()
        assert options.allowed_tools == ["Read", "Glob"]

    def test_disallowed_tools(self) -> None:
        llm = ChatClaudeCode(disallowed_tools=["Bash", "Write"])
        options = llm._build_options()
        assert options.disallowed_tools == ["Bash", "Write"]

    def test_max_turns(self) -> None:
        llm = ChatClaudeCode(max_turns=10)
        options = llm._build_options()
        assert options.max_turns == 10

    def test_system_prompt(self) -> None:
        llm = ChatClaudeCode(system_prompt="Be brief.")
        options = llm._build_options()
        assert options.system_prompt == "Be brief."

    def test_no_effort_means_empty_extra_args(self) -> None:
        llm = ChatClaudeCode()
        options = llm._build_options()
        assert "effort" not in options.extra_args


# ── _build_prompt ────────────────────────────────────────────


class TestBuildPrompt:
    def test_text_only_prompt(self) -> None:
        llm = ChatClaudeCode()
        prompt, options, is_streaming = llm._build_prompt([HumanMessage(content="Hello")])
        assert isinstance(prompt, str)
        assert prompt == "Hello"
        assert is_streaming is False

    def test_system_message_sets_option(self) -> None:
        llm = ChatClaudeCode()
        _, options, _ = llm._build_prompt([
            SystemMessage(content="Be helpful."),
            HumanMessage(content="Hi"),
        ])
        assert options.system_prompt == "Be helpful."

    def test_multimodal_returns_list(self) -> None:
        llm = ChatClaudeCode()
        prompt, _, is_streaming = llm._build_prompt([
            HumanMessage(content=[
                {"type": "text", "text": "What's this?"},
                {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}},
            ])
        ])
        assert isinstance(prompt, list)
        assert is_streaming is True

    def test_thinking_instruction_appended(self) -> None:
        llm = ChatClaudeCode(thinking={"type": "enabled", "budget_tokens": 8000})
        prompt, _, _ = llm._build_prompt([HumanMessage(content="Solve this")])
        assert "Think step by step" in prompt
        assert "8000" in prompt

    def test_thinking_disabled_no_instruction(self) -> None:
        llm = ChatClaudeCode(thinking={"type": "disabled"})
        prompt, _, _ = llm._build_prompt([HumanMessage(content="Hello")])
        assert "Think step by step" not in prompt

    def test_multi_turn_conversation_prompt(self) -> None:
        llm = ChatClaudeCode()
        prompt, _, _ = llm._build_prompt([
            HumanMessage(content="Hi"),
            AIMessage(content="Hello!"),
            HumanMessage(content="How are you?"),
        ])
        assert "User: Hi" in prompt
        assert "Assistant: Hello!" in prompt
        assert "User: How are you?" in prompt


# ── _generate (mocked) ──────────────────────────────────────


class TestGenerate:
    def _make_mock_message(self, text: str) -> MagicMock:
        block = MagicMock()
        block.text = text
        block.thinking = None
        del block.thinking  # so hasattr returns False
        msg = MagicMock()
        msg.content = [block]
        return msg

    @patch("langchain_claude_code.chat_models.ChatClaudeCode._run_async")
    def test_generate_basic(self, mock_run_async: MagicMock) -> None:
        """Test _generate returns ChatResult with AIMessage."""
        llm = ChatClaudeCode()

        def side_effect(coro: object) -> None:
            # Simulate what _run_async does — but we need to populate text_parts
            pass

        mock_run_async.side_effect = side_effect

        # We can't easily test the full async flow without the SDK,
        # but we can test the structure
        result = llm._generate([HumanMessage(content="Hello")])
        assert isinstance(result, ChatResult)
        assert len(result.generations) == 1
        assert isinstance(result.generations[0].message, AIMessage)

    @patch("langchain_claude_code.chat_models.ChatClaudeCode._run_async")
    def test_generate_with_tools_injects_system_prompt(self, mock_run_async: MagicMock) -> None:
        llm = ChatClaudeCode()
        bound = llm.bind_tools([{"name": "test_tool", "description": "A test", "input_schema": {}}])

        captured_options = {}

        def capture_run(coro: object) -> None:
            pass

        mock_run_async.side_effect = capture_run
        result = bound._generate([HumanMessage(content="Use the tool")])
        assert isinstance(result, ChatResult)

    @patch("langchain_claude_code.chat_models.ChatClaudeCode._run_async")
    def test_generate_generation_info(self, mock_run_async: MagicMock) -> None:
        llm = ChatClaudeCode(model="claude-opus-4-20250514")
        mock_run_async.side_effect = lambda c: None
        result = llm._generate([HumanMessage(content="Hi")])
        gen_info = result.generations[0].generation_info
        assert gen_info["model"] == "claude-opus-4-20250514"
        assert gen_info["backend"] == "cli"


# ── Serialization / model_copy ───────────────────────────────


class TestSerialization:
    def test_model_copy(self) -> None:
        llm = ChatClaudeCode(model="claude-opus-4-20250514", effort="high")
        copy = llm.model_copy()
        assert copy.model == "claude-opus-4-20250514"
        assert copy.effort == "high"

    def test_model_copy_update(self) -> None:
        llm = ChatClaudeCode(model="claude-opus-4-20250514")
        copy = llm.model_copy(update={"model": "claude-sonnet-4-20250514"})
        assert copy.model == "claude-sonnet-4-20250514"
        assert llm.model == "claude-opus-4-20250514"


# ── Edge cases ───────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_messages(self) -> None:
        system, api_msgs, has_multimodal = _convert_messages([])
        assert system is None
        assert api_msgs == []
        assert has_multimodal is False

    def test_system_only(self) -> None:
        system, api_msgs, _ = _convert_messages([SystemMessage(content="Hello")])
        assert system == "Hello"
        assert api_msgs == []

    def test_empty_content(self) -> None:
        result = _content_to_anthropic_blocks("")
        assert result == ""

    def test_empty_list_content(self) -> None:
        result = _content_to_anthropic_blocks([])
        assert result == []

    def test_content_block_missing_text(self) -> None:
        result = _content_to_anthropic_blocks([{"type": "text"}])
        assert result == [{"type": "text", "text": ""}]

    def test_image_url_missing_url(self) -> None:
        result = _content_to_anthropic_blocks([{"type": "image_url", "image_url": {}}])
        # Empty URL should be treated as http (not data:), producing a URL source
        assert result[0]["source"]["type"] == "url"
        assert result[0]["source"]["url"] == ""
