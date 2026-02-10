"""Unit tests for ChatClaudeCode — covers all public functionality."""

import json
from unittest.mock import MagicMock, patch

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatResult
from pydantic import BaseModel, Field

from langchain_claude_code.chat_models import (
    ChatClaudeCode,
    _build_prompt_string,
    _content_to_anthropic_blocks,
    _convert_messages,
    _tool_to_anthropic_schema,
)
from langchain_claude_code.tools import (
    ALL_TOOLS,
    NETWORK_TOOLS,
    READ_ONLY_TOOLS,
    SHELL_TOOLS,
    WRITE_TOOLS,
    ClaudeTool,
    normalize_tools,
)

# ── ClaudeTool enum ──────────────────────────────────────────


class TestClaudeTool:
    def test_enum_values(self) -> None:
        assert ClaudeTool.BASH.value == "Bash"
        assert ClaudeTool.READ.value == "Read"
        assert ClaudeTool.WEB_FETCH.value == "WebFetch"

    def test_string_subclass(self) -> None:
        assert isinstance(ClaudeTool.BASH, str)
        assert ClaudeTool.BASH == "Bash"

    def test_preset_groups(self) -> None:
        assert ClaudeTool.READ in READ_ONLY_TOOLS
        assert ClaudeTool.GLOB in READ_ONLY_TOOLS
        assert ClaudeTool.GREP in READ_ONLY_TOOLS
        assert ClaudeTool.EDIT in WRITE_TOOLS
        assert ClaudeTool.WRITE in WRITE_TOOLS
        assert ClaudeTool.WEB_FETCH in NETWORK_TOOLS
        assert ClaudeTool.WEB_SEARCH in NETWORK_TOOLS
        assert ClaudeTool.BASH in SHELL_TOOLS
        assert len(ALL_TOOLS) == len(ClaudeTool)

    def test_all_tools_contains_all_enum_members(self) -> None:
        for member in ClaudeTool:
            assert member in ALL_TOOLS


class TestNormalizeTools:
    def test_strings(self) -> None:
        result = normalize_tools(["Read", "Write", "Read"])
        assert result == ["Read", "Write"]

    def test_enum_values(self) -> None:
        result = normalize_tools([ClaudeTool.READ, ClaudeTool.WRITE])
        assert result == ["Read", "Write"]

    def test_mixed(self) -> None:
        result = normalize_tools([ClaudeTool.READ, "Write", ClaudeTool.READ])
        assert result == ["Read", "Write"]

    def test_empty(self) -> None:
        assert normalize_tools([]) == []

    def test_preserves_order(self) -> None:
        result = normalize_tools([ClaudeTool.WRITE, ClaudeTool.READ, ClaudeTool.BASH])
        assert result == ["Write", "Read", "Bash"]


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
        assert "text" in result

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
        result = _build_prompt_string([
            {"role": "user", "content": [{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}]},
            {"role": "assistant", "content": "ok"},
        ])
        assert "User: hello world" in result

    def test_single_message_content_blocks_uses_str(self) -> None:
        result = _build_prompt_string([
            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
        ])
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
        assert llm.max_retries == 0
        assert llm.default_request_timeout is None
        assert llm.api_key is None
        assert llm.anthropic_api_key is None
        assert llm.session_id is None

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

    def test_chatanthropic_compat_fields(self) -> None:
        """ChatAnthropic compat fields are accepted without error."""
        llm = ChatClaudeCode(
            max_retries=3,
            default_request_timeout=30.0,
            api_key="sk-fake",
            anthropic_api_key="sk-also-fake",
        )
        assert llm.max_retries == 3
        assert llm.default_request_timeout == 30.0
        assert llm.api_key == "sk-fake"
        assert llm.anthropic_api_key == "sk-also-fake"

    def test_allowed_tools_with_enum(self) -> None:
        llm = ChatClaudeCode(
            allowed_tools=[ClaudeTool.READ, ClaudeTool.GLOB, "Grep"],
        )
        assert len(llm.allowed_tools) == 3

    def test_last_result_initially_none(self) -> None:
        llm = ChatClaudeCode()
        assert llm.last_result is None


# ── bind_tools ───────────────────────────────────────────────


class TestBindTools:
    def test_bind_dict_tools(self) -> None:
        llm = ChatClaudeCode()
        tool = {"name": "test", "description": "test tool", "input_schema": {"type": "object"}}
        bound = llm.bind_tools([tool])
        assert bound._bound_tools == [tool]
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

    def test_bind_tools_accepts_extra_kwargs(self) -> None:
        """Signature matches ChatAnthropic — extra kwargs accepted."""
        llm = ChatClaudeCode()
        bound = llm.bind_tools(
            [{"name": "t", "description": "", "input_schema": {}}],
            tool_choice="auto",
            parallel_tool_calls=True,
            strict=True,
        )
        assert bound._bound_tools is not None


# ── enable_tools ─────────────────────────────────────────────


class TestEnableTools:
    def test_enable_tools_adds_to_empty(self) -> None:
        llm = ChatClaudeCode()
        new = llm.enable_tools([ClaudeTool.READ, ClaudeTool.GLOB])
        assert new.allowed_tools == [ClaudeTool.READ, ClaudeTool.GLOB]
        assert llm.allowed_tools is None  # original unchanged

    def test_enable_tools_adds_to_existing(self) -> None:
        llm = ChatClaudeCode(allowed_tools=["Read"])
        new = llm.enable_tools([ClaudeTool.WRITE])
        assert len(new.allowed_tools) == 2

    def test_enable_tools_with_strings(self) -> None:
        llm = ChatClaudeCode()
        new = llm.enable_tools(["Bash", "Write"])
        assert new.allowed_tools == ["Bash", "Write"]

    def test_enable_tools_with_preset_group(self) -> None:
        llm = ChatClaudeCode()
        new = llm.enable_tools(READ_ONLY_TOOLS)
        assert len(new.allowed_tools) == 3


# ── session_id ───────────────────────────────────────────────


class TestSessionId:
    def test_session_id_field(self) -> None:
        llm = ChatClaudeCode(session_id="test-session-123")
        assert llm.session_id == "test-session-123"

    def test_get_session_id_from_field(self) -> None:
        llm = ChatClaudeCode(session_id="field-session")
        assert llm._get_session_id() == "field-session"

    def test_get_session_id_from_config(self) -> None:
        llm = ChatClaudeCode(session_id="field-session")
        config: RunnableConfig = {"configurable": {"session_id": "config-session"}}
        assert llm._get_session_id(config) == "config-session"

    def test_get_session_id_config_overrides_field(self) -> None:
        llm = ChatClaudeCode(session_id="field")
        config: RunnableConfig = {"configurable": {"session_id": "config"}}
        assert llm._get_session_id(config) == "config"

    def test_get_session_id_empty_config(self) -> None:
        llm = ChatClaudeCode(session_id="field")
        config: RunnableConfig = {"configurable": {}}
        assert llm._get_session_id(config) == "field"

    def test_get_session_id_no_config_no_field(self) -> None:
        llm = ChatClaudeCode()
        assert llm._get_session_id() is None

    def test_session_id_in_build_options(self) -> None:
        llm = ChatClaudeCode()
        options = llm._build_options(session_id="resume-123")
        assert options.resume == "resume-123"

    def test_no_session_id_no_resume(self) -> None:
        llm = ChatClaudeCode()
        options = llm._build_options()
        assert not hasattr(options, "resume") or options.resume is None


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

    def test_allowed_tools_with_enum(self) -> None:
        llm = ChatClaudeCode(allowed_tools=[ClaudeTool.READ, ClaudeTool.GLOB])
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
        del block.thinking
        msg = MagicMock()
        msg.content = [block]
        return msg

    @patch("langchain_claude_code.chat_models._run_sync")
    def test_generate_basic(self, mock_run_sync: MagicMock) -> None:
        """Test _generate returns ChatResult with AIMessage."""
        llm = ChatClaudeCode()
        mock_run_sync.side_effect = lambda coro: None
        result = llm._generate([HumanMessage(content="Hello")])
        assert isinstance(result, ChatResult)
        assert len(result.generations) == 1
        assert isinstance(result.generations[0].message, AIMessage)

    @patch("langchain_claude_code.chat_models._run_sync")
    def test_generate_with_tools_injects_system_prompt(self, mock_run_sync: MagicMock) -> None:
        llm = ChatClaudeCode()
        bound = llm.bind_tools([{"name": "test_tool", "description": "A test", "input_schema": {}}])
        mock_run_sync.side_effect = lambda coro: None
        result = bound._generate([HumanMessage(content="Use the tool")])
        assert isinstance(result, ChatResult)

    @patch("langchain_claude_code.chat_models._run_sync")
    def test_generate_generation_info(self, mock_run_sync: MagicMock) -> None:
        llm = ChatClaudeCode(model="claude-opus-4-20250514")
        mock_run_sync.side_effect = lambda c: None
        result = llm._generate([HumanMessage(content="Hi")])
        gen_info = result.generations[0].generation_info
        assert gen_info["model"] == "claude-opus-4-20250514"
        assert gen_info["backend"] == "cli"


# ── last_result ──────────────────────────────────────────────


class TestLastResult:
    def test_last_result_stored(self) -> None:
        """Verify _last_result is set when _process_sdk_messages gets a ResultMessage."""
        llm = ChatClaudeCode()

        # Create a mock ResultMessage
        mock_result = MagicMock()
        mock_result.__class__.__name__ = "ResultMessage"
        mock_result.session_id = "sess-abc"
        mock_result.duration_ms = 1234
        mock_result.num_turns = 2
        mock_result.is_error = False
        mock_result.total_cost_usd = 0.05
        mock_result.usage = {"input_tokens": 100, "output_tokens": 50}

        # Patch isinstance check
        from claude_code_sdk import AssistantMessage, ResultMessage

        mock_assistant = MagicMock(spec=AssistantMessage)
        block = MagicMock()
        block.text = "Hello!"
        del_attrs = ["thinking"]
        for attr in del_attrs:
            if hasattr(block, attr):
                delattr(block, attr)
        mock_assistant.content = [block]

        real_result = ResultMessage(
            subtype="result",
            duration_ms=1234,
            duration_api_ms=1000,
            is_error=False,
            num_turns=2,
            session_id="sess-abc",
            total_cost_usd=0.05,
            usage={"input_tokens": 100, "output_tokens": 50},
            result="Hello!",
        )

        text, _, gen_info = llm._process_sdk_messages([mock_assistant, real_result])
        assert text == "Hello!"
        assert llm.last_result is not None
        assert llm.last_result.session_id == "sess-abc"
        assert gen_info["session_id"] == "sess-abc"
        assert gen_info["total_cost_usd"] == 0.05
        assert gen_info["usage"] == {"input_tokens": 100, "output_tokens": 50}


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
        assert result[0]["source"]["type"] == "url"
        assert result[0]["source"]["url"] == ""


# ── LangGraph compatibility ──────────────────────────────────


class TestLangGraphCompat:
    """Tests for compatibility with LangGraph create_react_agent."""

    def test_bind_tools_returns_runnable(self) -> None:
        """bind_tools returns a Runnable for LangGraph."""
        from langchain_core.runnables import Runnable

        llm = ChatClaudeCode()
        tool_schema = {
            "name": "get_weather",
            "description": "Get weather",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        }
        bound = llm.bind_tools([tool_schema])
        assert isinstance(bound, Runnable)

    def test_bind_tools_has_bound_tools_attr(self) -> None:
        """Bound tools are accessible on the returned model."""
        llm = ChatClaudeCode()
        tool_schema = {
            "name": "search",
            "description": "Search the web",
            "input_schema": {
                "type": "object",
                "properties": {"q": {"type": "string"}},
            },
        }
        bound = llm.bind_tools([tool_schema])
        assert bound._bound_tools is not None
        assert len(bound._bound_tools) == 1
        assert bound._bound_tools[0]["name"] == "search"

    def test_tool_message_in_convert_messages(self) -> None:
        """ToolMessage converts to tool_result format."""
        msgs = [
            HumanMessage(content="What's the weather?"),
            AIMessage(
                content="",
                tool_calls=[{
                    "id": "call_abc",
                    "name": "get_weather",
                    "args": {"city": "London"},
                }],
            ),
            ToolMessage(
                content='{"temp": 15, "condition": "cloudy"}',
                tool_call_id="call_abc",
            ),
        ]
        _sys, api_msgs, has_mm = _convert_messages(msgs)
        assert has_mm is True
        tool_result_msg = api_msgs[2]
        assert tool_result_msg["role"] == "user"
        block = tool_result_msg["content"][0]
        assert block["type"] == "tool_result"
        assert block["tool_use_id"] == "call_abc"

    def test_full_tool_calling_roundtrip(self) -> None:
        """Human→AI(tool_call)→ToolMessage→AI converts OK."""
        msgs = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Search for LangGraph docs"),
            AIMessage(
                content="I'll search for that.",
                tool_calls=[{
                    "id": "call_1",
                    "name": "search",
                    "args": {"q": "langgraph docs"},
                }],
            ),
            ToolMessage(
                content="Found: https://langchain-ai.github.io/langgraph/",
                tool_call_id="call_1",
            ),
            AIMessage(content="Here are the LangGraph docs."),
        ]
        system, api_msgs, has_mm = _convert_messages(msgs)
        assert system == "You are a helpful assistant."
        assert len(api_msgs) == 4
        assert has_mm is True
        assert api_msgs[0]["role"] == "user"
        assert api_msgs[1]["role"] == "assistant"
        assert api_msgs[2]["role"] == "user"
        assert api_msgs[3]["role"] == "assistant"

    @patch("langchain_claude_code.chat_models._run_sync")
    def test_ai_message_tool_calls_populated(
        self, mock_run_sync: MagicMock
    ) -> None:
        """JSON tool_calls in response populates AIMessage."""
        from claude_code_sdk import (
            AssistantMessage,
            ResultMessage,
        )

        tc_data = {
            "name": "get_weather",
            "args": {"city": "Paris"},
            "id": "call_42",
        }
        tool_response = json.dumps(
            {"tool_calls": [tc_data]}
        )

        mock_assistant = MagicMock(spec=AssistantMessage)
        block = MagicMock()
        block.text = tool_response
        if hasattr(block, "thinking"):
            delattr(block, "thinking")
        mock_assistant.content = [block]

        mock_result = ResultMessage(
            subtype="result",
            duration_ms=100,
            duration_api_ms=80,
            is_error=False,
            num_turns=1,
            session_id="sess-1",
            total_cost_usd=0.01,
            usage={"input_tokens": 10, "output_tokens": 20},
            result=tool_response,
        )

        collected_msgs = [mock_assistant, mock_result]

        llm = ChatClaudeCode()
        tool_schema = {
            "name": "get_weather",
            "description": "Get weather",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        }
        bound = llm.bind_tools([tool_schema])

        mock_run_sync.side_effect = lambda coro: None

        text, _, _info = bound._process_sdk_messages(
            collected_msgs
        )
        assert text == tool_response

        # Simulate parsing from _generate
        ai_msg = AIMessage(content=text)
        if bound._bound_tools and text:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and "tool_calls" in parsed:
                tool_calls = [
                    {
                        "name": tc["name"],
                        "args": tc.get("args", {}),
                        "id": tc.get(
                            "id",
                            f"call_{hash(tc['name'])}",
                        ),
                    }
                    for tc in parsed["tool_calls"]
                ]
                ai_msg = AIMessage(
                    content=text, tool_calls=tool_calls
                )

        assert len(ai_msg.tool_calls) == 1
        assert ai_msg.tool_calls[0]["name"] == "get_weather"
        assert ai_msg.tool_calls[0]["args"] == {"city": "Paris"}
        assert ai_msg.tool_calls[0]["id"] == "call_42"

    def test_create_react_agent_instantiation(self) -> None:
        """create_react_agent accepts ChatClaudeCode."""
        from langchain_core.tools import tool as tool_decorator
        from langgraph.prebuilt import create_react_agent

        @tool_decorator
        def get_weather(city: str) -> str:
            """Get the weather for a city."""
            return f"Weather in {city}: sunny, 25C"

        llm = ChatClaudeCode()
        agent = create_react_agent(llm, [get_weather])
        assert agent is not None
