"""Unit tests for ChatClaudeCode."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langchain_claude_code.chat_models import ChatClaudeCode, _convert_messages


def test_convert_messages_basic():
    msgs = [
        SystemMessage(content="Be helpful."),
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there"),
        HumanMessage(content="How are you?"),
    ]
    system, api_msgs, _ = _convert_messages(msgs)
    assert system == "Be helpful."
    assert len(api_msgs) == 3
    assert api_msgs[0] == {"role": "user", "content": "Hello"}
    assert api_msgs[1] == {"role": "assistant", "content": "Hi there"}
    assert api_msgs[2] == {"role": "user", "content": "How are you?"}


def test_convert_messages_no_system():
    msgs = [HumanMessage(content="Hello")]
    system, api_msgs, _ = _convert_messages(msgs)
    assert system is None
    assert len(api_msgs) == 1


def test_llm_type():
    llm = ChatClaudeCode()
    assert llm._llm_type == "claude-code"


def test_convert_messages_with_image():
    msgs = [
        HumanMessage(content=[
            {"type": "text", "text": "What is this?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}},
        ])
    ]
    system, api_msgs, has_multimodal = _convert_messages(msgs)
    assert system is None
    assert has_multimodal is True
    assert len(api_msgs) == 1
    content = api_msgs[0]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "What is this?"}
    assert content[1]["type"] == "image"
    assert content[1]["source"]["type"] == "base64"
    assert content[1]["source"]["media_type"] == "image/png"
    assert content[1]["source"]["data"] == "abc123"


def test_convert_messages_image_url():
    msgs = [
        HumanMessage(content=[
            {"type": "text", "text": "Describe"},
            {"type": "image_url", "image_url": "https://example.com/img.jpg"},
        ])
    ]
    _, api_msgs, has_multimodal = _convert_messages(msgs)
    assert has_multimodal is True
    content = api_msgs[0]["content"]
    assert content[1]["source"]["type"] == "url"
    assert content[1]["source"]["url"] == "https://example.com/img.jpg"


def test_identifying_params():
    llm = ChatClaudeCode(model="claude-opus-4-20250514", temperature=0.5)
    params = llm._identifying_params
    assert params["model"] == "claude-opus-4-20250514"
    assert params["temperature"] == 0.5
