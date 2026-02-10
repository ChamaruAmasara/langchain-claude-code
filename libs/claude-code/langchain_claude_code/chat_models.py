"""
ChatClaudeCode — LangChain BaseChatModel backed by Claude Code CLI.

Drop-in replacement for ChatAnthropic that uses your Claude Pro/Max
subscription via the Claude Code CLI. No API key needed.

Supports: invoke, stream, batch, images, tool calling, bind_tools,
with_structured_output, extended thinking, and effort levels.
"""

from __future__ import annotations

import asyncio
import json
import queue
import threading
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import (
    Any,
    Literal,
)

from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from langchain_claude_code.tools import ClaudeTool, normalize_tools

_SDK_IMPORT_ERR = (
    "claude-code-sdk is required. Install with: pip install claude-code-sdk"
)

# ── Message Conversion ───────────────────────────────────────


def _content_to_anthropic_blocks(content: str | list) -> str | list[dict]:
    """Convert LangChain message content to Anthropic content blocks.

    Handles text, image_url (base64 + URL), and direct Anthropic image blocks.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    blocks: list[dict] = []
    for item in content:
        if isinstance(item, str):
            blocks.append({"type": "text", "text": item})
        elif isinstance(item, dict):
            t = item.get("type", "")
            if t == "text":
                blocks.append({"type": "text", "text": item.get("text", "")})
            elif t == "image_url":
                img = item.get("image_url", {})
                url = img if isinstance(img, str) else img.get("url", "")
                if url.startswith("data:"):
                    header, b64data = url.split(",", 1)
                    media_type = header.split(":")[1].split(";")[0]
                    blocks.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": b64data,
                            },
                        }
                    )
                else:
                    blocks.append(
                        {"type": "image", "source": {"type": "url", "url": url}}
                    )
            elif t == "image":
                blocks.append(item)
            else:
                blocks.append({"type": "text", "text": str(item)})
        else:
            blocks.append({"type": "text", "text": str(item)})
    return blocks


def _convert_messages(
    messages: list[BaseMessage],
) -> tuple[str | None, list[dict], bool]:
    """Convert LangChain messages → Anthropic API format.

    Returns (system_prompt, messages, has_multimodal).
    """
    system: str | None = None
    api_msgs: list[dict] = []
    has_multimodal = False

    for msg in messages:
        if isinstance(msg, SystemMessage):
            system = str(msg.content)
        elif isinstance(msg, HumanMessage):
            content = _content_to_anthropic_blocks(msg.content)
            if isinstance(content, list):
                has_multimodal = True
            api_msgs.append({"role": "user", "content": content})
        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                content_blocks: list[dict] = []
                if msg.content:
                    content_blocks.append({"type": "text", "text": str(msg.content)})
                content_blocks.extend(
                    {
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["name"],
                        "input": tc["args"],
                    }
                    for tc in msg.tool_calls
                )
                api_msgs.append({"role": "assistant", "content": content_blocks})
                has_multimodal = True
            else:
                api_msgs.append({"role": "assistant", "content": str(msg.content)})
        elif isinstance(msg, ToolMessage):
            api_msgs.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.tool_call_id,
                            "content": str(msg.content),
                        }
                    ],
                }
            )
            has_multimodal = True
        else:
            api_msgs.append({"role": "user", "content": str(msg.content)})

    return system, api_msgs, has_multimodal


def _build_prompt_string(api_messages: list[dict]) -> str:
    """Build a plain text prompt from text-only messages."""
    if len(api_messages) == 1:
        content = api_messages[0]["content"]
        return content if isinstance(content, str) else str(content)

    parts = []
    for msg in api_messages:
        role = msg["role"].capitalize()
        content = msg["content"]
        if isinstance(content, str):
            parts.append(f"{role}: {content}")
        else:
            texts = [b.get("text", "") for b in content if b.get("type") == "text"]
            parts.append(f"{role}: {' '.join(texts)}")
    return "\n\n".join(parts)


def _tool_to_anthropic_schema(tool: BaseTool | dict | type) -> dict:
    """Convert a LangChain tool to Anthropic tool schema."""
    if isinstance(tool, dict):
        return tool
    if isinstance(tool, type):
        schema = tool.model_json_schema() if hasattr(tool, "model_json_schema") else {}
        return {
            "name": tool.__name__,
            "description": tool.__doc__ or "",
            "input_schema": schema,
        }
    # BaseTool instance
    return {
        "name": tool.name,
        "description": tool.description or "",
        "input_schema": (
            tool.args_schema.model_json_schema()
            if tool.args_schema
            else {"type": "object", "properties": {}}
        ),
    }


# ── Async runner ─────────────────────────────────────────────


def _run_sync(coro: Any) -> Any:
    """Run an async coroutine from sync context, handling event loop conflicts."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # We're inside an existing event loop — run in a separate thread
        result = [None]
        exc = [None]

        def _thread_target() -> None:
            try:
                result[0] = asyncio.run(coro)
            except Exception as e:
                exc[0] = e

        t = threading.Thread(target=_thread_target, daemon=True)
        t.start()
        t.join()
        if exc[0] is not None:
            raise exc[0]
        return result[0]
    return asyncio.run(coro)


# ── Main ChatModel ───────────────────────────────────────────


class ChatClaudeCode(BaseChatModel):
    """LangChain ChatModel using Claude Code CLI — no API key needed.

    Drop-in replacement for ChatAnthropic. Uses your Claude Pro/Max
    subscription via the Claude Code CLI subprocess.

    Supports:
      - invoke / stream / batch
      - System messages
      - Image input (base64 + URLs)
      - Tool calling (bind_tools)
      - Structured output (with_structured_output)
      - Extended thinking
      - Effort levels (low/medium/high)
      - Streaming (real token-by-token)
      - stop_sequences
      - Agentic mode (filesystem, bash, etc. via Claude Code's built-in tools)
      - Session resume via session_id

    Requirements:
      - ``claude`` CLI installed & authenticated
      - ``claude-code-sdk`` Python package

    Examples:
        .. code-block:: python

            from langchain_claude_code import ChatClaudeCode

            # Basic (safe text-only, no tool execution)
            llm = ChatClaudeCode(model="claude-sonnet-4-20250514")
            llm.invoke("Hello!")

            # Agentic mode (filesystem + bash access)
            agent = ChatClaudeCode(
                model="claude-sonnet-4-20250514",
                max_turns=10,
                permission_mode="bypassPermissions",
                cwd="/path/to/project",
            )
            agent.invoke("Read main.py and fix the bug on line 42")

            # Using ClaudeTool enum for type-safe tool config
            from langchain_claude_code import ClaudeTool, READ_ONLY_TOOLS

            reader = ChatClaudeCode(
                model="claude-sonnet-4-20250514",
                max_turns=5,
                allowed_tools=READ_ONLY_TOOLS,
            )
            reader.invoke("Find all TODO comments in this project")

            # Session resume
            result = llm.invoke(
                "Start a project",
                config={"configurable": {"session_id": "abc123"}},
            )
    """

    # ── Core params (ChatAnthropic-compatible) ───────────────

    model: str = "claude-sonnet-4-20250514"
    """Anthropic model ID or alias (sonnet, opus, haiku)."""

    max_tokens: int = 4096
    """Maximum tokens to generate."""

    temperature: float | None = None
    """Sampling temperature (0.0-1.0)."""

    top_k: int | None = None
    """Top-K sampling."""

    top_p: float | None = None
    """Nucleus sampling."""

    stop_sequences: list[str] | None = None
    """Stop sequences."""

    streaming: bool = False
    """Whether to stream by default."""

    # ── ChatAnthropic compat (accepted, limited or no-op) ────

    max_retries: int = 0
    """Accepted for ChatAnthropic compat. CLI doesn't retry; this is a no-op."""

    default_request_timeout: float | None = None
    """Accepted for ChatAnthropic compat. Not fully supported via CLI."""

    api_key: str | None = None
    """Accepted for ChatAnthropic compat. Ignored (CLI auth)."""

    anthropic_api_key: str | None = None
    """Accepted for ChatAnthropic compat. Ignored (CLI auth)."""

    # ── Extended thinking ────────────────────────────────────

    thinking: dict[str, Any] | None = None
    """Extended thinking config. E.g. {"type": "enabled", "budget_tokens": 5000}."""

    effort: Literal["high", "medium", "low"] | None = None
    """Effort level for the session (maps to Claude Code --effort flag)."""

    # ── Claude Code specific ─────────────────────────────────

    system_prompt: str | None = None
    """System prompt override."""

    permission_mode: (
        Literal["default", "acceptEdits", "plan", "bypassPermissions"] | None
    ) = None
    """Permission mode for the CLI."""

    cli_path: str | None = None
    """Path to claude CLI binary."""

    max_turns: int | None = None
    """Maximum conversation turns. Defaults to 1 (text-only, no tool execution).
    Set higher (e.g. 5-10) to enable agentic mode where Claude Code can use
    its built-in tools (Read, Write, Edit, Bash, Glob, Grep, etc.)."""

    cwd: str | None = None
    """Working directory for the CLI. Controls where file operations happen."""

    allowed_tools: list[str | ClaudeTool] | None = None
    """Whitelist of Claude Code tools the agent can use.
    Accepts strings or ClaudeTool enum values. E.g. [ClaudeTool.READ, ClaudeTool.GLOB]
    for read-only access. When None, all tools are available (if max_turns > 1)."""

    disallowed_tools: list[str | ClaudeTool] | None = None
    """Blacklist of Claude Code tools. Accepts strings or ClaudeTool enum values."""

    session_id: str | None = None
    """Session ID for resuming a previous conversation."""

    # ── Internal state ───────────────────────────────────────

    _bound_tools: list[dict] | None = None
    _last_result: Any | None = None  # Stores last ResultMessage

    model_config = {"arbitrary_types_allowed": True}

    @property
    def _llm_type(self) -> str:
        return "claude-code"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "effort": self.effort,
            "thinking": self.thinking,
            "max_turns": self.max_turns,
            "permission_mode": self.permission_mode,
        }

    @property
    def last_result(self) -> Any:
        """The last ResultMessage from the SDK, containing cost/usage/session info."""
        return self._last_result

    # ── Tool binding (ChatAnthropic-compatible) ──────────────

    def bind_tools(
        self,
        tools: Sequence[dict | type | BaseTool],
        *,
        tool_choice: str | dict | None = None,
        parallel_tool_calls: bool | None = None,
        strict: bool | None = None,
        **kwargs: Any,
    ) -> ChatClaudeCode:
        """Bind tools to the model (like ChatAnthropic.bind_tools).

        Note: Tool calling is implemented by injecting tool schemas into the
        system prompt. Proper MCP-based tool binding is planned for a future
        version using claude-agent-sdk.

        Args:
            tools: List of tools (BaseTool, dict, or Pydantic model).
            tool_choice: Not directly supported via CLI, included for API compat.
            parallel_tool_calls: Not supported via CLI, included for API compat.
            strict: Not directly supported via CLI, included for API compat.

        Returns:
            A new ChatClaudeCode instance with tools bound.
        """
        schemas = [_tool_to_anthropic_schema(t) for t in tools]
        new = self.model_copy()
        new._bound_tools = schemas
        return new

    # ── enable_tools helper ──────────────────────────────────

    def enable_tools(self, tools: list[str | ClaudeTool]) -> ChatClaudeCode:
        """Return a copy with additional allowed tools.

        Args:
            tools: Tools to add to the allowed list.

        Returns:
            New ChatClaudeCode with the combined allowed_tools.
        """
        existing = list(self.allowed_tools or [])
        combined = existing + list(tools)
        return self.model_copy(update={"allowed_tools": combined})

    # ── Build SDK options ────────────────────────────────────

    def _get_session_id(self, config: RunnableConfig | None = None) -> str | None:
        """Extract session_id from config or fall back to instance field."""
        if config:
            configurable = config.get("configurable", {})
            sid = configurable.get("session_id")
            if sid:
                return sid
        return self.session_id

    def _build_options(
        self,
        *,
        partial_messages: bool = False,
        session_id: str | None = None,
    ) -> Any:
        """Build ClaudeCodeOptions from model params."""
        from claude_code_sdk import ClaudeCodeOptions

        extra_args: dict[str, str | None] = {}

        if self.effort:
            extra_args["effort"] = self.effort

        options = ClaudeCodeOptions(
            model=self.model,
            system_prompt=self.system_prompt,
            max_turns=self.max_turns or 1,
            include_partial_messages=partial_messages,
            extra_args=extra_args,
        )

        if self.permission_mode:
            options.permission_mode = self.permission_mode  # type: ignore[assignment]

        if self.cwd:
            options.cwd = self.cwd

        if self.allowed_tools:
            options.allowed_tools = normalize_tools(self.allowed_tools)

        if self.disallowed_tools:
            options.disallowed_tools = normalize_tools(self.disallowed_tools)

        if session_id:
            options.resume = session_id

        return options

    # ── Prompt building ──────────────────────────────────────

    def _inject_tool_system_prompt(self, options: Any) -> None:
        """Inject bound tool schemas into system prompt if tools are bound.

        NOTE: This is a temporary approach. Proper MCP-based tool binding
        is planned for a future version using claude-agent-sdk.
        """
        if not self._bound_tools:
            return

        tool_desc = json.dumps(self._bound_tools, indent=2)
        tool_instruction = (
            f"\n\nYou have access to the following tools:\n{tool_desc}\n\n"
            "When you need to use a tool, respond with a JSON object containing "
            '"tool_calls" with "name" and "args" fields.'
        )
        if options.system_prompt:
            options.system_prompt += tool_instruction
        else:
            options.system_prompt = tool_instruction

    def _build_prompt(self, messages: list[BaseMessage]) -> tuple[Any, Any, bool]:
        """Build prompt and options from messages.

        Returns (prompt_arg, options, is_streaming_input).
        """
        system, api_messages, has_multimodal = _convert_messages(messages)
        options = self._build_options(partial_messages=False)

        if system:
            options.system_prompt = system

        # Inject thinking into the prompt if configured
        thinking_instruction = ""
        if self.thinking and self.thinking.get("type") == "enabled":
            budget = self.thinking.get("budget_tokens", 5000)
            thinking_instruction = (
                f"\n\n[Think step by step. Budget: {budget} tokens for thinking.]"
            )

        if has_multimodal:
            return api_messages, options, True
        prompt = _build_prompt_string(api_messages) + thinking_instruction
        return prompt, options, False

    # ── Process SDK messages ─────────────────────────────────

    def _process_sdk_messages(
        self, messages: list[Any]
    ) -> tuple[str, list[str], dict[str, Any]]:
        """Extract text, thinking, and generation_info from SDK messages.

        Returns (text, thinking_parts, generation_info).
        """
        from claude_code_sdk import AssistantMessage, ResultMessage

        text_parts: list[str] = []
        thinking_parts: list[str] = []
        gen_info: dict[str, Any] = {"model": self.model, "backend": "cli"}

        for msg in messages:
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if hasattr(block, "text"):
                        text_parts.append(block.text)
                    elif hasattr(block, "thinking"):
                        thinking_parts.append(block.thinking)
            elif isinstance(msg, ResultMessage):
                self._last_result = msg
                gen_info["session_id"] = msg.session_id
                gen_info["duration_ms"] = msg.duration_ms
                gen_info["num_turns"] = msg.num_turns
                gen_info["is_error"] = msg.is_error
                if msg.total_cost_usd is not None:
                    gen_info["total_cost_usd"] = msg.total_cost_usd
                if msg.usage:
                    gen_info["usage"] = msg.usage

        if thinking_parts:
            gen_info["thinking"] = "".join(thinking_parts)

        return "".join(text_parts), thinking_parts, gen_info

    # ── Generate ─────────────────────────────────────────────

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response via Claude Code CLI."""
        try:
            from claude_code_sdk import query as claude_query
        except ImportError:
            raise ImportError(_SDK_IMPORT_ERR)

        system, api_messages, has_multimodal = _convert_messages(messages)
        session_id = self._get_session_id(kwargs.get("config"))
        options = self._build_options(partial_messages=False, session_id=session_id)

        if system:
            options.system_prompt = system

        self._inject_tool_system_prompt(options)

        # Build thinking instruction
        thinking_instruction = ""
        if self.thinking and self.thinking.get("type") == "enabled":
            budget = self.thinking.get("budget_tokens", 5000)
            thinking_instruction = (
                f"\n\n[Think step by step. Budget: {budget} tokens for thinking.]"
            )

        collected: list[Any] = []

        async def _run() -> None:
            if has_multimodal:

                async def _input_stream() -> AsyncIterator[dict[str, Any]]:
                    for msg in api_messages:
                        yield {
                            "type": "user",
                            "message": {
                                "role": msg["role"],
                                "content": msg["content"],
                            },
                        }

                stream = claude_query(prompt=_input_stream(), options=options)
            else:
                prompt = _build_prompt_string(api_messages) + thinking_instruction
                stream = claude_query(prompt=prompt, options=options)

            collected.extend([msg async for msg in stream])

        _run_sync(_run())

        text, _thinking_parts, gen_info = self._process_sdk_messages(collected)

        # Parse tool calls from response if tools are bound
        ai_msg = AIMessage(content=text)
        if self._bound_tools and text:
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict) and "tool_calls" in parsed:
                    tool_calls = [
                        {
                            "name": tc["name"],
                            "args": tc.get("args", {}),
                            "id": tc.get("id", f"call_{hash(tc['name'])}"),
                        }
                        for tc in parsed["tool_calls"]
                    ]
                    ai_msg = AIMessage(content=text, tool_calls=tool_calls)
            except (json.JSONDecodeError, KeyError):
                pass

        return ChatResult(
            generations=[ChatGeneration(message=ai_msg, generation_info=gen_info)]
        )

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async generate a response via Claude Code CLI."""
        try:
            from claude_code_sdk import query as claude_query
        except ImportError:
            raise ImportError(_SDK_IMPORT_ERR)

        system, api_messages, has_multimodal = _convert_messages(messages)
        session_id = self._get_session_id(kwargs.get("config"))
        options = self._build_options(partial_messages=False, session_id=session_id)

        if system:
            options.system_prompt = system

        self._inject_tool_system_prompt(options)

        thinking_instruction = ""
        if self.thinking and self.thinking.get("type") == "enabled":
            budget = self.thinking.get("budget_tokens", 5000)
            thinking_instruction = (
                f"\n\n[Think step by step. Budget: {budget} tokens for thinking.]"
            )

        collected: list[Any] = []

        if has_multimodal:

            async def _input_stream() -> AsyncIterator[dict[str, Any]]:
                for msg in api_messages:
                    yield {
                        "type": "user",
                        "message": {
                            "role": msg["role"],
                            "content": msg["content"],
                        },
                    }

            stream = claude_query(prompt=_input_stream(), options=options)
        else:
            prompt = _build_prompt_string(api_messages) + thinking_instruction
            stream = claude_query(prompt=prompt, options=options)

        collected = [msg async for msg in stream]

        text, _thinking_parts, gen_info = self._process_sdk_messages(collected)

        ai_msg = AIMessage(content=text)
        if self._bound_tools and text:
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict) and "tool_calls" in parsed:
                    tool_calls = [
                        {
                            "name": tc["name"],
                            "args": tc.get("args", {}),
                            "id": tc.get("id", f"call_{hash(tc['name'])}"),
                        }
                        for tc in parsed["tool_calls"]
                    ]
                    ai_msg = AIMessage(content=text, tool_calls=tool_calls)
            except (json.JSONDecodeError, KeyError):
                pass

        return ChatResult(
            generations=[ChatGeneration(message=ai_msg, generation_info=gen_info)]
        )

    # ── Stream ───────────────────────────────────────────────

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream response tokens as they arrive.

        Uses a background thread + queue pattern for thread-safe sync streaming.
        """
        try:
            from claude_code_sdk import query as claude_query
            from claude_code_sdk.types import StreamEvent
        except ImportError:
            raise ImportError(_SDK_IMPORT_ERR)

        system, api_messages, has_multimodal = _convert_messages(messages)
        session_id = self._get_session_id(kwargs.get("config"))
        options = self._build_options(partial_messages=True, session_id=session_id)

        if system:
            options.system_prompt = system

        self._inject_tool_system_prompt(options)

        # Sentinel for end of stream
        _DONE = object()
        chunk_queue: queue.Queue = queue.Queue()

        async def _run() -> None:
            try:
                if has_multimodal:

                    async def _input_stream() -> AsyncIterator[dict[str, Any]]:
                        for msg in api_messages:
                            yield {
                                "type": "user",
                                "message": {
                                    "role": msg["role"],
                                    "content": msg["content"],
                                },
                            }

                    prompt_arg: Any = _input_stream()
                else:
                    prompt_arg = _build_prompt_string(api_messages)

                async for msg in claude_query(prompt=prompt_arg, options=options):
                    if isinstance(msg, StreamEvent):
                        event = msg.event
                        if isinstance(event, dict):
                            evt_type = event.get("type", "")
                            if evt_type == "content_block_delta":
                                delta = event.get("delta", {})
                                text = delta.get("text", "")
                                if text:
                                    chunk_queue.put(text)
                    else:
                        # AssistantMessage or ResultMessage
                        from claude_code_sdk import ResultMessage

                        if isinstance(msg, ResultMessage):
                            self._last_result = msg
            except Exception as e:
                chunk_queue.put(e)
            finally:
                chunk_queue.put(_DONE)

        thread = threading.Thread(target=lambda: asyncio.run(_run()), daemon=True)
        thread.start()

        while True:
            item = chunk_queue.get()
            if item is _DONE:
                break
            if isinstance(item, Exception):
                raise item
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=item))
            if run_manager:
                run_manager.on_llm_new_token(item)
            yield chunk

        thread.join()

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Async stream response tokens."""
        try:
            from claude_code_sdk import ResultMessage
            from claude_code_sdk import query as claude_query
            from claude_code_sdk.types import StreamEvent
        except ImportError:
            raise ImportError(_SDK_IMPORT_ERR)

        system, api_messages, has_multimodal = _convert_messages(messages)
        session_id = self._get_session_id(kwargs.get("config"))
        options = self._build_options(partial_messages=True, session_id=session_id)

        if system:
            options.system_prompt = system

        self._inject_tool_system_prompt(options)

        if has_multimodal:

            async def _input_stream() -> AsyncIterator[dict[str, Any]]:
                for msg in api_messages:
                    yield {
                        "type": "user",
                        "message": {
                            "role": msg["role"],
                            "content": msg["content"],
                        },
                    }

            prompt_arg: Any = _input_stream()
        else:
            prompt_arg = _build_prompt_string(api_messages)

        async for msg in claude_query(prompt=prompt_arg, options=options):
            if isinstance(msg, StreamEvent):
                event = msg.event
                if isinstance(event, dict):
                    evt_type = event.get("type", "")
                    if evt_type == "content_block_delta":
                        delta = event.get("delta", {})
                        text = delta.get("text", "")
                        if text:
                            chunk = ChatGenerationChunk(
                                message=AIMessageChunk(content=text)
                            )
                            if run_manager:
                                await run_manager.on_llm_new_token(text)
                            yield chunk
            elif isinstance(msg, ResultMessage):
                self._last_result = msg
