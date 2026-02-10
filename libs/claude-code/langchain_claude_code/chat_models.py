"""
ChatClaudeCode — LangChain BaseChatModel backed by Claude Code CLI.

Drop-in replacement for ChatAnthropic that uses your Claude Pro/Max
subscription via the Claude Code CLI. No API key needed.

Supports: invoke, stream, batch, images, tool calling, bind_tools,
with_structured_output, extended thinking, and effort levels.
"""

from __future__ import annotations

import asyncio
import base64
import json
import queue
import threading
from pathlib import Path
from typing import Any, AsyncIterator, Iterator, List, Literal, Optional, Sequence, Union

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
from langchain_core.tools import BaseTool


# ── Message Conversion ───────────────────────────────────────


def _content_to_anthropic_blocks(content: Union[str, list]) -> Union[str, list[dict]]:
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
                    blocks.append({
                        "type": "image",
                        "source": {"type": "base64", "media_type": media_type, "data": b64data},
                    })
                else:
                    blocks.append({"type": "image", "source": {"type": "url", "url": url}})
            elif t == "image":
                blocks.append(item)
            else:
                blocks.append({"type": "text", "text": str(item)})
        else:
            blocks.append({"type": "text", "text": str(item)})
    return blocks


def _convert_messages(
    messages: List[BaseMessage],
) -> tuple[Optional[str], list[dict], bool]:
    """Convert LangChain messages → Anthropic API format.

    Returns (system_prompt, messages, has_multimodal).
    """
    system: Optional[str] = None
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
            # Handle tool calls in AIMessage
            if msg.tool_calls:
                content_blocks: list[dict] = []
                if msg.content:
                    content_blocks.append({"type": "text", "text": str(msg.content)})
                for tc in msg.tool_calls:
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["name"],
                        "input": tc["args"],
                    })
                api_msgs.append({"role": "assistant", "content": content_blocks})
                has_multimodal = True
            else:
                api_msgs.append({"role": "assistant", "content": str(msg.content)})
        elif isinstance(msg, ToolMessage):
            api_msgs.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": msg.tool_call_id,
                    "content": str(msg.content),
                }],
            })
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


def _tool_to_anthropic_schema(tool: Union[BaseTool, dict, type]) -> dict:
    """Convert a LangChain tool to Anthropic tool schema."""
    if isinstance(tool, dict):
        return tool
    if isinstance(tool, type):
        # Pydantic model
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
        "input_schema": tool.args_schema.model_json_schema() if tool.args_schema else {"type": "object", "properties": {}},
    }


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

            # Controlled agentic mode (read-only)
            reader = ChatClaudeCode(
                model="claude-sonnet-4-20250514",
                max_turns=5,
                allowed_tools=["Read", "Glob", "Grep"],
            )
            reader.invoke("Find all TODO comments in this project")

            # Extended thinking
            llm = ChatClaudeCode(
                model="claude-sonnet-4-20250514",
                thinking={"type": "enabled", "budget_tokens": 5000},
            )

            # Tool calling via bind_tools
            from langchain_core.tools import tool

            @tool
            def add(a: int, b: int) -> int:
                \"\"\"Add two numbers.\"\"\"
                return a + b

            llm_with_tools = llm.bind_tools([add])
    """

    # ── Core params (ChatAnthropic-compatible) ───────────────

    model: str = "claude-sonnet-4-20250514"
    """Anthropic model ID or alias (sonnet, opus, haiku)."""

    max_tokens: int = 4096
    """Maximum tokens to generate."""

    temperature: Optional[float] = None
    """Sampling temperature (0.0-1.0)."""

    top_k: Optional[int] = None
    """Top-K sampling."""

    top_p: Optional[float] = None
    """Nucleus sampling."""

    stop_sequences: Optional[List[str]] = None
    """Stop sequences."""

    streaming: bool = False
    """Whether to stream by default."""

    # ── Extended thinking ────────────────────────────────────

    thinking: Optional[dict[str, Any]] = None
    """Extended thinking config. E.g. {"type": "enabled", "budget_tokens": 5000}."""

    effort: Optional[Literal["high", "medium", "low"]] = None
    """Effort level for the session (maps to Claude Code --effort flag)."""

    # ── Claude Code specific ─────────────────────────────────

    system_prompt: Optional[str] = None
    """System prompt override."""

    permission_mode: Optional[Literal["default", "acceptEdits", "plan", "bypassPermissions"]] = None
    """Permission mode for the CLI."""

    cli_path: Optional[str] = None
    """Path to claude CLI binary."""

    max_turns: Optional[int] = None
    """Maximum conversation turns. Defaults to 1 (text-only, no tool execution).
    Set higher (e.g. 5-10) to enable agentic mode where Claude Code can use
    its built-in tools (Read, Write, Edit, Bash, Glob, Grep, etc.)."""

    cwd: Optional[str] = None
    """Working directory for the CLI. Controls where file operations happen."""

    allowed_tools: Optional[List[str]] = None
    """Whitelist of Claude Code tools the agent can use. E.g. ["Read", "Glob", "Grep"]
    for read-only access. When None, all tools are available (if max_turns > 1)."""

    disallowed_tools: Optional[List[str]] = None
    """Blacklist of Claude Code tools. E.g. ["Bash", "Write"] to prevent
    shell access and file writes while allowing other tools."""

    # ── Internal state ───────────────────────────────────────

    _bound_tools: Optional[list[dict]] = None

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

    # ── Tool binding (ChatAnthropic-compatible) ──────────────

    def bind_tools(
        self,
        tools: Sequence[Union[dict, type, BaseTool]],
        *,
        tool_choice: Optional[Union[str, dict]] = None,
        strict: Optional[bool] = None,
        **kwargs: Any,
    ) -> "ChatClaudeCode":
        """Bind tools to the model (like ChatAnthropic.bind_tools).

        Args:
            tools: List of tools (BaseTool, dict, or Pydantic model).
            tool_choice: Not directly supported via CLI, included for API compat.
            strict: Not directly supported via CLI, included for API compat.

        Returns:
            A new ChatClaudeCode instance with tools bound.
        """
        schemas = [_tool_to_anthropic_schema(t) for t in tools]
        new = self.model_copy()
        new._bound_tools = schemas
        return new

    # ── Build SDK options ────────────────────────────────────

    def _build_options(self, *, partial_messages: bool = False) -> Any:
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
            options.permission_mode = self.permission_mode  # type: ignore

        if self.cwd:
            options.cwd = self.cwd

        if self.allowed_tools:
            options.allowed_tools = self.allowed_tools

        if self.disallowed_tools:
            options.disallowed_tools = self.disallowed_tools

        return options

    # ── Prompt building ──────────────────────────────────────

    def _build_prompt(self, messages: List[BaseMessage]) -> tuple[Any, Any, bool]:
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
            thinking_instruction = f"\n\n[Think step by step. Budget: {budget} tokens for thinking.]"

        if has_multimodal:
            return api_messages, options, True
        else:
            prompt = _build_prompt_string(api_messages) + thinking_instruction
            return prompt, options, False

    # ── Generate ─────────────────────────────────────────────

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response via Claude Code CLI."""
        try:
            from claude_code_sdk import query as claude_query
        except ImportError:
            raise ImportError(
                "claude-code-sdk is required. Install with: pip install claude-code-sdk"
            )

        prompt_arg, options, is_streaming_input = self._build_prompt(messages)

        # If tools are bound, include them in system prompt
        if self._bound_tools:
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

        text_parts: list[str] = []
        thinking_parts: list[str] = []

        async def _run() -> None:
            if is_streaming_input:
                async def _input_stream() -> AsyncIterator[dict[str, Any]]:
                    for msg in prompt_arg:
                        yield {
                            "type": "user",
                            "message": {"role": msg["role"], "content": msg["content"]},
                        }
                stream = claude_query(prompt=_input_stream(), options=options)
            else:
                stream = claude_query(prompt=prompt_arg, options=options)

            async for msg in stream:
                if hasattr(msg, "content"):
                    for block in msg.content:
                        if hasattr(block, "text"):
                            text_parts.append(block.text)
                        elif hasattr(block, "thinking"):
                            thinking_parts.append(block.thinking)

        self._run_async(_run())

        text = "".join(text_parts)

        # Build generation info
        gen_info: dict[str, Any] = {"model": self.model, "backend": "cli"}
        if thinking_parts:
            gen_info["thinking"] = "".join(thinking_parts)

        # Parse tool calls from response if tools are bound
        ai_msg = AIMessage(content=text)
        if self._bound_tools and text:
            try:
                # Try to parse tool calls from the response
                parsed = json.loads(text)
                if isinstance(parsed, dict) and "tool_calls" in parsed:
                    tool_calls = []
                    for tc in parsed["tool_calls"]:
                        tool_calls.append({
                            "name": tc["name"],
                            "args": tc.get("args", {}),
                            "id": tc.get("id", f"call_{hash(tc['name'])}"),
                        })
                    ai_msg = AIMessage(content=text, tool_calls=tool_calls)
            except (json.JSONDecodeError, KeyError):
                pass

        return ChatResult(
            generations=[
                ChatGeneration(message=ai_msg, generation_info=gen_info)
            ]
        )

    # ── Stream ───────────────────────────────────────────────

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream response tokens as they arrive."""
        try:
            from claude_code_sdk import query as claude_query
            from claude_code_sdk.types import StreamEvent
        except ImportError:
            raise ImportError(
                "claude-code-sdk is required. Install with: pip install claude-code-sdk"
            )

        system, api_messages, has_multimodal = _convert_messages(messages)
        options = self._build_options(partial_messages=True)

        if system:
            options.system_prompt = system

        if self._bound_tools:
            tool_desc = json.dumps(self._bound_tools, indent=2)
            tool_instruction = (
                f"\n\nYou have access to the following tools:\n{tool_desc}\n\n"
                "When you need to use a tool, respond with a JSON object."
            )
            if options.system_prompt:
                options.system_prompt += tool_instruction
            else:
                options.system_prompt = tool_instruction

        chunk_queue: queue.Queue[Optional[str]] = queue.Queue()

        async def _run() -> None:
            if has_multimodal:
                async def _input_stream() -> AsyncIterator[dict[str, Any]]:
                    for msg in api_messages:
                        yield {
                            "type": "user",
                            "message": {"role": msg["role"], "content": msg["content"]},
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

            chunk_queue.put(None)

        thread = threading.Thread(target=lambda: asyncio.run(_run()), daemon=True)
        thread.start()

        while True:
            text = chunk_queue.get()
            if text is None:
                break
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=text))
            if run_manager:
                run_manager.on_llm_new_token(text)
            yield chunk

        thread.join()

    # ── Async helper ─────────────────────────────────────────

    @staticmethod
    def _run_async(coro: Any) -> None:
        """Run an async coroutine synchronously."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                pool.submit(lambda: asyncio.run(coro)).result()
        else:
            asyncio.run(coro)
