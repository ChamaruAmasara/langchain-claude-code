"""LangChain integration for Claude Code — use Claude Pro/Max subscription as a LangChain ChatModel."""

from langchain_claude_code.chat_models import ChatClaudeCode
from langchain_claude_code.tools import (
    ALL_TOOLS,
    NETWORK_TOOLS,
    READ_ONLY_TOOLS,
    SHELL_TOOLS,
    WRITE_TOOLS,
    ClaudeTool,
    normalize_tools,
)

__all__ = [
    "ChatClaudeCode",
    "ClaudeTool",
    "normalize_tools",
    "ALL_TOOLS",
    "READ_ONLY_TOOLS",
    "WRITE_TOOLS",
    "NETWORK_TOOLS",
    "SHELL_TOOLS",
]
__version__ = "0.1.0"
