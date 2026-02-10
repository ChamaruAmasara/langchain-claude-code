"""LangChain integration for Claude Code.

Use Claude Pro/Max subscription as a LangChain ChatModel.
"""

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
    "ALL_TOOLS",
    "NETWORK_TOOLS",
    "READ_ONLY_TOOLS",
    "SHELL_TOOLS",
    "WRITE_TOOLS",
    "ChatClaudeCode",
    "ClaudeTool",
    "normalize_tools",
]
__version__ = "0.1.0"
