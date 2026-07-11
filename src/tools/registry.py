"""
src/tools/registry.py — Centralised tool registry.

Base tools are always present.
MCP tools are registered at startup and rebound to the LLM lazily.
"""

from __future__ import annotations

import structlog
from langchain_core.tools import BaseTool

logger = structlog.get_logger(__name__)

# ── Base tools (always present) ──────────────────────────────────────────────
_BASE_TOOLS: list[BaseTool] = []

# ── MCP tools (populated at startup) ────────────────────────────────────────
_mcp_tools: list[BaseTool] = []

# ── Track last known count for lazy LLM rebinding ───────────────────────────
_last_tool_count: int = 0


def register_base_tools() -> None:
    """Import and register the always-present base tools."""
    global _BASE_TOOLS
    from src.tools.knowledge_base import (
        billing_search,
        compliance_search,
        knowledge_base_search,
        technical_search,
    )

    _BASE_TOOLS = [knowledge_base_search, billing_search, technical_search, compliance_search]
    logger.info("base_tools_registered", count=len(_BASE_TOOLS))


def register_mcp_tools(tools: list[BaseTool]) -> None:
    """Called at startup after MCP tools are loaded."""
    global _mcp_tools, _last_tool_count
    _mcp_tools = tools
    _last_tool_count = len(_BASE_TOOLS) + len(_mcp_tools)
    logger.info("mcp_tools_registered", count=len(_mcp_tools), names=[t.name for t in tools])


def get_tools() -> list[BaseTool]:
    """Return the full live tool set (base + MCP)."""
    return list(_BASE_TOOLS) + list(_mcp_tools)


def get_tools_by_name() -> dict[str, BaseTool]:
    """Return a name-keyed dict of all tools for fast dispatch."""
    return {t.name: t for t in get_tools()}


def tool_count_changed() -> bool:
    """True if new tools were registered since last check (triggers LLM rebind)."""
    global _last_tool_count
    current = len(get_tools())
    if current != _last_tool_count:
        _last_tool_count = current
        return True
    return False
