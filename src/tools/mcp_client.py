"""
src/tools/mcp_client.py — MCP server loader with per-server isolation.

Reads mcp.servers from configs/config.yaml.
Each server loads independently — one failure never blocks others.
"""

from __future__ import annotations

import asyncio
from typing import Any

import structlog
from langchain_core.tools import BaseTool

logger = structlog.get_logger(__name__)


async def _load_single_server(name: str, server_cfg: dict[str, Any]) -> list[BaseTool]:
    """Load tools from a single MCP server. Returns [] on failure."""
    if not server_cfg.get("enabled", True):
        logger.info("mcp_server_disabled", server=name)
        return []

    transport = server_cfg.get("transport", "stdio")

    try:
        if transport == "stdio":
            from langchain_mcp_adapters.client import MultiServerMCPClient

            command = server_cfg["command"]
            args = server_cfg.get("args", [])

            client = MultiServerMCPClient(
                {
                    name: {
                        "command": command,
                        "args": args,
                        "transport": "stdio",
                    }
                }
            )
            tools = await client.get_tools()

        elif transport in ("http", "sse", "streamable_http"):
            from langchain_mcp_adapters.client import MultiServerMCPClient

            url = server_cfg["url"]
            headers: dict[str, str] = {}

            # Inject auth headers from settings if specified
            if name == "tavily":
                from src.config import settings

                if settings.tavily_api_key:
                    headers["Authorization"] = f"Bearer {settings.tavily_api_key}"

            client = MultiServerMCPClient(
                {
                    name: {
                        "url": url,
                        "transport": transport,
                        "headers": headers,
                    }
                }
            )
            tools = await client.get_tools()
        else:
            logger.warning("mcp_unknown_transport", server=name, transport=transport)
            return []

        logger.info(
            "mcp_server_loaded",
            server=name,
            tool_count=len(tools),
            tool_names=[t.name for t in tools],
        )
        return tools

    except Exception as exc:
        logger.warning("mcp_server_load_failed", server=name, error=str(exc))
        return []


async def load_mcp_tools() -> list[BaseTool]:
    """Load all enabled MCP servers from config and return their tools."""
    from src.config import cfg, settings

    if not settings.mcp_enabled:
        logger.info("mcp_disabled_globally")
        return []

    servers: dict[str, Any] = cfg.mcp.get("servers", {}) if cfg.mcp else {}
    if not servers:
        logger.info("no_mcp_servers_configured")
        return []

    tasks = [_load_single_server(name, srv_cfg) for name, srv_cfg in servers.items()]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_tools: list[BaseTool] = []
    for result in results:
        if isinstance(result, Exception):
            logger.warning("mcp_gather_exception", error=str(result))
        elif isinstance(result, list):
            all_tools.extend(result)

    logger.info("mcp_tools_total", count=len(all_tools))
    return all_tools
