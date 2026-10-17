"""MCP server exposing the wa-agent tool registry over stdio.

The agent (waagent/tools.py) and any MCP client (Claude Desktop, Cursor, ...)
share one source of truth for tools. Run:

    python -m mcp_server.server        # stdio MCP server
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server.fastmcp import FastMCP  # noqa: E402

from waagent.tools import run_tool  # noqa: E402

mcp = FastMCP("wa-agent-tools")


@mcp.tool()
def search_faq(query: str, limit: int = 3) -> dict:
    """Search the help-centre FAQ for an answer to the user's question."""
    return run_tool("search_faq", {"query": query, "limit": limit})


@mcp.tool()
def get_order_status(order_id: str) -> dict:
    """Look up an order's status and ETA by its order id."""
    return run_tool("get_order_status", {"order_id": order_id})


@mcp.tool()
def check_refund_policy(product: str = "default") -> dict:
    """Return the current refund policy."""
    return run_tool("check_refund_policy", {"product": product})


@mcp.tool()
def get_session_info(user_id: str) -> dict:
    """Return session flags (language, handoff state) for a user."""
    return run_tool("get_session_info", {"user_id": user_id})


@mcp.tool()
def escalate_to_human(reason: str) -> dict:
    """Queue the conversation for a human agent."""
    return run_tool("escalate_to_human", {"reason": reason})


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
