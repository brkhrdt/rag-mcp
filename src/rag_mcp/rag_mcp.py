from typing import Dict, Optional, List
import uuid
import asyncio
from mcp.server.fastmcp import FastMCP


# Initialize FastMCP server
mcp = FastMCP("interactive-shell")


@mcp.tool()
async def do_something(
    arg1: List[str],
) -> str:
    """What it does

    Args:

    Returns:
    """
