"""
MCP filesystem server integration.

Replaces the local file_reader tool with Anthropic's official MCP filesystem
server. The server runs as a subprocess (Node.js via npx) and is sandboxed
to the directory we specify, so it can't read files outside our project.
"""

import asyncio
from pathlib import Path
from langchain_mcp_adapters.client import MultiServerMCPClient

async def _load_filesystem_tools():
    """Async helper that actually connects and gets tools."""
    allowed_dir = str(Path.cwd().resolve())
    client = MultiServerMCPClient({
        "filesystem": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", allowed_dir],
            "transport": "stdio",
        }
    })
    return await client.get_tools()

def get_filesystem_tools():
    """
    Connect to the filesystem MCP server and return its tools.
    
    Handles both contexts:
    - Standalone script (no event loop): use asyncio.run()
    - Inside a running loop (uvicorn): use the existing loop
    """
    try:
        # Is there already a running event loop?
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No event loop, safe to use asyncio.run()
        return asyncio.run(_load_filesystem_tools())
    else:
        # A loop is already running (e.g. uvicorn). Run our corooutine on it synchronously.
        # We use nest_asyncio to allow nested event loop execution.
        import nest_asyncio
        nest_asyncio.apply()
        return loop.run_until_complete(_load_filesystem_tools())

# Quick standalone test
if __name__ == "__main__":
    print("Connecting to filesystem MCP server (may take ~30s on first run)...")
    tools = get_filesystem_tools()
    print(f"\nLoaded {len(tools)} MCP filesystem tools:")
    for t in tools:
        # Truncate description to keep output clean
        desc = (t.description or "")[:80]
        print(f"  - {t.name}: {desc}...")