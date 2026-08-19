"""
Section 5 (bonus) - Swapping the mock Notion server for the real thing
==========================================================================
Same idea as assistant_live_gmail.py: only the MultiServerMCPClient config
changes. Notion's own open-source, token-based MCP server is deprecated -
the current recommended approach is Notion's *hosted* MCP server at
mcp.notion.com, reached locally via the `mcp-remote` stdio bridge and
authenticated with a one-time browser OAuth flow (see README.md).

Email is left mocked here since it's demoed separately in
assistant_live_gmail.py - see section_8_final_assistant for wiring both
real servers into one agent at once.
"""

import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

HERE = Path(__file__).resolve().parent

client = MultiServerMCPClient(
    {
        "email": {
            "transport": "stdio",
            "command": "python",
            "args": [str(HERE / "mcp_server_email.py")],
        },
        "notion": {
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "mcp-remote", "https://mcp.notion.com/mcp", "--silent"],
        },
    }
)

ARIA_SYSTEM_PROMPT = """
You are Aria, an AI executive assistant. You reach your tools over MCP -
use the Notion tools to search/read/create pages, and the email tools for
the inbox. Be concise.
"""


async def build_agent():
    tools = await client.get_tools()
    return create_agent("gpt-5-nano", tools=tools, system_prompt=ARIA_SYSTEM_PROMPT)


async def main():
    agent = await build_agent()
    question = HumanMessage(content="Search my Notion workspace for anything about the mermaid.ai.")
    response = await agent.ainvoke({"messages": [question]})
    print(response["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
