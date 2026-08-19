"""
Section 5 - MCP (Model Context Protocol)
==========================================
Concepts: MultiServerMCPClient, stdio transport, connecting an agent to
tools that live in a completely separate process.

Instead of importing @tool functions directly, Aria now talks to two small
MCP servers (mcp_server_email.py, mcp_server_notion.py) over stdio. This is
the exact shape you'd use to plug in *real* Gmail/Notion MCP servers -
only the "args" below would need to change. See README.md for that swap.
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
        
        "gmail": {
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@gongrzhe/server-gmail-autoauth-mcp"],
        },
        "notion": {
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "mcp-remote", "https://mcp.notion.com/mcp", "--silent"],
        },
    }
)

ARIA_SYSTEM_PROMPT = """
You are Aria, an AI executive assistant for the user, a busy founder.
You reach your tools over MCP - use check_inbox, send_email, search_notion,
and create_notion_page as needed. Be concise.
"""


async def build_agent():
    tools = await client.get_tools()
    print(f"Tools: {tools}")
    return create_agent("gpt-5-nano", tools=tools, system_prompt=ARIA_SYSTEM_PROMPT)


async def main():
    agent = await build_agent()
    question = HumanMessage(content="What's the lastest unread message in my inbox, and do I have Notion notes about mermaid.ai?")
    response = await agent.ainvoke({"messages": [question]})
    print(response["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
