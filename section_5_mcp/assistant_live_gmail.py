"""
Section 5 (bonus) - Swapping the mock Gmail server for the real thing
========================================================================
Everything else about `assistant.py` stays identical - only this one entry
in the MultiServerMCPClient config changes. This is the "going further"
step from the README: once you've run

    npx @gongrzhe/server-gmail-autoauth-mcp auth

once (see README.md) and have a cached token in ~/.gmail-mcp/, Aria can
read and send from your real inbox instead of the mock one.

Notion is left mocked here since it needs its own token - swap it the same
way once you have a NOTION_TOKEN (see the commented block in assistant.py).
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
            "command": "python",
            "args": [str(HERE / "mcp_server_notion.py")],
        },
    }
)

ARIA_SYSTEM_PROMPT = """
You are Aria, an AI executive assistant. You reach your tools over MCP -
use the Gmail tools to check the inbox / search / send, and search_notion
for notes. Be concise. Never send an email without repeating the draft
back to the user first.
"""


async def build_agent():
    tools = await client.get_tools()
    return create_agent("gpt-5-nano", tools=tools, system_prompt=ARIA_SYSTEM_PROMPT)


async def main():
    agent = await build_agent()
    question = HumanMessage(content="What's last unread email in my inbox right now?")
    response = await agent.ainvoke({"messages": [question]})
    print(response["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
