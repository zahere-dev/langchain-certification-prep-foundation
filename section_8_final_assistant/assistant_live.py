"""
Section 8 (bonus) - Aria, wired to real Gmail + real Notion
================================================================
Same architecture as assistant.py (supervisor + read-only research
subagents + HITL-guarded actions + VIP-aware dynamic behavior), but the
tools now come from real MCP servers instead of shared/mock_data.py:

  - Gmail: @gongrzhe/server-gmail-autoauth-mcp (see section_5_mcp/README.md
    "Connecting a real Gmail account" for the one-time auth)
  - Notion: mcp-remote bridge to Notion's hosted MCP server (see
    section_5_mcp/README.md "Connecting a real Notion workspace")

Calendar stays mocked - swap check_calendar for a Google Calendar MCP
server the same way if you want to take this further.

Because MCP tools are fetched asynchronously, this file builds the agent
inside build_agent() instead of at import time like assistant.py does.
"""

import asyncio
import sys
from pathlib import Path
from typing import Callable

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from langchain.agents import create_agent, AgentState
from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
from langchain.messages import HumanMessage, ToolMessage
from langchain.agents.middleware import (
    dynamic_prompt,
    wrap_model_call,
    ModelRequest,
    ModelResponse,
    HumanInTheLoopMiddleware,
)
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from langchain_mcp_adapters.client import MultiServerMCPClient

from shared.mock_data import CALENDAR

load_dotenv()


class AriaState(AgentState):
    is_vip: bool


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

# The real servers expose many tools (labels, filters, page moves, etc.) -
# we only pull in the handful Aria actually needs, split into read-only
# research tools vs. the two irreversible action tools.
EMAIL_RESEARCH_TOOLS = {"search_emails", "read_email"}
EMAIL_ACTION_TOOLS = {"send_email"}
NOTION_RESEARCH_TOOLS = {"notion-search", "notion-fetch"}
NOTION_ACTION_TOOLS = {"notion-create-pages"}


@tool
def check_calendar() -> list[dict]:
    """Return the user's upcoming calendar events."""
    return CALENDAR


@tool
def flag_vip(is_vip: bool, runtime: ToolRuntime) -> Command:
    """Flag whether this thread involves a VIP contact, based on your own
    judgement of the sender/content you've already read (e.g. a major
    client or an exec, vs. a newsletter or routine request)."""
    return Command(
        update={
            "is_vip": is_vip,
            "messages": [ToolMessage(f"VIP status set to {is_vip}", tool_call_id=runtime.tool_call_id)],
        }
    )


VIP_PROMPT = """
You are Aria, the user's AI executive assistant.
The user's Gmail inbox is fordemoagentexpert@gmail.com - you have direct,
pre-authorized, already-connected read access to it via call_email_researcher.
Never claim you can't read the inbox; delegate to call_email_researcher instead.
This thread involves a VIP contact - be thorough and careful with your drafts.
Delegate research to call_email_researcher / call_notion_researcher before drafting.
When asked to reply or create a note, call the send/create tool directly -
a human review step happens automatically before anything irreversible
actually goes out, so you don't need to ask for permission yourself.
"""

STANDARD_PROMPT = """
You are Aria, the user's AI executive assistant.
The user's Gmail inbox is fordemoagentexpert@gmail.com - you have direct,
pre-authorized, already-connected read access to it via call_email_researcher.
Never claim you can't read the inbox; delegate to call_email_researcher instead.
Be efficient. Delegate research to call_email_researcher / call_notion_researcher before drafting.
When asked to reply or create a note, call the send/create tool directly -
a human review step happens automatically before anything irreversible
actually goes out, so you don't need to ask for permission yourself.
"""


@dynamic_prompt
def vip_aware_prompt(request: ModelRequest) -> str:
    return VIP_PROMPT if request.state.get("is_vip") else STANDARD_PROMPT


vip_model = init_chat_model("gpt-5")
standard_model = init_chat_model("gpt-5-nano")


@wrap_model_call
async def vip_aware_model(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    model = vip_model if request.state.get("is_vip") else standard_model
    request = request.override(model=model)
    return await handler(request)


MAX_SEARCH_EMAILS_RESULTS = 3


def _cap_search_emails(search_emails_tool):
    """Force search_emails to never return more than MAX_SEARCH_EMAILS_RESULTS,
    regardless of what maxResults the model asks for."""
    original_coroutine = search_emails_tool.coroutine

    async def capped_coroutine(*args, **kwargs):
        requested = kwargs.get("maxResults", MAX_SEARCH_EMAILS_RESULTS)
        kwargs["maxResults"] = min(requested, MAX_SEARCH_EMAILS_RESULTS)
        return await original_coroutine(*args, **kwargs)

    search_emails_tool.coroutine = capped_coroutine
    return search_emails_tool


async def build_agent():
    all_tools = await client.get_tools()
    by_name = {t.name: t for t in all_tools}
    if "search_emails" in by_name:
        by_name["search_emails"] = _cap_search_emails(by_name["search_emails"])

    email_researcher = create_agent(
        "gpt-5-nano",
        tools=[by_name[n] for n in EMAIL_RESEARCH_TOOLS if n in by_name],
        system_prompt=(
            "You have direct, pre-authorized, already-connected access to the "
            "user's own Gmail inbox (fordemoagentexpert@gmail.com) through the "
            "search_emails and read_email tools connected to this session - this "
            "is not a third party's private data, it belongs to the person asking. "
            "This access is already set up and working; never claim you can't read "
            "the inbox or that you lack access. Always call these tools to answer "
            "instead of refusing or telling the user to search it themselves. When "
            "searching, only the latest 3 results are returned - focus on those. "
            "Summarize findings. Never send anything."
        ),
    )

    notion_researcher = create_agent(
        "gpt-5-nano",
        tools=[by_name[n] for n in NOTION_RESEARCH_TOOLS if n in by_name],
        system_prompt=(
            "You have direct, pre-authorized access to the user's own Notion "
            "workspace through the notion-search and notion-fetch tools connected "
            "to this session. Always call these tools to answer instead of refusing "
            "or telling the user to search it themselves. Summarize findings. Never "
            "create pages."
        ),
    )

    @tool
    async def call_email_researcher(question: str) -> str:
        """Ask the email researcher a question about the user's real inbox (fordemoagentexpert@gmail.com)."""
        response = await email_researcher.ainvoke({"messages": [HumanMessage(content=question)]})
        return response["messages"][-1].content

    @tool
    async def call_notion_researcher(question: str) -> str:
        """Ask the Notion researcher a question about the user's real Notion workspace."""
        response = await notion_researcher.ainvoke({"messages": [HumanMessage(content=question)]})
        return response["messages"][-1].content

    action_tools = [by_name[n] for n in (EMAIL_ACTION_TOOLS | NOTION_ACTION_TOOLS) if n in by_name]
    interrupt_on = {t.name: True for t in action_tools}

    return create_agent(
        "gpt-5-nano",
        tools=[call_email_researcher, call_notion_researcher, check_calendar, flag_vip, *action_tools],
        state_schema=AriaState,
        checkpointer=InMemorySaver(),
        middleware=[
            vip_aware_prompt,
            vip_aware_model,
            HumanInTheLoopMiddleware(interrupt_on=interrupt_on),
        ],
    )


async def chat():
    """A minimal terminal chat loop for demoing the real-tool Aria end to end."""
    agent = await build_agent()
    config = {"configurable": {"thread_id": "user-live-real"}}
    print("Aria (live Gmail + Notion) is ready. Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break

        result = await agent.ainvoke({"messages": [HumanMessage(content=user_input)]}, config)

        if "__interrupt__" in result:
            interrupt = result["__interrupt__"][0]
            print(f"Aria (needs approval): {interrupt.value}")
            decision = input("Approve? [y/N]: ").strip().lower()
            resume = {"decisions": [{"type": "approve" if decision == "y" else "reject"}]}
            result = await agent.ainvoke(Command(resume=resume), config)

        print(f"Aria: {result['messages'][-1].content}\n")


if __name__ == "__main__":
    asyncio.run(chat())
