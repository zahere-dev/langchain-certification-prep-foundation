"""
Section 4 - Multi-Agent System
================================
Concepts: subagents, the supervisor pattern, delegation via tools, memory.

One agent juggling email, Notion, and a calendar starts to get confused about
which tools to use when. We split Aria into three specialists and one
supervisor that delegates to them - the same pattern from the course's
multi-agent notebook, applied to our project:

    Aria (supervisor)
      -> email_agent   (reads/drafts email)
      -> notion_agent  (searches/writes Notion)
      -> calendar_agent (checks the user's schedule)

Memory (section 3) is added at the supervisor level: a checkpointer keyed by
thread_id lets Aria remember what the user told her earlier in the
conversation. The specialists stay stateless - they only see the single
question they're asked and don't need their own memory.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from shared.mock_data import INBOX, NOTION_PAGES, CALENDAR, find_notion_page

load_dotenv()


# ---- Email specialist -------------------------------------------------

@tool
def check_inbox() -> list[dict]:
    """Return the list of unread emails, most recent first."""
    return INBOX


email_agent = create_agent(
    "gpt-5-nano",
    tools=[check_inbox],
    system_prompt="You are an email specialist. Read the inbox and answer questions about it concisely.",
)


@tool
def call_email_agent(question: str) -> str:
    """Ask the email specialist a question about the user's inbox."""
    response = email_agent.invoke({"messages": [HumanMessage(content=question)]})
    return response["messages"][-1].content


# ---- Notion specialist -------------------------------------------------

@tool
def search_notion(query: str) -> list[dict]:
    """Search the user's Notion workspace for pages matching the query."""
    return find_notion_page(query)


notion_agent = create_agent(
    "gpt-5-nano",
    tools=[search_notion],
    system_prompt="You are a Notion specialist. Search the user's workspace and answer questions about their notes.",
)


@tool
def call_notion_agent(question: str) -> str:
    """Ask the Notion specialist a question about the user's notes/pages."""
    response = notion_agent.invoke({"messages": [HumanMessage(content=question)]})
    return response["messages"][-1].content


# ---- Calendar specialist ------------------------------------------------

@tool
def check_calendar() -> list[dict]:
    """Return the user's upcoming calendar events."""
    return CALENDAR


calendar_agent = create_agent(
    "gpt-5-nano",
    tools=[check_calendar],
    system_prompt="You are a calendar specialist. Answer questions about the user's schedule concisely.",
)


@tool
def call_calendar_agent(question: str) -> str:
    """Ask the calendar specialist a question about the user's schedule."""
    response = calendar_agent.invoke({"messages": [HumanMessage(content=question)]})
    return response["messages"][-1].content


# ---- Supervisor -----------------------------------------------------------

ARIA_SUPERVISOR_PROMPT = """
You are Aria, an AI executive assistant for the user, a busy founder.

You don't do the work yourself - you delegate to specialists:
- call_email_agent for anything about the user's inbox
- call_notion_agent for anything about the user's notes/pages
- call_calendar_agent for anything about the user's schedule

Combine their answers into one short, direct response for the user.
Remember details the user tells you during the conversation and use them later.
"""

agent = create_agent(
    "gpt-5-nano",
    tools=[call_email_agent, call_notion_agent, call_calendar_agent],
    system_prompt=ARIA_SUPERVISOR_PROMPT,
    checkpointer=InMemorySaver(),
)


if __name__ == "__main__":
    config = {"configurable": {"thread_id": "multi-agent-thread-1"}}

    question = HumanMessage(
        content="Do I have anything urgent in my inbox, and am I free to prep for the board deck tomorrow morning? "
        "Keep replies under 3 sentences from now on."
    )
    response = agent.invoke({"messages": [question]}, config=config)
    print(response["messages"][-1].content)
    print()
    print(15 * "-")
    print()
    question = HumanMessage(
        content="Now check my Notion for board deck notes, and keep the same reply-length rule as before."
    )
    response = agent.invoke({"messages": [question]}, config=config)
    print(response["messages"][-1].content)
