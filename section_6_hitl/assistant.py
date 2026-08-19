"""
Section 6 - Human-in-the-Loop
================================
Concepts: HumanInTheLoopMiddleware, interrupt_on, resuming a graph.

This is the section that makes Aria trustworthy. She can draft an email or a
Notion page on her own, but she must NOT send/create anything without the
user's explicit approval. We use HumanInTheLoopMiddleware to pause execution
right before the risky tools (send_email, create_notion_page) and hand control
back to the user.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import HumanMessage
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from shared.mock_data import INBOX, NOTION_PAGES, find_notion_page

load_dotenv()


@tool
def check_inbox() -> list[dict]:
    """Return the list of unread emails, most recent first."""
    return INBOX


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to the given recipient with the given subject and body."""
    return f"Email sent to {to} | subject: {subject}"


@tool
def search_notion(query: str) -> list[dict]:
    """Search the user's Notion workspace for pages matching the query."""
    return find_notion_page(query)


@tool
def create_notion_page(title: str, content: str) -> str:
    """Create a new Notion page with the given title and content."""
    NOTION_PAGES.append({"id": f"p{len(NOTION_PAGES) + 1}", "title": title, "content": content})
    return f"Created Notion page '{title}'"


ARIA_SYSTEM_PROMPT = """
You are Aria, an AI executive assistant for the user, a busy founder.
When asked to reply to an email or create a note, go ahead and call the
appropriate tool directly - a human review step happens automatically
before anything irreversible actually goes out, so you don't need to ask
for permission yourself.
"""

agent = create_agent(
    "gpt-5-nano",
    tools=[check_inbox, send_email, search_notion, create_notion_page],
    system_prompt=ARIA_SYSTEM_PROMPT,
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                # Read-only tools run freely...
                "check_inbox": False,
                "search_notion": False,
                # ...but anything that touches the outside world needs sign-off.
                "send_email": True,
                "create_notion_page": True,
            }
        ),
    ],
)


if __name__ == "__main__":
    config = {"configurable": {"thread_id": "user-approval-demo"}}

    result = agent.invoke(
        {"messages": [HumanMessage(content="Reply to Sam accepting the coffee for Tuesday.")]},
        config,
    )

    if "__interrupt__" in result:
        interrupt = result["__interrupt__"][0]
        print("Aria wants approval for:")
        print(interrupt.value)

        # The user approves the drafted tool call and Aria finishes the job.
        final = agent.invoke(Command(resume={"decisions": [{"type": "reject"}]}), config)
        print(final["messages"][-1].content)
    else:
        print(result["messages"][-1].content)
