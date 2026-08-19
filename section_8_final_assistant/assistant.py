"""
Section 8 - Aria, Final Assembly
===================================
Every concept from sections 1-7, combined into one assistant:

  - structured triage              (section 1)
  - tools                          (section 2)
  - memory via checkpointer        (section 3)
  - multi-agent delegation         (section 4)
  - MCP-ready tool layer           (section 5 - see README to swap in)
  - human-in-the-loop approval     (section 6)
  - VIP-aware dynamic behavior     (section 7)

Design note: the two *irreversible* tools (send_email, create_notion_page)
live directly on the supervisor so HumanInTheLoopMiddleware can guard them
in one place. The subagents below are read-only researchers - they gather
information but never take action themselves.
"""

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

from shared.mock_data import INBOX, NOTION_PAGES, CALENDAR, find_email, find_notion_page

load_dotenv()


class AriaState(AgentState):
    is_vip: bool


# ---- Read-only research subagents -----------------------------------------

@tool
def check_inbox() -> list[dict]:
    """Return the list of unread emails, most recent first."""
    return INBOX


email_researcher = create_agent(
    "gpt-5-nano",
    tools=[check_inbox],
    system_prompt="You research the user's inbox and summarize findings. You never send anything.",
)


@tool
def call_email_researcher(question: str) -> str:
    """Ask the email researcher a question about the user's inbox."""
    response = email_researcher.invoke({"messages": [HumanMessage(content=question)]})
    return response["messages"][-1].content


@tool
def search_notion(query: str) -> list[dict]:
    """Search the user's Notion workspace for pages matching the query."""
    return find_notion_page(query)


notion_researcher = create_agent(
    "gpt-5-nano",
    tools=[search_notion],
    system_prompt="You research the user's Notion workspace and summarize findings. You never create pages.",
)


@tool
def call_notion_researcher(question: str) -> str:
    """Ask the Notion researcher a question about the user's notes/pages."""
    response = notion_researcher.invoke({"messages": [HumanMessage(content=question)]})
    return response["messages"][-1].content


# ---- Supervisor-owned tools --------------------------------------------

@tool
def check_calendar() -> list[dict]:
    """Return the user's upcoming calendar events."""
    return CALENDAR


@tool
def flag_vip(email_id: str, runtime: ToolRuntime) -> Command:
    """Check whether the sender of an email is a VIP and update state accordingly."""
    email = find_email(email_id)
    is_vip = bool(email and email.get("vip"))
    return Command(
        update={
            "is_vip": is_vip,
            "messages": [ToolMessage(f"VIP status: {is_vip}", tool_call_id=runtime.tool_call_id)],
        }
    )


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to the given recipient with the given subject and body."""
    return f"Email sent to {to} | subject: {subject}"


@tool
def create_notion_page(title: str, content: str) -> str:
    """Create a new Notion page with the given title and content."""
    NOTION_PAGES.append({"id": f"p{len(NOTION_PAGES) + 1}", "title": title, "content": content})
    return f"Created Notion page '{title}'"


# ---- Dynamic, VIP-aware behavior -----------------------------------------

VIP_PROMPT = """
You are Aria, the user's AI executive assistant.
This thread involves a VIP contact - be thorough and careful with your drafts.
Delegate research to call_email_researcher / call_notion_researcher before drafting.
When asked to reply or create a note, call send_email / create_notion_page
directly - a human review step happens automatically before anything
irreversible actually goes out, so you don't need to ask for permission yourself.
"""

STANDARD_PROMPT = """
You are Aria, the user's AI executive assistant.
Be efficient. Delegate research to call_email_researcher / call_notion_researcher before drafting.
When asked to reply or create a note, call send_email / create_notion_page
directly - a human review step happens automatically before anything
irreversible actually goes out, so you don't need to ask for permission yourself.
"""


@dynamic_prompt
def vip_aware_prompt(request: ModelRequest) -> str:
    return VIP_PROMPT if request.state.get("is_vip") else STANDARD_PROMPT


vip_model = init_chat_model("gpt-5")
standard_model = init_chat_model("gpt-5-nano")


@wrap_model_call
def vip_aware_model(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """VIP threads get the stronger model; everything else uses the cheap one."""
    model = vip_model if request.state.get("is_vip") else standard_model
    request = request.override(model=model)
    return handler(request)


# ---- Aria ------------------------------------------------------------------

agent = create_agent(
    "gpt-5-nano",
    tools=[
        call_email_researcher,
        call_notion_researcher,
        check_calendar,
        flag_vip,
        send_email,
        create_notion_page,
    ],
    state_schema=AriaState,
    checkpointer=InMemorySaver(),
    middleware=[
        vip_aware_prompt,
        vip_aware_model,
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email": True,
                "create_notion_page": True,
            }
        ),
    ],
)


def chat():
    """A minimal terminal chat loop for demoing Aria end to end."""
    config = {"configurable": {"thread_id": "user-live"}}
    print("Aria is ready. Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break

        result = agent.invoke({"messages": [HumanMessage(content=user_input)]}, config)

        if "__interrupt__" in result:
            interrupt = result["__interrupt__"][0]
            print(f"Aria (needs approval): {interrupt.value}")
            decision = input("Approve? [y/N]: ").strip().lower()
            resume = {"decisions": [{"type": "approve" if decision == "y" else "reject"}]}
            result = agent.invoke(Command(resume=resume), config)

        print(f"Aria: {result['messages'][-1].content}\n")


if __name__ == "__main__":
    chat()
