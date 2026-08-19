"""
Section 7 - Dynamic Behavior
==============================
Concepts: dynamic_prompt, wrap_model_call (dynamic tools/model), state schema.

Not every email deserves the same treatment. A VIP sender (the user's biggest
client, their CEO) should get Aria's full attention and best model; a
newsletter shouldn't. We use middleware to change the system prompt, the
tool set, and even the model itself based on state - the same pattern as
the course's authenticated email_agent, applied to VIP routing.
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
from langchain.agents.middleware import dynamic_prompt, wrap_model_call, ModelRequest, ModelResponse
from langgraph.types import Command

from shared.mock_data import INBOX, find_email

load_dotenv()


class TriageState(AgentState):
    is_vip: bool


@tool
def check_inbox() -> list[dict]:
    """Return the list of unread emails, most recent first."""
    return INBOX


@tool
def load_email(email_id: str, runtime: ToolRuntime) -> Command:
    """Load a specific email by id and flag whether the sender is a VIP."""
    email = find_email(email_id)
    is_vip = bool(email and email.get("vip"))
    return Command(
        update={
            "is_vip": is_vip,
            "messages": [ToolMessage(str(email), tool_call_id=runtime.tool_call_id)],
        }
    )


@tool
def draft_thoughtful_reply(body: str) -> str:
    """Draft a carefully considered reply. Reserved for VIP senders."""
    return f"[VIP draft]\n{body}"


@tool
def draft_quick_reply(body: str) -> str:
    """Draft a short, low-effort reply. Used for non-VIP senders."""
    return f"[Quick draft]\n{body}"


VIP_PROMPT = """
You are Aria, the user's AI executive assistant. This sender is a VIP.
Take your time, be thorough, and use draft_thoughtful_reply.
"""

STANDARD_PROMPT = """
You are Aria, the user's AI executive assistant. This is a routine email.
Be efficient and use draft_quick_reply.
"""


@dynamic_prompt
def vip_aware_prompt(request: ModelRequest) -> str:
    return VIP_PROMPT if request.state.get("is_vip") else STANDARD_PROMPT


vip_model = init_chat_model("gpt-5")
standard_model = init_chat_model("gpt-5-nano")


@wrap_model_call
def vip_aware_routing(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Give VIP emails the stronger model and the thoughtful-reply tool."""
    is_vip = request.state.get("is_vip")

    model = vip_model if is_vip else standard_model
    tools = [check_inbox, load_email, draft_thoughtful_reply if is_vip else draft_quick_reply]

    request = request.override(model=model, tools=tools)
    return handler(request)


agent = create_agent(
    "gpt-5-nano",
    tools=[check_inbox, load_email, draft_thoughtful_reply, draft_quick_reply],
    state_schema=TriageState,
    middleware=[vip_aware_prompt, vip_aware_routing],
)


if __name__ == "__main__":
    question = HumanMessage(content="Load email e3 and draft a reply.")
    response = agent.invoke({"messages": [question]})
    print(response["messages"][-1].content)
