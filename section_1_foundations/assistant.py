"""
Section 1 - Foundations
========================
Concepts: create_agent, system prompts, structured output.

This is the very first version of Aria. She can't read your inbox yet - all
she can do is take a raw email and tell you, in a structured way, whether it
needs your attention. Every later section builds directly on top of this one.
"""

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.messages import HumanMessage

load_dotenv()


class EmailTriage(BaseModel):
    """Structured verdict Aria produces for a single email."""

    priority: str = Field(description="One of: urgent, reply_needed, fyi, ignore")
    reason: str = Field(description="One sentence explanation for the priority")
    suggested_action: str = Field(description="What the user should do next")


ARIA_SYSTEM_PROMPT = """
You are Aria, an AI executive assistant for the user, a busy founder.

Your job right now is simple: read one email at a time and triage it.
Be decisive. Busy people don't want a wall of text - give short, direct answers.
"""

triage_agent = create_agent(
    "gpt-5-nano",
    system_prompt=ARIA_SYSTEM_PROMPT,
    response_format=EmailTriage,
)


def triage_email(raw_email: str) -> EmailTriage:
    response = triage_agent.invoke({"messages": [HumanMessage(content=raw_email)]})
    return response["structured_response"]


if __name__ == "__main__":
    email = """
    From: ceo@mycompany.com
    Subject: URGENT: board deck needed tonight

    Can you pull together the Q3 numbers for the board deck? I need it by 6pm today.
    """

    result = triage_email(email)
    print(result)
