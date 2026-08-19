"""
Section 3 - Memory
===================
Concepts: checkpointers, thread_id, short-term (per-conversation) memory.

Right now Aria forgets everything the second the process restarts. We add a
checkpointer so she remembers the conversation within a thread (e.g. "remember
I like short replies") - the same short-term memory pattern from module 1,
applied to our assistant.

Long-term memory (facts that persist *across* threads, like the user's writing
style) is handled differently - Aria reads that from Notion via search_notion,
which is effectively her long-term memory store. This split (checkpointer for
short-term, external store for long-term) is the pattern real agents use.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from shared.mock_data import INBOX, find_notion_page

load_dotenv()


@tool
def check_inbox() -> list[dict]:
    """Return the list of unread emails, most recent first."""
    return INBOX


@tool
def search_notion(query: str) -> list[dict]:
    """Search the user's Notion workspace for pages matching the query."""
    return find_notion_page(query)


ARIA_SYSTEM_PROMPT = """
You are Aria, an AI executive assistant for the user, a busy founder.

You can check the user's inbox and search their Notion workspace. Use these tools
before answering - never guess at the contents of the inbox or notes.
Be concise: bullet points over paragraphs.
Remember details the user tells you during the conversation and use them later.
"""

agent = create_agent(
    "gpt-5-nano",
    tools=[check_inbox, search_notion],
    system_prompt=ARIA_SYSTEM_PROMPT,
    checkpointer=InMemorySaver(),
)


if __name__ == "__main__":
    config = {"configurable": {"thread_id": "test-thread-1"}}

    question = HumanMessage(
           content="Get the latest meeting notes from Notion and summarize any action items for me. Summary Rule: Summary should be not more than 5 words"
       )
    response = agent.invoke({"messages": [question]}, config=config)
    print(response["messages"][-1].content)
    print()
    print(15 * "-")   
    print()
    question = HumanMessage(
            content="Get the latest meeting notes from Notion and summarize any action items for me. Keep the same summary rule as before."
        )
    response = agent.invoke({"messages": [question]}, config=config)
    print(response["messages"][-1].content)
