"""
Section 2 - Tools
==================
Concepts: @tool, giving an agent capabilities, tool calling loops.

Aria can now actually look things up instead of relying on whatever text you
paste into the chat. We give her two tools backed by the mock inbox/Notion
data in shared/mock_data.py: one to check the inbox, one to search Notion.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import HumanMessage

from shared.mock_data import INBOX, find_notion_page

load_dotenv()


@tool
def check_inbox() -> list[dict]:
    """Return the list of unread emails, most recent first."""
    return INBOX


@tool(description="Search the user's Notion workspace for pages matching the query.")
def search_notion(query: str) -> list[dict]:
    """Search the user's Notion workspace for pages matching the query."""
    return find_notion_page(query)


ARIA_SYSTEM_PROMPT = """
You are Aria, an AI executive assistant for the user, a busy founder.

You can check the user's inbox and search their Notion workspace. Use these tools
before answering - never guess at the contents of the inbox or notes.
Be concise: bullet points over paragraphs.
"""

agent = create_agent(
    "gpt-5-nano",
    tools=[check_inbox, search_notion],
    system_prompt=ARIA_SYSTEM_PROMPT,
)


if __name__ == "__main__":
    question = HumanMessage(
        content="Get the latest meeting notes from Notion and summarize any action items for me. Return only the summary. Summary Rule: Summary should be not more than 5 words"
    )
    response = agent.invoke({"messages": [question]})
    print(response["messages"][-1].content)

    #************************Uncomment the following lines to test the memory functionality in section 3************************
    # print()
    # print(15 * "-")   
    # print()
    # question = HumanMessage(
    #         content="Get the latest meeting notes from Notion and summarize any action items for me. Return only the summary. Keep the same summary rule as before."
    #     )
    # response = agent.invoke({"messages": [question]})
    # print(response["messages"][-1].content)
