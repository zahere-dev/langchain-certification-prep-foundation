"""
Fake data for Aria, the AI Executive Assistant project.

Everything here is hardcoded on purpose: viewers can run every section of this
project without needing a Gmail or Notion account. section_5_mcp swaps these
in-memory stores for real MCP servers once the concepts are in place.
"""

INBOX = [
    {
        "id": "e1",
        "from": "jane@bigclient.com",
        "subject": "Contract renewal - need your sign-off",
        "body": "Hi, our contract renews in 2 weeks. Can you confirm the new terms by Friday? - Jane",
        "vip": True,
    },
    {
        "id": "e2",
        "from": "noreply@newsletter.com",
        "subject": "10 productivity hacks you need to know",
        "body": "Click here to read our latest newsletter...",
        "vip": False,
    },
    {
        "id": "e3",
        "from": "sam@partner.io",
        "subject": "Quick coffee next week?",
        "body": "Hey, I'm in town Tuesday and Wednesday next week, want to grab a coffee? - Sam",
        "vip": False,
    },
    {
        "id": "e4",
        "from": "ceo@mycompany.com",
        "subject": "URGENT: board deck needed tonight",
        "body": "Can you pull together the Q3 numbers for the board deck? I need it by 6pm today.",
        "vip": True,
    },
]

NOTION_PAGES = [
    {
        "id": "p1",
        "title": "Q3 Board Deck",
        "content": "Draft outline: revenue growth, churn, hiring plan. Status: in progress.",
    },
    {
        "id": "p2",
        "title": "Meeting Notes - Partner Sync",
        "content": "Discussed Q4 roadmap with Sam from partner.io. Follow up on integration timeline.",
    },
    {
        "id": "p3",
        "title": "Writing Style Notes",
        "content": "Keep replies short, friendly but professional. Sign off with 'Best, the user'. Avoid exclamation points.",
    },
]

CALENDAR = [
    {"date": "2026-08-18", "time": "10:00", "title": "Board prep sync"},
    {"date": "2026-08-19", "time": "14:00", "title": "Partner call - Sam"},
]


def find_email(email_id: str):
    return next((e for e in INBOX if e["id"] == email_id), None)


def find_notion_page(query: str):
    query = query.lower()
    return [p for p in NOTION_PAGES if query in p["title"].lower() or query in p["content"].lower()]
