# LangChain Certification Prep

Hands-on prep for the LangChain/LangGraph certification. Instead of isolated
exercises, the material is taught by building one project end-to-end:
**Aria**, an AI executive assistant that triages your inbox, researches your
Notion workspace, checks your calendar, drafts replies/notes, and asks for
your approval before anything irreversible actually happens.

Each `section_N_*` folder is a **standalone, runnable script** that builds on
the previous section's concepts, so a concept from the certification syllabus
always has a concrete reason to exist instead of a toy example: memory
matters because Aria needs to remember your writing style; multi-agent
matters because one agent juggling email+Notion+calendar gets confused;
human-in-the-loop matters because nobody wants an AI that sends emails on
its own.

The scripts run on mock, hardcoded data (`shared/mock_data.py`) — no real
Gmail/Notion OAuth needed to work through the material. `section_5_mcp` and
`section_8_final_assistant` also ship a tested `assistant_live*.py` variant
wired to real Gmail (via `@gongrzhe/server-gmail-autoauth-mcp`) and real
Notion (via Notion's hosted MCP server), for when you want to see it running
against real data.

## Structure

| # | Folder | Concept(s) introduced | Certification topic |
|---|--------|------------------------|----------------|
| 1 | `section_1_foundations` | model init, system prompts, structured output | Module 1 |
| 2 | `section_2_tools` | `@tool`, giving the agent capabilities | Module 1 |
| 3 | `section_3_memory` | checkpointers, `thread_id`, short vs. long-term memory | Module 1 |
| 4 | `section_4_multi_agent` | supervisor pattern, subagents as tools | Module 2 |
| 5 | `section_5_mcp` | MCP servers, `MultiServerMCPClient`, going from mock to real tools | Module 2 |
| 6 | `section_6_hitl` | `HumanInTheLoopMiddleware`, interrupt/resume | Module 3 |
| 7 | `section_7_dynamic_behavior` | `dynamic_prompt`, `wrap_model_call`, state-driven routing | Module 3 |
| 8 | `section_8_final_assistant` | everything combined into one deployable agent + CLI + LangGraph Studio config | Capstone |

Each folder has its own `README.md` explaining the concept being taught, why
it's needed for Aria specifically, and the exact command to run the demo.

## Setup

```bash
cp .env.example .env
pip install -r requirements.txt
python section_1_foundations/assistant.py
```

Every section only needs `OPENAI_API_KEY`. Nothing else is required to work
through the material — that's intentional, so you can clone and run the
whole thing with just an OpenAI key.

## Connecting real Gmail / Notion (for the `assistant_live*.py` variants)

One-time setup — see `section_5_mcp/README.md` for the full steps, and the
official docs below for the authoritative configuration reference:

- **Gmail**: Google Cloud OAuth client (Web application type, redirect URI
  `http://localhost:3000/oauth2callback`) + `npx @gongrzhe/server-gmail-autoauth-mcp auth`
  once to cache a token at `~/.gmail-mcp/`. Official docs:
  https://developers.google.com/workspace/gmail/api/guides/configure-mcp-server
- **Notion**: `npx -y mcp-remote https://mcp.notion.com/mcp` once to do a
  browser OAuth flow and cache a token at `~/.mcp-auth/`. (Notion's older
  token-based server, `@notionhq/notion-mcp-server`, is deprecated — don't
  use it, it 400s on tool calls.) Official docs:
  https://developers.notion.com/guides/mcp/get-started-with-mcp#json-configuration-format

Both tokens live outside the repo and are never committed. Once set up:
- `section_5_mcp/assistant_live_gmail.py` / `assistant_live_notion.py` —
  single-tool-source demos.
- `section_8_final_assistant/assistant_live.py` — the full supervisor +
  HITL + dynamic-behavior assistant, running against your real inbox and
  real Notion workspace at once.

## Going further

- Point `section_8_final_assistant` at LangGraph Studio:
  `langgraph dev` from inside `section_8_final_assistant/` (uses the
  included `langgraph.json`) to see the visual graph + trace.
- Front `section_8_final_assistant/assistant.py` (or `assistant_live.py`)
  with a chat UI such as `agent-chat-ui` for a fuller demo.
- Real Google Calendar MCP server, to replace the still-mocked
  `check_calendar` tool in section 4/7/8.
