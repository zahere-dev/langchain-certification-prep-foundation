# Section 8 — Aria, Final Assembly

**Run it:** `python assistant.py` (interactive chat loop in the terminal, mock data)
**Live version:** `python assistant_live.py` (same architecture, real Gmail + real Notion via MCP)
**Or view the graph:** `langgraph dev` from this folder (uses `langgraph.json`)

## What we're building

Everything from sections 1-7 in one assistant:

- structured triage (1) → tools (2) → memory (3) → multi-agent research
  subagents (4) → an MCP-ready tool layer (5, see README) → human approval
  on irreversible actions (6) → VIP-aware dynamic behavior (7).

## Design decision worth explaining on camera

The two irreversible tools (`send_email`, `create_notion_page`) live
directly on the supervisor, not inside a subagent. `HumanInTheLoopMiddleware`
only guards tool calls made by the agent it's attached to — if a dangerous
tool lived inside a subagent invoked via a wrapper tool, the approval pause
wouldn't surface cleanly to the top-level conversation. Keeping research
(read-only, delegated) separate from action (owned by the supervisor, guarded)
is the actual architectural lesson of this section, not just "more code."

## `assistant.py` vs `assistant_live.py`

Same supervisor/research/HITL/dynamic-behavior architecture in both -
`assistant_live.py` only swaps where the tools come from (real MCP servers
instead of `shared/mock_data.py`) and reworks two things that don't
translate directly to a real inbox:

- `flag_vip` no longer looks up a hardcoded email - it lets Aria set the
  flag based on her own read of the sender/content, which is closer to how
  you'd actually want this to work in production.
- The research subagents' system prompts explicitly state the inbox/Notion
  access is the user's own, pre-authorized data. Without this, gpt-5-nano
  tends to refuse to call `search_emails`/`notion-search`, reading "the
  user's inbox" as a third party's private data - worth calling out on
  camera as a real prompting gotcha, not a fabricated one.

Because MCP tools load asynchronously, `assistant_live.py` builds the agent
inside `build_agent()` at runtime instead of exposing a module-level
`agent` like every other section - mention this as the one structural
difference `langgraph dev` can't easily visualize live for this file.

## Talking points for the video

- Do a live end-to-end run in the terminal (`assistant.py` for
  reproducibility, or `assistant_live.py` if you want the real-data payoff):
  ask Aria to triage the inbox, flag a VIP, draft a reply, and watch it
  pause for approval before sending. This is the "full circle" moment that
  pays off the whole video.
- Show `langgraph dev` and the Studio graph view for 30-60 seconds — seeing
  the supervisor/subagent/middleware structure as a diagram makes the whole
  architecture click for visual learners.
- Close by pointing at `notebooks/module-3/agent-chat-ui` as the natural
  next step (a real chat UI instead of a terminal loop) without building it
  live — good "if you want to go further" CTA before the outro.
- Reiterate the "going further" list from the top-level README: real MCP
  servers, a persistent checkpointer, a real chat UI. Gives viewers a clear
  next project instead of ending on "and that's it."
