"""
agent.py

The FitFindr planning loop. Orchestrates the three tools in response to a
natural language user query, passing state between them via a session dict.

Complete tools.py and test each tool in isolation before implementing this file.

Usage (once implemented):
    from agent import run_agent
    from utils.data_loader import get_example_wardrobe

    result = run_agent(
        query="vintage graphic tee under $30, size M",
        wardrobe=get_example_wardrobe(),
    )
    print(result["fit_card"])
    print(result["error"])   # None on success
"""

import json

from tools import (
    search_listings,
    suggest_outfit,
    create_fit_card,
    _get_groq_client,
    _MODEL,
)


# ── session state ─────────────────────────────────────────────────────────────

def _new_session(query: str, wardrobe: dict) -> dict:
    """
    Initialize and return a fresh session dict for one user interaction.

    The session dict is the single source of truth for everything that happens
    during a run — it stores the original query, parsed parameters, tool results,
    and any error that caused early termination.

    You may add fields to this dict as needed for your implementation.
    """
    return {
        "query": query,              # original user query
        "parsed": {},                # extracted description / size / max_price
        "search_results": [],        # list of matching listing dicts
        "selected_item": None,       # top result, passed into suggest_outfit
        "wardrobe": wardrobe,        # user's wardrobe dict
        "outfit_suggestion": None,   # string returned by suggest_outfit
        "fit_card": None,            # string returned by create_fit_card
        "final_message": None,       # LLM's natural-language wrap-up to the user
        "error": None,               # set if the interaction ended early
    }


# ── tool schemas + system prompt (the LLM's view of the world) ──────────────────

# The LLM picks WHICH tool and supplies lightweight args. The real item/wardrobe/
# outfit objects are injected from `session` by _dispatch(), so suggest_outfit and
# create_fit_card take no arguments here.
_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_listings",
            "description": (
                "Search secondhand clothing listings for items matching a "
                "description, with an optional size and maximum price. Call this "
                "first. Returns the top matches, or nothing if none match."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Keywords for the item, e.g. 'vintage graphic tee'.",
                    },
                    "size": {
                        "type": "string",
                        "description": "Desired size, e.g. 'M'. Omit if unspecified.",
                    },
                    "max_price": {
                        "type": "number",
                        "description": "Maximum price in dollars. Omit if unspecified.",
                    },
                },
                "required": ["description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "suggest_outfit",
            "description": (
                "Style the item found by search_listings against the user's "
                "wardrobe. Call only after a successful search. Takes no arguments."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_fit_card",
            "description": (
                "Write a short, shareable social-media caption for the found item "
                "and suggested outfit. Call only after suggest_outfit. Takes no "
                "arguments."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

_SYSTEM_PROMPT = (
    "You are FitFindr, an assistant that helps people find secondhand clothing "
    "and figure out how to wear it.\n\n"
    "You have three tools: search_listings, suggest_outfit, and create_fit_card.\n"
    "The usual flow is: (1) search_listings using the user's description, size, "
    "and budget; (2) if it returns items, call suggest_outfit to style the top "
    "find; (3) then call create_fit_card for a shareable caption. Choose each "
    "tool based on what the previous result gave you — do not call tools blindly.\n\n"
    "IMPORTANT: If search_listings returns no results, STOP. Do not call the "
    "other tools. Tell the user nothing matched and suggest broadening the search "
    "(drop the size or price limit, or try more general keywords).\n\n"
    "The system manages the actual item and wardrobe data, so suggest_outfit and "
    "create_fit_card need no arguments — just call them. When you are finished, "
    "reply with a short, friendly summary of what you found for the user."
)

# Hard ceiling on planning turns so a misbehaving LLM can never spin forever.
_MAX_ITERS = 6


# ── tool dispatch (executes the real tool, mutates session, returns a summary) ──

def _dispatch(tool_call, session: dict) -> str:
    """Run the tool the LLM selected, update `session`, and return a compact
    text result to feed back into the conversation.

    Real objects (selected_item, wardrobe, outfit_suggestion) are injected from
    `session` — the LLM never has to re-serialize them. Out-of-order calls are
    caught by state guards that tell the LLM what to do first.
    """
    name = tool_call.function.name

    try:
        args = json.loads(tool_call.function.arguments or "{}")
    except (json.JSONDecodeError, TypeError):
        return "Error: tool arguments were not valid JSON. Call the tool again with valid arguments."

    if name == "search_listings":
        description = args.get("description", "")
        size = args.get("size")
        max_price = args.get("max_price")
        session["parsed"] = {
            "description": description,
            "size": size,
            "max_price": max_price,
        }
        results = search_listings(description, size, max_price)
        session["search_results"] = results

        if not results:
            # No-results branch: terminate early and ask the user to broaden.
            session["error"] = (
                "No listings matched that search. Try broadening it — drop the "
                "size or price limit, or use more general keywords."
            )
            return (
                "no_results: nothing matched this search. Tell the user no "
                "listings were found and suggest broadening the search. Do NOT "
                "call any other tools."
            )

        session["selected_item"] = results[0]
        lines = [
            f'- {r["id"]}: {r["title"]} — ${r["price"]:g} ({r["platform"]})'
            for r in results
        ]
        return (
            "Found these listings (the first is auto-selected as the top match):\n"
            + "\n".join(lines)
        )

    if name == "suggest_outfit":
        if not session.get("selected_item"):
            return "Error: no item selected yet. Call search_listings first."
        suggestion = suggest_outfit(session["selected_item"], session["wardrobe"])
        session["outfit_suggestion"] = suggestion
        return f"Outfit suggestion created:\n{suggestion}"

    if name == "create_fit_card":
        if not session.get("outfit_suggestion"):
            return "Error: no outfit suggestion yet. Call suggest_outfit first."
        card = create_fit_card(session["outfit_suggestion"], session["selected_item"])
        session["fit_card"] = card
        return f"Fit card created:\n{card}"

    return f"Error: unknown tool '{name}'."


def _assistant_message(msg) -> dict:
    """Serialize a Groq assistant message (with any tool calls) back into a
    plain dict so it can be appended to the running `messages` list."""
    out = {"role": "assistant", "content": msg.content or ""}
    if msg.tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in msg.tool_calls
        ]
    return out


# ── planning loop ─────────────────────────────────────────────────────────────

def run_agent(query: str, wardrobe: dict) -> dict:
    """
    Main agent entry point. Runs the FitFindr planning loop for a single
    user interaction and returns the completed session dict.

    Args:
        query:    Natural language user request
                  (e.g., "vintage graphic tee under $30, size M")
        wardrobe: User's wardrobe dict — use get_example_wardrobe() or
                  get_empty_wardrobe() from utils/data_loader.py

    Returns:
        The session dict after the interaction completes. Check session["error"]
        first — if it is not None, the interaction ended early and the other
        output fields (outfit_suggestion, fit_card) will be None.

    TODO — implement this function using the planning loop you designed in planning.md:

        Step 1: Initialize the session with _new_session().

        Step 2: Parse the user's query to extract a description, size, and
                max_price. You can use regex, string splitting, or ask the LLM
                to parse it — document your choice in planning.md.
                Store the result in session["parsed"].

        Step 3: Call search_listings() with the parsed parameters.
                Store results in session["search_results"].
                If no results: set session["error"] to a helpful message and
                return the session early. Do NOT proceed to suggest_outfit
                with empty input.

        Step 4: Select the item to use (e.g., the top result).
                Store it in session["selected_item"].

        Step 5: Call suggest_outfit() with the selected item and wardrobe.
                Store the result in session["outfit_suggestion"].

        Step 6: Call create_fit_card() with the outfit suggestion and selected item.
                Store the result in session["fit_card"].

        Step 7: Return the session.

    Before writing code, complete the Planning Loop and State Management sections
    of planning.md — your implementation should match what you described there.
    """
    session = _new_session(query, wardrobe)

    # The client may fail to initialize (e.g. missing GROQ_API_KEY) — surface
    # that as a clean error rather than crashing the agent.
    try:
        client = _get_groq_client()
    except ValueError as exc:
        session["error"] = str(exc)
        return session

    # `messages` is the LLM's conversational memory; `session` is the structured
    # source of truth. The LLM picks tools; _dispatch executes them and stores
    # results in `session`.
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]

    try:
        for _ in range(_MAX_ITERS):
            response = client.chat.completions.create(
                model=_MODEL,
                messages=messages,
                tools=_TOOL_SCHEMAS,
                tool_choice="auto",
                temperature=0.3,  # keep tool selection focused and predictable
            )
            msg = response.choices[0].message

            # No tool calls → the LLM is done; capture its wrap-up and stop.
            if not msg.tool_calls:
                session["final_message"] = (msg.content or "").strip()
                break

            # Record the assistant turn, then run each requested tool.
            messages.append(_assistant_message(msg))
            for tool_call in msg.tool_calls:
                result = _dispatch(tool_call, session)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    }
                )

            # A tool set a terminal error (e.g. no_results) → exit early down the
            # error branch without calling any further tools.
            if session["error"]:
                break
        else:
            # Loop exhausted without a natural finish.
            if not session["error"] and not session["fit_card"]:
                session["error"] = (
                    "FitFindr couldn't complete the request in the allotted steps. "
                    "Please try rephrasing your query."
                )
    except Exception as exc:  # network/API failure mid-loop — fail gracefully
        if not session["error"]:
            session["error"] = f"Something went wrong while planning: {exc}"

    return session


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from utils.data_loader import get_example_wardrobe, get_empty_wardrobe

    print("=== Happy path: graphic tee ===\n")
    session = run_agent(
        query="looking for a vintage graphic tee under $30",
        wardrobe=get_example_wardrobe(),
    )
    if session["error"]:
        print(f"Error: {session['error']}")
    else:
        print(f"Found: {session['selected_item']['title']}")
        print(f"\nOutfit: {session['outfit_suggestion']}")
        print(f"\nFit card: {session['fit_card']}")

    print("\n\n=== No-results path ===\n")
    session2 = run_agent(
        query="designer ballgown size XXS under $5",
        wardrobe=get_example_wardrobe(),
    )
    print(f"Error message: {session2['error']}")
