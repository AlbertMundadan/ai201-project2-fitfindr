# FitFindr — planning.md

## Tools

List every tool your agent will use. For each tool, fill in all four fields.
You must have at least 3 tools. The three required tools are listed — add any additional tools below them.

### Tool 1: search_listings

**What it does:**

<!-- Describe what this tool does in 1–2 sentences -->

Finds and returns the top 3 matching listings for the input description, sorted by relevance.

**Input parameters:**

<!-- List each parameter, its type, and what it represents -->

- `description` (str): Target item description
- `size` (str, optional): Target item size — case-insensitive substring match. Defaults to `None` (no size filter).
- `max_price` (float, optional): Inclusive max price. Defaults to `None` (no price filter).

**What it returns:**

<!-- Describe the return value — what fields does a result contain? -->

Returns a **list** of up to the top 3 matching listing dicts, sorted by relevance (best first). Each dict has the original listing fields (`id`, `title`, `description`, `category`, `style_tags`, `size`, `condition`, `price`, `colors`, `brand`, `platform`) — the relevance score is used internally for sorting and is **not** added to the output. Returns an empty list if nothing matches (never raises).

**What happens if it fails or returns nothing:**

If it fails, the agent should not call any other tools and return a response to the user asking them to broaden their search or try another search because no listings were found.

<!-- What should the agent do if no listings match? -->

---

### Tool 2: suggest_outfit

**What it does:**

<!-- Describe what this tool does in 1–2 sentences -->

Given an input of a new article of clothing, find a pairing article of clothing that is already in the user's wardrobe.

**Input parameters:**

<!-- List each parameter, its type, and what it represents -->

- `new_item` (dict): A listing dict (the item the user is considering buying).
- `wardrobe` (dict): A wardrobe dictionary containing all clothign in wardrobe. This may be empty.

**What it returns:**

<!-- Describe the return value -->

A non-empty string with outfit suggestions.

**What happens if it fails or returns nothing:**

<!-- What should the agent do if the wardrobe is empty or no outfit can be suggested? -->

If the wardrobe is empty, offer general styling advice for the item
rather than raising an exception or returning an empty string.

---

### Tool 3: create_fit_card

**What it does:**

<!-- Describe what this tool does in 1–2 sentences -->

Generates a short, casual Instagram/TikTok-style caption for the chosen item and
the suggested outfit. Uses a higher LLM temperature so the caption comes out
different for different inputs.

**Input parameters:**

<!-- List each parameter, its type, and what it represents -->

- `outfit` (str): The outfit suggestion text returned by suggest_outfit()
- `new_item` (dict): A listing dict for the new item

**What it returns:**

A short caption to use as a caption for social media relevant to the new outfit item.

<!-- Describe the return value -->

**What happens if it fails or returns nothing:**

<!-- What should the agent do if the outfit data is incomplete? -->

If it is incomplete write a caption for whatever the new item is.
If it is empty, exit and return output to user that it was unable to generate a caption b/c there were no sugggested item

---

### Additional Tools (if any)

<!-- Copy the block above for any tools beyond the required three -->

N/A

## Planning Loop

**How does your agent decide which tool to call next?**

<!-- Describe the logic your planning loop uses. What does it look at? What conditions change its behavior? How does it know when it's done? -->

The agent uses an **LLM tool-calling loop** (Groq's OpenAI-compatible
function-calling API, model `llama-3.3-70b-versatile`, which supports tool use).
The agent does **not** hard-code the order of tool calls, the LLM chooses the next tool based on what previous tools have returned.

**The loop:**

1. `run_agent(query, wardrobe)` builds a `messages` list with a **system prompt**
   (describes FitFindr's job, the three tools, and the rules below) and the
   user's query.
2. The three tools are passed to the LLM as **function schemas** with
   `tool_choice="auto"`.
3. Each turn:
   - Call Groq chat completions with `messages` + tool schemas.
   - **If the response has no `tool_calls`**, the LLM has decided it is done —
     store its text as `session["final_message"]` and exit the loop.
   - **Otherwise**, for each tool call: a `dispatch()` function executes the
     _real_ Python tool, mutates `session`, and appends a compact tool-result
     message back into `messages`. Then loop again.
4. The loop is wrapped in a `MAX_ITERS` cap so a misbehaving LLM can never spin
   forever.

**Conditions that change its behavior (why it isn't a fixed sequence):**

- If `search_listings` returns nothing, `dispatch` sets `session["error"]` and its
  tool result is a `no_results` message. The system prompt instructs the LLM to
  **stop and ask the user to broaden the search** — it must not call
  `suggest_outfit` on empty input.
- For a "how would I style my X" query with no buying intent, the LLM may skip
  `search_listings` entirely.
- If the LLM calls a tool out of order (e.g. `suggest_outfit` before any search),
  the per-tool **state guards** return an error result telling it what to do
  first, and the LLM self-corrects.

**How it knows it's done:** the loop ends when the LLM returns a turn with no
tool calls (a natural-language wrap-up), or when `MAX_ITERS` is hit.

---

## State Management

**How does information from one tool get passed to the next?**

<!-- Describe how your agent stores and accesses state within a session. What data is tracked? How is it passed between tool calls? -->

There are **two memories** in a session:

1. **`messages`** — the LLM's conversational memory. Holds the system prompt, the
   user query, the LLM's tool-call requests, and the compact tool results. This
   is what lets the LLM reason about what's already happened.
2. **`session` dict** — the structured **source of truth** (defined in
   `_new_session()` in `agent.py`). Every tool result is stored here.

**Key idea — state injection.** The LLM only chooses _which_ tool to call and
supplies lightweight arguments (search keywords/size/price). It does **not**
re-serialize the full item or outfit dicts between tools — that would be fragile
and error-prone. Instead, `dispatch()` injects the real objects straight from
`session`. So the item found by `search_listings` flows into `suggest_outfit`
automatically, and the user never re-enters anything.

To keep the LLM's context small, tool results returned into `messages` are
**compact summaries** (e.g. top listing titles + ids + prices), while the full
dicts live only in `session`.

**Session fields and which tool writes each:**

| Field               | Written by           | Contents                                                                                |
| ------------------- | -------------------- | --------------------------------------------------------------------------------------- |
| `query`             | init                 | original user query                                                                     |
| `parsed`            | search_listings call | `description` / `size` / `max_price` the LLM extracted (this is the query-parsing step) |
| `search_results`    | search_listings      | list of matching listing dicts                                                          |
| `selected_item`     | search_listings      | top result, injected into the next two tools                                            |
| `wardrobe`          | init                 | user's wardrobe, injected into suggest_outfit                                           |
| `outfit_suggestion` | suggest_outfit       | suggestion string, injected into create_fit_card                                        |
| `fit_card`          | create_fit_card      | final caption string                                                                    |
| `final_message`     | loop end             | LLM's natural-language wrap-up to the user                                              |
| `error`             | any tool / loop      | set if the interaction ended early                                                      |

---

## Error Handling

For each tool, describe the specific failure mode you're handling and what the agent does in response.

| Tool            | Failure mode                          | Agent response                                                                                                                                                                                                                                                |
| --------------- | ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| search_listings | No results match the query            | Tool returns `[]`; `dispatch` sets `session["error"]` and sends a `no_results` message back to the LLM. The system prompt tells the LLM to **stop and ask the user to broaden** (drop size/price, try other keywords) — it must not call the downstream tools.   |
| suggest_outfit  | Wardrobe is empty                     | Handled **inside the tool**: it returns general styling advice instead of an empty string, so it never fails. Loop guard: if `session["selected_item"]` is missing (called out of order), `dispatch` returns an error result telling the LLM to search first. |
| create_fit_card | Outfit input is missing or incomplete | Tool guards against an empty/whitespace `outfit` and returns a descriptive message (no exception). Loop guard: if `session["outfit_suggestion"]` is missing, `dispatch` returns an error result telling the LLM to run `suggest_outfit` first.                |

**Loop-level guards (cross-cutting):**

- **`MAX_ITERS` cap** — the loop runs a bounded number of turns so a misbehaving
  LLM can't spin forever; on hitting the cap, set `session["error"]` and return.
- **Malformed tool arguments** — if a tool call's JSON arguments fail to parse,
  catch it and return an error result into `messages` so the LLM can retry with
  corrected arguments instead of crashing the agent.
- **Missing API key** — `_get_groq_client()` raises a clear `ValueError` if
  `GROQ_API_KEY` is not set.

---

## Architecture

<!-- Draw a diagram of your agent showing how the components connect:
     User input → Planning Loop → Tools (search_listings, suggest_outfit, create_fit_card)
                                                                          ↕
                                                                   State / Session
     Show what triggers each tool, how state flows between them, and where error paths branch off.
     ASCII art, a Mermaid diagram (https://mermaid.js.org/syntax/flowchart.html), or an embedded
     sketch are all fine. You'll share this diagram with an AI tool when asking it to implement
     the planning loop and each individual tool. -->

```text
User query  ("vintage graphic tee under $30, size M"  + wardrobe)
    │
    ▼
┌─ PLANNING LOOP (run_agent) ──────────────────────────────────────────┐
│   messages=[system, user] ──► Groq LLM (tools=3, tool_choice="auto")  │
│            ▲                              │                           │
│            │ compact result              │ LLM picks next tool        │
│            └────────── dispatch() ◄───────┘ (re-loops until no calls)  │
└───────────────────────────┬──────────────────────────────────────────┘
                            │  dispatch() runs the real tool & reads/writes Session
    ┌───────────────────────┼──────────────────────────────────────────────┐
    │                                                                       │
    ├─► search_listings(description, size, max_price)                       │
    │       │ results=[]                                                    │
    │       ├──► [ERROR] no_results → LLM stops, asks user to broaden ──────┤
    │       │                                                               │
    │       │ results=[item, ...]                                           │
    │       ▼                                                               │
    │   Session: search_results=[...],  selected_item = results[0]          │
    │       │      inject selected_item + wardrobe                          │
    ├─► suggest_outfit(selected_item, wardrobe)                             │
    │       │      (empty wardrobe → general advice; never empty)           │
    │   Session: outfit_suggestion = "..."                                  │
    │       │      inject outfit_suggestion + selected_item                 │
    └─► create_fit_card(outfit_suggestion, selected_item)                   │
            │                                                               │
        Session: fit_card = "..."                                          │
            │                                          error path returns here
            ▼                                                               │
        Return session  ◄────────────────────────────────────────────────────┘
        (selected_item, outfit_suggestion, fit_card, final_message, error)
```

**Components:** *User* (query + wardrobe) → *Planning Loop* (the Groq LLM +
`dispatch()`) → the three *tools* → *Session state* (single source of truth).

**Data-flow arrows (labeled above):** the LLM extracts `description/size/max_price`
into `search_listings`; `dispatch()` then **injects** `selected_item` + `wardrobe`
into `suggest_outfit`, and `outfit_suggestion` + `selected_item` into
`create_fit_card`. The LLM picks *which* tool; the real objects flow through
`Session`, not through the LLM (state injection).

**Error branch:** when `search_listings` returns `[]`, the `no_results` result
makes the LLM stop and ask the user to broaden — the flow exits early down the
right rail to `Return session` without ever calling the downstream tools.
(`MAX_ITERS` and out-of-order state guards are the other early-exit points.)

---

## AI Tool Plan

<!-- For each part of the implementation below, describe:
     - Which AI tool you plan to use (Claude, Copilot, ChatGPT, etc.)
     - What you'll give it as input (which sections of this planning.md, your agent diagram)
     - What you expect it to produce
     - How you'll verify the output matches your spec before moving on

     "I'll use AI to help me code" is not a plan.
     "I'll give Claude my Tool 1 spec (inputs, return value, failure mode) and ask it to implement
     search_listings() using load_listings() from the data loader — then test it against 3 queries
     before trusting it" is a plan. -->

**Milestone 3 — Individual tool implementations:**

I'll use **Claude (Claude Code)** to implement `tools.py`. For each tool I hand it
that tool's section from this planning.md — the inputs, the return shape, and the
failure mode — plus the existing stub docstrings and `utils/data_loader.py`
(`load_listings`, `get_example_wardrobe`, `get_empty_wardrobe`). I expect:

- `search_listings`: keyword-overlap scoring with optional size/price filters,
  returns top matches sorted by relevance, `[]` on no match (no exception).
- `suggest_outfit` / `create_fit_card`: Groq prompts that follow the style rules,
  with the empty-wardrobe / empty-outfit fallbacks built in.

**Verification:** run each tool standalone against a few queries before wiring the
agent — confirm "vintage graphic tee under $30" returns graphic-tee listings (top
match `lst_006`, the bootleg graphic tee), an empty wardrobe still yields advice,
and an empty outfit yields a message rather than a crash.

**Milestone 4 — Planning loop and state management:**

I'll give Claude the **Planning Loop**, **State Management**, **Error Handling**,
and **Architecture** sections above as the spec for `agent.py` — the
LLM-tool-calling loop, the `dispatch()` state-injection pattern, the session
fields, and the loop-level guards (`MAX_ITERS`, JSON-arg errors). Then wire
`handle_query` in `app.py` to call `run_agent` and render the session fields into
the three Gradio panels.

**Verification:** the two CLI test cases already in `agent.py` —

- Happy path `"looking for a vintage graphic tee under $30"` → finds a tee,
  produces an outfit + fit card.
- No-results path `"designer ballgown size XXS under $5"` → loop stops after the
  empty search and returns an ask-to-broaden message, with no downstream calls.

---

## A Complete Interaction (Step by Step)

Write out what a full user interaction looks like from start to finish — tool call by tool call. Use a specific example query.

**Example user query:** "I'm looking for a vintage graphic tee under $30. I mostly wear baggy jeans and chunky sneakers. What's out there and how would I style it?"

**Step 1:**

<!-- What does the agent do first? Which tool is called? With what input? -->

Agent passes input to LLM along with the tool descriptions to determine which tool to use. The LLM will split the data up into different arguements for the tool call.
In this case the LLM should determine the need to call the search_listings() tool with appropriate arguements.

**Step 2:**

<!-- What happens next? What was returned from step 1? What tool is called now? -->

The agent's `dispatch()` runs the real `search_listings()` tool and stores the
results in `session["search_results"]`, with the top match saved as
`session["selected_item"]`. It hands a **compact summary** (titles + ids +
prices) back to the LLM, not the full dicts.
In this case `search_listings()` returns a matching tee. **If instead it returned
`[]`** (the no-results branch), `dispatch()` would send back `no_results` and the
LLM would stop here and ask the user to broaden the search — `suggest_outfit`
would not be called.

**Step 3:**

The agent calls the LLM again with the tool result to decide what's next. Here the
LLM chooses `suggest_outfit()`. Note the LLM does **not** re-pass the item — when
`dispatch()` runs `suggest_outfit()` it **injects** `session["selected_item"]` and
`session["wardrobe"]` itself (state injection), so the found tee flows straight in
without the user re-entering anything. The suggestion is stored in
`session["outfit_suggestion"]` and summarized back to the LLM.
The loop continues: the LLM then calls `create_fit_card()`, and `dispatch()`
injects `session["outfit_suggestion"]` + `session["selected_item"]`, storing the
caption in `session["fit_card"]`.

**Final output to user:**
When the LLM has nothing left to call, it returns a natural-language wrap-up
(stored as `session["final_message"]`) and the loop ends. The Gradio UI renders
the structured session fields into three panels: the selected listing, the outfit
suggestion, and the fit-card caption — so the user sees the result of every tool
call, not just the caption.

<!-- What does the user actually see at the end? -->
