"""
tools.py

The three required FitFindr tools. Each tool is a standalone function that
can be called and tested independently before being wired into the agent loop.

Complete and test each tool before moving to agent.py.

Tools:
    search_listings(description, size, max_price)  → list[dict]
    suggest_outfit(new_item, wardrobe)              → str
    create_fit_card(outfit, new_item)               → str
"""

import os
import re

from dotenv import load_dotenv
from groq import Groq

from utils.data_loader import load_listings

load_dotenv()

# Groq model used for the styling tools. llama-3.3-70b supports tool use and
# follows instructions well for short, voice-y copy.
_MODEL = "llama-3.3-70b-versatile"


# ── Groq client ───────────────────────────────────────────────────────────────

def _get_groq_client():
    """Initialize and return a Groq client using GROQ_API_KEY from .env."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not set. Add it to a .env file in the project root."
        )
    return Groq(api_key=api_key)


def _chat(messages: list[dict], temperature: float = 0.7, max_tokens: int = 400) -> str:
    """Send a chat completion to Groq and return the response text, stripped."""
    client = _get_groq_client()
    resp = client.chat.completions.create(
        model=_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


def _format_item(item: dict) -> str:
    """Render a listing dict as a compact one-line description for prompts."""
    item = item or {}
    parts = [item.get("title", "this piece")]
    if item.get("category"):
        parts.append(f"category: {item['category']}")
    if item.get("colors"):
        parts.append(f"colors: {', '.join(item['colors'])}")
    if item.get("style_tags"):
        parts.append(f"style: {', '.join(item['style_tags'])}")
    if item.get("size"):
        parts.append(f"size: {item['size']}")
    return "; ".join(parts)


# ── search helpers ──────────────────────────────────────────────────────────

# Filler words that carry no relevance signal — dropped before scoring so a
# phrase like "looking for a vintage tee" matches on "vintage" and "tee".
_STOPWORDS = {
    "a", "an", "the", "for", "with", "under", "and", "or", "in", "of", "to",
    "my", "i", "im", "looking", "want", "wanna", "need", "size", "that", "this",
    "some", "something", "really", "please", "find", "me", "is", "are", "on",
    "at", "it", "like", "kinda", "sort",
}


def _tokens(text: str) -> list[str]:
    """Lowercase alphanumeric tokens with stopwords and 1-char tokens removed."""
    return [
        tok
        for tok in re.findall(r"[a-z0-9]+", text.lower())
        if tok not in _STOPWORDS and len(tok) >= 2
    ]


def _score_listing(listing: dict, query_tokens: list[str]) -> int:
    """Score a listing by weighted keyword overlap with the query tokens.

    Matches in the title or style tags count most; description/color/brand
    matches add smaller signal. A token can match several fields and stack.
    """
    title = listing.get("title", "").lower()
    tags = " ".join(listing.get("style_tags", [])).lower()
    category = listing.get("category", "").lower()
    description = listing.get("description", "").lower()
    colors = " ".join(listing.get("colors", [])).lower()
    brand = (listing.get("brand") or "").lower()

    score = 0
    for tok in query_tokens:
        if tok in title:
            score += 3
        if tok in tags:
            score += 3
        if tok in category:
            score += 2
        if tok in description:
            score += 1
        if tok in colors:
            score += 1
        if tok in brand:
            score += 1
    return score


# ── Tool 1: search_listings ───────────────────────────────────────────────────

def search_listings(
    description: str,
    size: str | None = None,
    max_price: float | None = None,
) -> list[dict]:
    """
    Search the mock listings dataset for items matching the description,
    optional size, and optional price ceiling.

    Args:
        description: Keywords describing what the user is looking for
                     (e.g., "vintage graphic tee").
        size:        Size string to filter by, or None to skip size filtering.
                     Matching is case-insensitive (e.g., "M" matches "S/M").
        max_price:   Maximum price (inclusive), or None to skip price filtering.

    Returns:
        A list of matching listing dicts, sorted by relevance (best match first).
        Returns an empty list if nothing matches — does NOT raise an exception.

    Each listing dict has the following fields:
        id, title, description, category, style_tags (list), size,
        condition, price (float), colors (list), brand, platform

    TODO:
        1. Load all listings with load_listings().
        2. Filter by max_price and size (if provided).
        3. Score each remaining listing by keyword overlap with `description`.
        4. Drop any listings with a score of 0 (no relevant matches).
        5. Sort by score, highest first, and return the listing dicts.

    Before writing code, fill in the Tool 1 section of planning.md.
    """
    listings = load_listings()
    query_tokens = _tokens(description or "")

    # Without any usable keywords we can't score relevance — return nothing
    # rather than dumping the whole dataset.
    if not query_tokens:
        return []

    scored: list[tuple[int, dict]] = []
    for listing in listings:
        # Hard filters first — these are constraints, not preferences.
        if max_price is not None and listing.get("price", 0) > max_price:
            continue
        if size and size.strip():
            if size.strip().upper() not in str(listing.get("size", "")).upper():
                continue

        score = _score_listing(listing, query_tokens)
        if score <= 0:
            continue
        scored.append((score, listing))

    # Best relevance first; break ties with the cheaper item.
    scored.sort(key=lambda pair: (-pair[0], pair[1].get("price", 0)))
    return [listing for _, listing in scored[:3]]


# ── Tool 2: suggest_outfit ────────────────────────────────────────────────────

def suggest_outfit(new_item: dict, wardrobe: dict) -> str:
    """
    Given a thrifted item and the user's wardrobe, suggest 1–2 complete outfits.

    Args:
        new_item: A listing dict (the item the user is considering buying).
        wardrobe: A wardrobe dict with an 'items' key containing a list of
                  wardrobe item dicts. May be empty — handle this gracefully.

    Returns:
        A non-empty string with outfit suggestions.
        If the wardrobe is empty, offer general styling advice for the item
        rather than raising an exception or returning an empty string.

    TODO:
        1. Check whether wardrobe['items'] is empty.
        2. If empty: call the LLM with a prompt for general styling ideas
           (what kinds of items pair well, what vibe it suits, etc.).
        3. If not empty: format the wardrobe items into a prompt and ask
           the LLM to suggest specific outfit combinations using the new item
           and named pieces from the wardrobe.
        4. Return the LLM's response as a string.

    Before writing code, fill in the Tool 2 section of planning.md.
    """
    new_item = new_item or {}
    items = wardrobe.get("items", []) if isinstance(wardrobe, dict) else []
    item_desc = _format_item(new_item)

    system = (
        "You are FitFindr, a sharp, friendly secondhand-fashion stylist. "
        "Give concrete, wearable styling advice in a warm, casual voice. "
        "Be specific about pieces, silhouettes, and how to wear them. "
        "Keep it to 2-4 sentences. No preamble, no bullet lists."
    )

    if not items:
        # Empty wardrobe → general styling advice instead of failing.
        user = (
            f"A user is considering this secondhand piece:\n{item_desc}\n\n"
            "They haven't told us what's in their closet yet. Suggest how to style "
            "this piece in general: what kinds of items pair well with it, what vibe "
            "it suits, and one concrete outfit idea using common wardrobe staples."
        )
    else:
        closet_lines = []
        for w in items:
            line = f"- {w.get('name', 'item')} ({w.get('category', 'item')})"
            if w.get("style_tags"):
                line += f" — {', '.join(w['style_tags'])}"
            if w.get("notes"):
                line += f" [{w['notes']}]"
            closet_lines.append(line)
        closet = "\n".join(closet_lines)
        user = (
            f"A user is considering this secondhand piece:\n{item_desc}\n\n"
            f"Here is what's already in their wardrobe:\n{closet}\n\n"
            "Suggest 1-2 complete outfit combinations that pair the new piece with "
            "specific items named from their wardrobe. Reference the wardrobe pieces "
            "by name and add a small styling tip (how to tuck, roll, layer, etc.)."
        )

    try:
        suggestion = _chat(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            temperature=0.7,
        )
        if suggestion:
            return suggestion
    except Exception:
        pass  

    tags = new_item.get("style_tags") or ["vintage"]
    return (
        f"Make the {new_item.get('title', 'new piece')} the star of the look and "
        f"keep everything else simple — neutral basics, clean denim or trousers, and "
        f"shoes that match its {', '.join(tags[:2])} energy. Let the piece do the talking."
    )


# ── Tool 3: create_fit_card ───────────────────────────────────────────────────

def create_fit_card(outfit: str, new_item: dict) -> str:
    """
    Generate a short, shareable outfit caption for the thrifted find.

    Args:
        outfit:   The outfit suggestion string from suggest_outfit().
        new_item: The listing dict for the thrifted item.

    Returns:
        A 2–3 sentence string usable as an Instagram/TikTok caption.
        If outfit is empty or missing, return a descriptive error message
        string — do NOT raise an exception.

    The caption should:
    - Feel casual and authentic (like a real OOTD post, not a product description)
    - Mention the item name, price, and platform naturally (once each)
    - Capture the outfit vibe in specific terms
    - Sound different each time for different inputs (use higher LLM temperature)

    TODO:
        1. Guard against an empty or whitespace-only outfit string.
        2. Build a prompt that gives the LLM the item details and the outfit,
           and asks for a caption matching the style guidelines above.
        3. Call the LLM and return the response.

    Before writing code, fill in the Tool 3 section of planning.md.
    """
    # Guard: no outfit to caption → descriptive message, not an exception.
    if not outfit or not outfit.strip():
        return (
            "Couldn't generate a fit card — there's no outfit suggestion to caption "
            "yet. Try getting a styling suggestion for the item first."
        )

    new_item = new_item or {}
    title = new_item.get("title", "this piece")
    price = new_item.get("price")
    platform = new_item.get("platform", "secondhand")
    price_str = f"${price:g}" if isinstance(price, (int, float)) else "a steal"

    system = (
        "You write short, authentic outfit captions for Instagram/TikTok OOTD posts. "
        "Sound like a real person hyped about a thrift find — casual, specific, a "
        "little playful. NOT a product description. 2-3 short, chic, and trendy sentences. An emoji or two is "
        "fine. Mention the item, its price, and where it's from naturally, once each."
    )
    user = (
        f"Item: {title} (around {price_str}, from {platform}).\n"
        f"The outfit it's styled in: {outfit}\n\n"
        "Write the caption."
    )

    try:
        # Higher temperature so captions vary across different inputs/runs.
        caption = _chat(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            temperature=1.5,
            max_tokens=200,
        )
        if caption:
            return caption
    except Exception:
        pass  # fall through to a non-LLM fallback so we never crash or return ""

    # Fallback caption — still mentions item, price, and platform.
    return (
        f"thrifted this {title.lower()} off {platform} for {price_str} and it’s "
        f"already my favorite thing 🖤 styled it up — full look soon"
    )
