"""
Unit tests for the three FitFindr tools, with at least one test per failure mode.

"""

import os

from tools import search_listings, suggest_outfit, create_fit_card
from utils.data_loader import get_example_wardrobe, get_empty_wardrobe


# ── search_listings ─────────────────────────────────────────────────────────

def test_search_returns_results():
    results = search_listings("vintage graphic tee", size=None, max_price=50)
    assert isinstance(results, list)
    assert len(results) > 0


def test_search_empty_results():
    # Failure mode: nothing matches → empty list, never an exception.
    results = search_listings("designer ballgown", size="XXS", max_price=5)
    assert results == []


def test_search_price_filter():
    # Failure mode: respect the price ceiling — no over-budget items leak through.
    results = search_listings("jacket", size=None, max_price=10)
    assert all(item["price"] <= 10 for item in results)


def test_search_size_filter():
    # Size filter is case-insensitive and substring-based ("M" matches "S/M").
    results = search_listings("track jacket", size="M", max_price=None)
    assert all("M" in item["size"].upper() for item in results)


def test_search_empty_description():
    # Failure mode: no usable keywords → empty list rather than the whole dataset.
    assert search_listings("", size=None, max_price=None) == []


def test_search_caps_at_three():
    # Returns at most the top 3 matches by relevance.
    results = search_listings("vintage", size=None, max_price=None)
    assert len(results) <= 3


# ── suggest_outfit ──────────────────────────────────────────────────────────

_ITEM = {
    "title": "Vintage Band Tee — Faded Grey",
    "category": "tops",
    "colors": ["grey"],
    "style_tags": ["vintage", "grunge", "band tee"],
    "size": "L",
    "price": 19.0,
    "platform": "depop",
}


def test_suggest_outfit_with_wardrobe():
    out = suggest_outfit(_ITEM, get_example_wardrobe())
    assert isinstance(out, str)
    assert out.strip()


def test_suggest_outfit_empty_wardrobe():
    # Failure mode: empty wardrobe → still returns non-empty general advice.
    out = suggest_outfit(_ITEM, get_empty_wardrobe())
    assert isinstance(out, str)
    assert out.strip()


def test_suggest_outfit_fallback_without_api():
    # Failure mode: LLM unavailable (no API key) → non-empty fallback, no exception.
    saved = os.environ.pop("GROQ_API_KEY", None)
    try:
        out = suggest_outfit(_ITEM, get_example_wardrobe())
    finally:
        if saved is not None:
            os.environ["GROQ_API_KEY"] = saved
    assert isinstance(out, str)
    assert out.strip()


# ── create_fit_card ─────────────────────────────────────────────────────────

def test_create_fit_card_normal():
    card = create_fit_card("Pair with baggy jeans and combat boots", _ITEM)
    assert isinstance(card, str)
    assert card.strip()


def test_create_fit_card_empty_outfit_guard():
    # Failure mode: empty/missing outfit → descriptive message (no LLM call needed).
    for bad_outfit in ("", "   ", None):
        msg = create_fit_card(bad_outfit, _ITEM)
        assert isinstance(msg, str)
        assert msg.strip()


def test_create_fit_card_fallback_without_api():
    # Failure mode: LLM unavailable (no API key) → non-empty fallback caption.
    saved = os.environ.pop("GROQ_API_KEY", None)
    try:
        card = create_fit_card("Pair with baggy jeans and combat boots", _ITEM)
    finally:
        if saved is not None:
            os.environ["GROQ_API_KEY"] = saved
    assert isinstance(card, str)
    assert card.strip()
