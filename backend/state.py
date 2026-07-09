"""
Shared LangGraph state - the "notebook" passed from node to node in the
trip-planning graph (see graph.py).

Each node in agents.py reads fields earlier nodes have written, and writes
its own results back into this same dict. LangGraph routes it according to
the edges defined in graph.py.
"""

from __future__ import annotations

from typing import TypedDict

from backend.token_tracker import TokenTracker
from backend.vector_store import SessionMemory


class TripState(TypedDict):
    # --- Set once, before the graph runs ---
    trip_id: str       # generated upfront so research can tag FAISS chunks
    origin: str        # where the trip starts from, e.g. "Bangalore, India"
    destination: str
    start_date: str   # "YYYY-MM-DD"
    end_date: str     # "YYYY-MM-DD"
    budget: float
    email: str
    preferences: str  # free text, e.g. "vegetarian, relaxed pace, beaches"

    # --- Built up by research ---
    search_results: list[dict]  # [{"title":, "url":, "content":}, ...]

    # --- Built up by itinerary ---
    itinerary: str

    # --- Built up by budget_check ---
    estimated_cost: float
    over_budget: bool
    budget_notes: str
    retry_count: int
    retry: bool

    # --- Shared helper objects (not persisted to trips.json) ---
    memory: SessionMemory
    token_tracker: TokenTracker
