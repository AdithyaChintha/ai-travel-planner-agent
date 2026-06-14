"""
The three agent nodes of the trip-planning graph (see graph.py for how
they're wired together):

    research -> itinerary -> budget_check -> (loop back to research, or END)

Each function takes the shared TripState dict, reads what it needs, writes
its results back into it, and returns it.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from backend.config import settings
from backend.llm import get_llm
from backend.state import TripState
from backend.token_tracker import TokenTracker, track_usage
from backend.tools import web_search
from backend.vector_store import SessionMemory


# ---------------------------------------------------------------------------
# research: no LLM call. Runs templated Tavily searches, using FAISS session
# memory to skip searches we've effectively already done.
# ---------------------------------------------------------------------------

def _build_queries(state: TripState) -> list[str]:
    origin = state["origin"]
    destination = state["destination"]
    preferences = state.get("preferences", "")
    month = datetime.strptime(state["start_date"], "%Y-%m-%d").strftime("%B")

    if state.get("retry_count", 0) == 0:
        queries = [
            f"flights, trains or buses from {origin} to {destination}",
            f"budget hotels and guesthouses in {destination}",
            f"top budget-friendly things to do in {destination}",
            f"weather in {destination} in {month}",
        ]
        if preferences:
            queries.append(f"{preferences} food and dining options in {destination}")
    else:
        # Retry: the previous itinerary was over budget - look specifically
        # for cheaper alternatives instead of repeating the same searches.
        queries = [
            f"cheapest flights, trains or buses from {origin} to {destination}",
            f"cheapest hostels and guesthouses in {destination}",
            f"free or low-cost things to do in {destination}",
        ]

    return queries


def research(state: TripState) -> TripState:
    memory = state["memory"]
    seen_urls = {r["url"] for r in state["search_results"]}

    for query in _build_queries(state):
        for result in web_search(query):
            if result["url"] in seen_urls:
                # Already have this exact source - skip the duplicate.
                continue
            seen_urls.add(result["url"])
            state["search_results"].append(result)
            memory.remember(
                result["content"],
                metadata={"title": result["title"], "url": result["url"]},
            )

    return state


# ---------------------------------------------------------------------------
# itinerary: 1 LLM call. Writes a day-by-day plan using search_results as
# context, respecting budget and preferences.
# ---------------------------------------------------------------------------

def itinerary(state: TripState) -> TripState:
    llm = get_llm()

    context = "\n\n".join(
        f"- {r['title']}: {r['content'][:300]}" for r in state["search_results"]
    )

    retry_hint = ""
    if state.get("retry_count", 0) > 0:
        retry_hint = (
            f"\nIMPORTANT: A previous plan was estimated at "
            f"${state['estimated_cost']:.2f}, which exceeds the budget of "
            f"${state['budget']:.2f}. Revise it to be cheaper - prefer "
            "cheaper transport, budget accommodation, and free or low-cost "
            "activities.\n"
        )

    prompt = (
        f"Plan a trip from {state['origin']} to {state['destination']}, "
        f"from {state['start_date']} to {state['end_date']}.\n"
        f"Total budget: ${state['budget']:.2f} USD - this must cover transport "
        f"to/from {state['destination']} as well as on-ground costs.\n"
        "All costs in this itinerary must be in USD ($). If your research "
        "gives prices in a different currency, convert them to USD using a "
        "reasonable exchange rate.\n"
        f"Preferences: {state.get('preferences') or 'none specified'}.\n"
        f"{retry_hint}\n"
        f"Use the following research to inform the plan:\n{context}\n\n"
        "Write a concise day-by-day itinerary, including how to get there "
        "and back and its estimated cost."
    )

    response = llm.invoke(prompt)
    track_usage(response.usage_metadata, state["token_tracker"])

    state["itinerary"] = response.content
    return state


# ---------------------------------------------------------------------------
# budget_check: 1 LLM call with structured output. Estimates the itinerary's
# total cost and decides whether to loop back to research.
# ---------------------------------------------------------------------------

class BudgetEstimate(BaseModel):
    estimated_cost: float = Field(
        description="Estimated total cost of the trip, in USD."
    )
    over_budget: bool = Field(
        description="True if estimated_cost is greater than the given budget."
    )
    notes: str = Field(
        description="One or two sentence explanation of how the estimate was reached."
    )


def budget_check(state: TripState) -> TripState:
    llm = get_llm()
    structured_llm = llm.with_structured_output(BudgetEstimate, include_raw=True)

    prompt = (
        f"Budget available: ${state['budget']:.2f} USD.\n\n"
        f"Itinerary:\n{state['itinerary']}\n\n"
        "Estimate the total cost of this itinerary in USD (convert any "
        "prices given in other currencies using a reasonable exchange "
        "rate), and determine whether it exceeds the available budget."
    )

    result = structured_llm.invoke(prompt)
    track_usage(result["raw"].usage_metadata, state["token_tracker"])

    estimate: BudgetEstimate = result["parsed"]
    state["estimated_cost"] = estimate.estimated_cost
    state["over_budget"] = estimate.over_budget
    state["budget_notes"] = estimate.notes

    retry_count = state.get("retry_count", 0)
    if estimate.over_budget and retry_count < settings.budget_max_retries:
        state["retry_count"] = retry_count + 1
        state["retry"] = True
    else:
        state["retry"] = False

    return state


if __name__ == "__main__":
    # Manual test: run `python agents.py` (requires GOOGLE_API_KEY and
    # TAVILY_API_KEY in .env). Runs one pass: research -> itinerary ->
    # budget_check, and prints the result.
    state: TripState = {
        "origin": "Bangalore, India",
        "destination": "Goa, India",
        "start_date": "2026-07-01",
        "end_date": "2026-07-04",
        "budget": 200.0,
        "email": "you@example.com",
        "preferences": "vegetarian, relaxed pace",
        "search_results": [],
        "itinerary": "",
        "estimated_cost": 0.0,
        "over_budget": False,
        "budget_notes": "",
        "retry_count": 0,
        "retry": False,
        "memory": SessionMemory(),
        "token_tracker": TokenTracker(),
    }

    state = research(state)
    print(f"--- research: {len(state['search_results'])} search results ---")
    for r in state["search_results"]:
        print(f"  - {r['title']}")

    state = itinerary(state)
    print("\n--- itinerary ---")
    print(state["itinerary"])

    state = budget_check(state)
    print("\n--- budget_check ---")
    print(f"estimated_cost: {state['estimated_cost']}")
    print(f"over_budget: {state['over_budget']}")
    print(f"notes: {state['budget_notes']}")
    print(f"retry: {state['retry']} (retry_count={state['retry_count']})")

    print("\n--- token usage (this session) ---")
    print(state["token_tracker"].summary())
