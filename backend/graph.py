"""
LangGraph wiring for the trip-planning pipeline:

    research -> itinerary -> budget_check -> (loop back to research, or END)

See state.py for the shared TripState ("notebook") and agents.py for what
each node does. This file is just the wiring: nodes + edges.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from backend.agents import budget_check, itinerary, research
from backend.state import TripState


def _route_after_budget_check(state: TripState) -> str:
    """Conditional edge: budget_check sets state["retry"], we just read it."""
    return "research" if state["retry"] else END


def build_graph():
    graph = StateGraph(TripState)

    graph.add_node("research", research)
    graph.add_node("itinerary", itinerary)
    graph.add_node("budget_check", budget_check)

    graph.set_entry_point("research")
    graph.add_edge("research", "itinerary")
    graph.add_edge("itinerary", "budget_check")
    graph.add_conditional_edges("budget_check", _route_after_budget_check)

    return graph.compile()


if __name__ == "__main__":
    # Manual test: run `python graph.py` (requires GOOGLE_API_KEY and
    # TAVILY_API_KEY in .env). Runs the full pipeline, including the
    # budget retry loop if the first plan is over budget.
    from backend.token_tracker import TokenTracker
    from backend.vector_store import SessionMemory

    app = build_graph()

    initial_state: TripState = {
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

    final_state = app.invoke(initial_state)

    print("--- Final itinerary ---")
    print(final_state["itinerary"])

    print("\n--- Budget ---")
    print(f"estimated_cost: {final_state['estimated_cost']}")
    print(f"over_budget: {final_state['over_budget']}")
    print(f"retry_count: {final_state['retry_count']}")
    print(f"notes: {final_state['budget_notes']}")

    print("\n--- Token usage ---")
    print(final_state["token_tracker"].summary())
    print(f"\nTotal search_results: {len(final_state['search_results'])}")
