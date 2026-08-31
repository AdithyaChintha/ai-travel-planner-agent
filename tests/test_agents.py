import os
import builtins
from types import SimpleNamespace

import pytest

# Set dummy environment variables required by backend.config.Settings before importing modules
os.environ.setdefault("GOOGLE_API_KEY", "dummy_key")
os.environ.setdefault("TAVILY_API_KEY", "dummy_key")
os.environ.setdefault("GMAIL_ADDRESS", "dummy@example.com")
os.environ.setdefault("GMAIL_APP_PASSWORD", "dummy_password")

# Import the functions we want to test after env vars are set
from backend.agents import _build_queries, budget_check
from backend.state import TripState

# Helper to create a minimal token tracker placeholder
class DummyTokenTracker:
    def add(self, *args, **kwargs):
        pass
    def summary(self):
        return ""

# ---------------------------------------------------------------------------
# _build_queries tests
# ---------------------------------------------------------------------------

def test_build_queries_initial_no_preferences():
    state: TripState = {
        "origin": "Bangalore",
        "destination": "Goa",
        "start_date": "2026-07-01",
        "preferences": "",
        "retry_count": 0,
        "search_results": [],
        "memory": SimpleNamespace(),
    }
    queries = _build_queries(state)
    assert len(queries) == 4
    assert queries[0].startswith("flights, trains or buses from")
    assert "weather in" in queries[-1]


def test_build_queries_initial_with_preferences():
    state: TripState = {
        "origin": "Bangalore",
        "destination": "Goa",
        "start_date": "2026-07-01",
        "preferences": "vegetarian",
        "retry_count": 0,
        "search_results": [],
        "memory": SimpleNamespace(),
    }
    queries = _build_queries(state)
    # Should include an extra query for preferences
    assert len(queries) == 5
    assert any("vegetarian" in q for q in queries)

# ---------------------------------------------------------------------------
# budget_check retry logic test
# ---------------------------------------------------------------------------

def test_budget_check_sets_retry_when_over_budget(monkeypatch):
    # Mock the LLM to return a structured response indicating over budget
    class DummyRaw:
        usage_metadata = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}

    class DummyResponse:
        def __init__(self, parsed):
            self.parsed = parsed
            self.raw = DummyRaw()

    class DummyLLM:
        def with_structured_output(self, model_cls, include_raw=False):
            return self
        def invoke(self, prompt):
            parsed = SimpleNamespace(estimated_cost=150.0, over_budget=True, notes="too pricey")
            return {"parsed": parsed, "raw": DummyRaw()}

    monkeypatch.setattr("backend.agents.get_llm", lambda *args, **kwargs: DummyLLM())
    monkeypatch.setattr("backend.agents.track_usage", lambda *a, **kw: None)

    state: TripState = {
        "budget": 100.0,
        "itinerary": "Day 1: ...",
        "token_tracker": DummyTokenTracker(),
        "retry_count": 0,
        "retry": False,
    }
    updated = budget_check(state)
    assert updated["over_budget"] is True
    assert updated["retry"] is True
    assert updated["retry_count"] == 1
    assert updated["estimated_cost"] == 150.0
    assert "too pricey" in updated["budget_notes"]
