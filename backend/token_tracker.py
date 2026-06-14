"""
Token usage tracking.

Every Gemini call returns response.usage_metadata, e.g.:
    {"input_tokens": 12, "output_tokens": 22, "total_tokens": 34, ...}

We track this at two levels:

1. Per-session (TokenTracker instance) - created fresh for each /plan
   request, stored in the LangGraph state alongside search_results etc.
   Lets us tell the user "this trip plan cost 312 tokens".

2. Global (the `global_tracker` singleton below) - one running total for
   the whole app's lifetime, across all sessions. Lets us watch overall
   free-tier quota usage.

Usage in an agent node:
    response = llm.invoke(prompt)
    track_usage(response.usage_metadata, state["token_tracker"])
"""

from __future__ import annotations


class TokenTracker:
    """Accumulates token usage across one or more LLM calls."""

    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0
        self.calls = 0

    def add(self, usage_metadata: dict) -> None:
        """Add one LLM call's usage_metadata to the running totals."""
        self.input_tokens += usage_metadata.get("input_tokens", 0)
        self.output_tokens += usage_metadata.get("output_tokens", 0)
        self.total_tokens += usage_metadata.get("total_tokens", 0)
        self.calls += 1

    def summary(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "calls": self.calls,
        }


# Shared across the whole app's lifetime - every session's usage adds to
# this too, so we can see total tokens spent across all users/sessions.
global_tracker = TokenTracker()


def track_usage(usage_metadata: dict, session_tracker: TokenTracker | None = None) -> None:
    """
    Record one LLM call's usage in the global tracker, and (if given) also
    in a per-session tracker.

    session_tracker is typically state["token_tracker"] - pass None if
    there's no session context (e.g. a one-off script).
    """
    global_tracker.add(usage_metadata)
    if session_tracker is not None:
        session_tracker.add(usage_metadata)


if __name__ == "__main__":
    # Manual test: run `python token_tracker.py` - no API keys needed.
    session = TokenTracker()

    # Simulate three LLM calls during one /plan request.
    track_usage({"input_tokens": 12, "output_tokens": 22, "total_tokens": 34}, session)
    track_usage({"input_tokens": 150, "output_tokens": 80, "total_tokens": 230}, session)
    track_usage({"input_tokens": 40, "output_tokens": 15, "total_tokens": 55}, session)

    print("Session summary:", session.summary())
    print("Global summary:", global_tracker.summary())

    # A second, separate session - global keeps accumulating, session doesn't.
    other_session = TokenTracker()
    track_usage({"input_tokens": 10, "output_tokens": 10, "total_tokens": 20}, other_session)

    print("\nOther session summary:", other_session.summary())
    print("Global summary (after second session):", global_tracker.summary())
