"""
Tavily web search wrapper.

The Research agent node calls web_search(query) to get real-time results
from the web. Each result becomes one item in state["search_results"]
(the JSON list the Itinerary agent reads directly to write the plan) and
is also passed to SessionMemory.remember() for dedup/follow-up retrieval
(see vector_store.py).
"""

from __future__ import annotations

from tavily import TavilyClient

from backend.config import settings

_client = TavilyClient(api_key=settings.tavily_api_key)


def web_search(query: str, max_results: int | None = None) -> list[dict]:
    """
    Run a Tavily search and return results as a list of
    {"title": ..., "url": ..., "content": ...} dicts.

    max_results defaults to settings.tavily_max_results (kept small to
    stay budget-friendly on the free tier and keep prompts to Gemini
    compact).
    """
    response = _client.search(
        query=query,
        max_results=max_results or settings.tavily_max_results,
    )
    return [
        {"title": r["title"], "url": r["url"], "content": r["content"]}
        for r in response.get("results", [])
    ]


if __name__ == "__main__":
    # Manual test: run `python tools.py` (requires TAVILY_API_KEY in .env)
    results = web_search("best time to visit Goa for budget travel")
    for r in results:
        print(f"- {r['title']} ({r['url']})")
        print(f"  {r['content'][:120]}...")
