"""
Session memory: a small wrapper around FAISS for storing and retrieving
text "chunks" by meaning (embeddings), scoped to a single planning session.

Relationship to state["search_results"]:
- Tavily returns JSON: a list of {"title": ..., "url": ..., "content": ...}.
- That JSON list IS state["search_results"] - the Itinerary agent reads it
  directly to write the plan. Nothing about that changes.
- SessionMemory is a SEPARATE index built FROM the same JSON: for each
  result, Research calls remember(content, metadata={"title":, "url":}).
  It doesn't replace the JSON - it's an additional, searchable-by-meaning
  copy of it, used for follow-up Q&A:

  "what do I already know that might help answer this new question?"
  -> recall(question, k=3), hand the chunks (+ source metadata) to Gemini
  as context. Falls back to a fresh Tavily search if nothing relevant.
"""

from __future__ import annotations

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from backend.config import settings


class SessionMemory:
    def __init__(self) -> None:
        self._embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
            google_api_key=settings.google_api_key,
        )
        self._store: FAISS | None = None

    def remember(self, text: str, metadata: dict | None = None) -> None:
        """Embed `text` (e.g. one Tavily result's content) and store it,
        keeping `metadata` (e.g. title/url) attached for later."""
        doc = Document(page_content=text, metadata=metadata or {})
        if self._store is None:
            self._store = FAISS.from_documents([doc], self._embeddings)
        else:
            self._store.add_documents([doc])

    def recall(self, query: str, k: int = 3) -> list[tuple[Document, float]]:
        """
        Return up to `k` stored (Document, distance) pairs, closest first.
        Lower distance means more similar. doc.page_content is the text,
        doc.metadata carries whatever was passed to remember() (e.g.
        title/url, for citing sources). Returns [] if nothing remembered yet.
        """
        if self._store is None:
            return []
        return self._store.similarity_search_with_score(query, k=k)


if __name__ == "__main__":
    # Manual test: run `python vector_store.py` (requires GOOGLE_API_KEY in .env)
    memory = SessionMemory()

    # This is what a Tavily response looks like - a list of JSON objects.
    # This list IS what becomes state["search_results"].
    search_results = [
        {
            "title": "Goa Budget Hotels Guide",
            "url": "https://example.com/goa-hotels",
            "content": "Most guesthouses near Calangute and Baga beach "
                        "cost $15-25/night for a double room.",
        },
        {
            "title": "Goa for Vegetarians",
            "url": "https://example.com/goa-veg",
            "content": "Most beach shacks offer a separate veg menu, and "
                        "Anjuna market has several all-veg cafes.",
        },
        {
            "title": "Best Time for Goa Beaches",
            "url": "https://example.com/goa-beaches",
            "content": "The best time to visit Goa's beaches is early "
                        "morning (7-9am) before the heat and crowds build up.",
        },
    ]

    # ...and each item ALSO gets remembered for follow-up retrieval:
    for r in search_results:
        memory.remember(r["content"], metadata={"title": r["title"], "url": r["url"]})

    print("--- Follow-up retrieval (top 2, with sources) ---")
    for doc, distance in memory.recall("What about vegetarian food options?", k=2):
        print(f"distance={distance:.4f}  source={doc.metadata['title']!r}  "
              f"chunk={doc.page_content[:50]!r}...")
