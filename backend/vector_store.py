"""
Session memory: a small wrapper around FAISS for storing and retrieving
text "chunks" by meaning (embeddings), scoped to a single planning session.

Relationship to state["search_results"]:
- Tavily returns JSON: a list of {"title": ..., "url": ..., "content": ...}.
- That JSON list IS state["search_results"] - the Itinerary agent reads it
  directly to write the plan. Nothing about that changes.
- SessionMemory is a SEPARATE index built FROM the same JSON: for each
  result, Research calls remember(content, metadata={"title":, "url":,
  "trip_id":}). It doesn't replace the JSON - it's an additional,
  searchable-by-meaning copy of it, used for follow-up Q&A:

  While server is running: recall() searches the in-RAM session index.
  After restart: recall_by_trip() loads the persistent index from disk,
  extracts only that trip's vectors (no re-embedding), and searches them.
"""

from __future__ import annotations

import numpy as np

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
        self._path = settings.faiss_store_path
        self._store: FAISS | None = self._load()

    def _load(self) -> FAISS | None:
        if self._path.exists():
            return FAISS.load_local(
                str(self._path),
                self._embeddings,
                allow_dangerous_deserialization=True,
            )
        return None

    def _save(self) -> None:
        self._path.mkdir(parents=True, exist_ok=True)
        self._store.save_local(str(self._path))

    def remember(self, text: str, metadata: dict | None = None) -> None:
        """Embed `text` and store it with metadata (title, url, trip_id)."""
        doc = Document(page_content=text, metadata=metadata or {})
        if self._store is None:
            self._store = FAISS.from_documents([doc], self._embeddings)
        else:
            self._store.add_documents([doc])
        self._save()

    def recall(self, query: str, k: int = 3) -> list[tuple[Document, float]]:
        """Semantic search over the full in-RAM index. Used during /ask when
        the session is still alive (server not restarted)."""
        if self._store is None:
            return []
        return self._store.similarity_search_with_score(query, k=k)

    def recall_by_trip(self, query: str, trip_id: str, k: int = 3) -> list[tuple[Document, float]]:
        """Semantic search scoped to one trip, used after a server restart.

        Instead of re-embedding the chunks from scratch, this reconstructs
        the existing vectors from the persistent index by internal ID —
        only 1 Google API call (to embed the query), not ~15.
        """
        if self._store is None:
            return []

        # Pull docs + their existing vectors for this trip only
        text_embeddings: list[tuple[str, list[float]]] = []
        trip_docs: list[Document] = []

        for internal_id, doc_uuid in self._store.index_to_docstore_id.items():
            doc = self._store.docstore._dict.get(doc_uuid)
            if doc is None or doc.metadata.get("trip_id") != trip_id:
                continue
            vector = self._store.index.reconstruct(int(internal_id))
            text_embeddings.append((doc.page_content, vector.tolist()))
            trip_docs.append(doc)

        if not text_embeddings:
            return []

        # Build a small temp index from the extracted vectors — no API calls
        temp_store = FAISS.from_embeddings(
            text_embeddings=text_embeddings,
            embedding=self._embeddings,
        )
        return temp_store.similarity_search_with_score(query, k=k)


if __name__ == "__main__":
    # Manual test: run `python vector_store.py` (requires GOOGLE_API_KEY in .env)
    memory = SessionMemory()

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

    for r in search_results:
        memory.remember(r["content"], metadata={"title": r["title"], "url": r["url"]})

    print("--- Follow-up retrieval (top 2, with sources) ---")
    for doc, distance in memory.recall("What about vegetarian food options?", k=2):
        print(f"distance={distance:.4f}  source={doc.metadata['title']!r}  "
              f"chunk={doc.page_content[:50]!r}...")
