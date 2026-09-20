from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb

            client = chromadb.EphemeralClient()
            # Cosine space: embeddings are normalized, so Chroma's own ranking
            # matches the dot-product ranking used by the in-memory fallback.
            self._collection = client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        """Normalize one Document into the shape the store keeps internally."""
        metadata = dict(doc.metadata or {})
        # doc_id lets delete_document() find every chunk of the same document,
        # even when the caller stored several chunks under one id. A caller that
        # already grouped its chunks keeps its own doc_id — do not clobber it.
        metadata.setdefault("doc_id", doc.id)

        record = {
            # Unique per stored chunk: the same doc id may be added more than once.
            "id": f"{doc.id}#{self._next_index}",
            "doc_id": metadata["doc_id"],
            "content": doc.content,
            "metadata": metadata,
            "embedding": self._embedding_fn(doc.content),
        }
        self._next_index += 1
        return record

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        """Rank `records` against `query` by dot product, highest score first."""
        if top_k <= 0 or not records:
            return []

        query_embedding = self._embedding_fn(query)
        results = [
            {
                "id": record["id"],
                "content": record["content"],
                "metadata": record["metadata"],
                "score": _dot(query_embedding, record["embedding"]),
            }
            for record in records
        ]
        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:top_k]

    def _all_records(self) -> list[dict[str, Any]]:
        """Return every stored chunk in the internal record shape."""
        if not self._use_chroma:
            return list(self._store)

        stored = self._collection.get(include=["documents", "metadatas", "embeddings"])
        embeddings = stored.get("embeddings") or []
        return [
            {
                "id": chunk_id,
                "doc_id": (metadata or {}).get("doc_id"),
                "content": content,
                "metadata": dict(metadata or {}),
                "embedding": list(embeddings[i]) if i < len(embeddings) else [],
            }
            for i, (chunk_id, content, metadata) in enumerate(
                zip(stored.get("ids", []), stored.get("documents", []), stored.get("metadatas", []))
            )
        ]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        if not docs:
            return

        records = [self._make_record(doc) for doc in docs]

        if self._use_chroma:
            self._collection.add(
                ids=[r["id"] for r in records],
                documents=[r["content"] for r in records],
                embeddings=[r["embedding"] for r in records],
                metadatas=[r["metadata"] for r in records],
            )
        else:
            self._store.extend(records)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        return self._search_records(query, self._all_records(), top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        if self._use_chroma:
            return self._collection.count()
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        records = self._all_records()

        if metadata_filter:
            records = [
                record
                for record in records
                if all(record["metadata"].get(key) == value for key, value in metadata_filter.items())
            ]

        return self._search_records(query, records, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        if self._use_chroma:
            doomed = [r["id"] for r in self._all_records() if r["metadata"].get("doc_id") == doc_id]
            if not doomed:
                return False
            self._collection.delete(ids=doomed)
            return True

        kept = [record for record in self._store if record["metadata"].get("doc_id") != doc_id]
        if len(kept) == len(self._store):
            return False
        self._store = kept
        return True
