"""
Stores and retrieves documents in ChromaDB.
"""
import os

from rag.store import ChromaStore
from rag.semantic_cache import SemanticCache


class DocumentStore:

    def __init__(self, store: ChromaStore, cache: SemanticCache | None = None):
        self._store = store
        self._cache = cache

    def get_ingested_documents(self) -> list[str]:
        try:
            results = self._store.vectorstore._collection.get(include=["metadatas"])
            sources = set()
            for meta in results["metadatas"]:
                source = meta.get("source", "")
                if source:
                    sources.add(os.path.basename(source))
            return sorted(sources)
        except Exception:
            return []

    def ingest(self, filepath: str, loader, mode_label: str = "") -> str:
        filename = os.path.basename(filepath)
        if filename in self.get_ingested_documents():
            return f"{filename} is already ingested."
        chunks = loader.load(filepath)
        self._store.add_documents(chunks)
        if self._cache:
            self._cache.clear()
        suffix = f" ({mode_label})" if mode_label else ""
        return f"Ingested {filename} - {len(chunks)} chunks{suffix}."


def make_store(cache: SemanticCache | None = None) -> DocumentStore:
    return DocumentStore(store=ChromaStore(), cache=cache)
