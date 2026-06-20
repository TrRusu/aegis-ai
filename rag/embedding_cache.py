"""Caches embedding vectors so repeated text is never re-embedded.
"""

from langchain_core.embeddings import Embeddings
from langchain_core.stores import ByteStore
from langchain_classic.embeddings import CacheBackedEmbeddings


class CachedEmbeddings:

    def __init__(self, embeddings: Embeddings, store: ByteStore, namespace: str):
        self._cached = CacheBackedEmbeddings.from_bytes_store(
            embeddings, store, namespace=namespace, query_embedding_cache=True, key_encoder="sha256",
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._cached.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._cached.embed_query(text)
