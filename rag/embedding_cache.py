"""Caches embedding vectors so repeated text is never re-embedded.
"""

from langchain_core.embeddings import Embeddings
from langchain_core.stores import ByteStore
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_openai import OpenAIEmbeddings
from app.config import OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL, EMBEDDING_CACHE_DIR


class CachedEmbeddings:

    def __init__(self, embeddings: Embeddings, store: ByteStore, namespace: str):
        self._cached = CacheBackedEmbeddings.from_bytes_store(
            embeddings, store, namespace=namespace, query_embedding_cache=True, key_encoder="sha256",
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._cached.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._cached.embed_query(text)


def make_cached_embeddings() -> CachedEmbeddings:
    underlying = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
    store = LocalFileStore(EMBEDDING_CACHE_DIR)
    return CachedEmbeddings(embeddings=underlying, store=store, namespace=OPENAI_EMBEDDING_MODEL)
