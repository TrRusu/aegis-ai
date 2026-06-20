from unittest.mock import MagicMock
from langchain_classic.storage import InMemoryByteStore
from rag.embedding_cache import CachedEmbeddings


def _make_embedder(vectors: dict) -> MagicMock:
    embedder = MagicMock()
    embedder.embed_documents.side_effect = lambda texts: [vectors[t] for t in texts]
    embedder.embed_query.side_effect = lambda text: vectors[text]
    return embedder


def test_embed_documents_returns_vectors_from_underlying_embedder():
    underlying = _make_embedder({"hello": [0.1, 0.2]})
    cached = CachedEmbeddings(embeddings=underlying, store=InMemoryByteStore(), namespace="test-model")
    assert cached.embed_documents(["hello"]) == [[0.1, 0.2]]


def test_embed_documents_calls_underlying_once_per_unique_text():
    underlying = _make_embedder({"hello": [0.1, 0.2]})
    cached = CachedEmbeddings(embeddings=underlying, store=InMemoryByteStore(), namespace="test-model")
    cached.embed_documents(["hello"])
    cached.embed_documents(["hello"])
    assert underlying.embed_documents.call_count == 1


def test_embed_documents_handles_multiple_texts():
    underlying = _make_embedder({"a": [0.1], "b": [0.2]})
    cached = CachedEmbeddings(embeddings=underlying, store=InMemoryByteStore(), namespace="test-model")
    assert cached.embed_documents(["a", "b"]) == [[0.1], [0.2]]


def test_embed_query_returns_vector_from_underlying_embedder():
    underlying = _make_embedder({"hello": [0.3, 0.4]})
    cached = CachedEmbeddings(embeddings=underlying, store=InMemoryByteStore(), namespace="test-model")
    assert cached.embed_query("hello") == [0.3, 0.4]


def test_embed_query_calls_underlying_once_per_unique_text():
    underlying = _make_embedder({"hello": [0.1, 0.2]})
    cached = CachedEmbeddings(embeddings=underlying, store=InMemoryByteStore(), namespace="test-model")
    cached.embed_query("hello")
    cached.embed_query("hello")
    assert underlying.embed_query.call_count == 1


def test_different_namespace_does_not_share_cache():
    store = InMemoryByteStore()
    underlying = _make_embedder({"hello": [0.1, 0.2]})
    cached_a = CachedEmbeddings(embeddings=underlying, store=store, namespace="model-a")
    cached_b = CachedEmbeddings(embeddings=underlying, store=store, namespace="model-b")
    cached_a.embed_documents(["hello"])
    cached_b.embed_documents(["hello"])
    assert underlying.embed_documents.call_count == 2
