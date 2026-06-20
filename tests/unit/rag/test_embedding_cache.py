from unittest.mock import MagicMock, patch
from langchain_classic.storage import InMemoryByteStore
from rag.embedding_cache import CachedEmbeddings, make_cached_embeddings


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


def test_make_cached_embeddings_wraps_openai_embeddings_with_local_file_store():
    with patch("rag.embedding_cache.OpenAIEmbeddings") as mock_openai_cls, \
         patch("rag.embedding_cache.LocalFileStore") as mock_store_cls, \
         patch("rag.embedding_cache.CacheBackedEmbeddings") as mock_cached_cls:
        make_cached_embeddings()
        mock_openai_cls.assert_called_once()
        mock_store_cls.assert_called_once()
        assert mock_cached_cls.from_bytes_store.call_args[0][0] is mock_openai_cls.return_value
        assert mock_cached_cls.from_bytes_store.call_args[0][1] is mock_store_cls.return_value


def test_make_cached_embeddings_returns_cached_embeddings_instance():
    with patch("rag.embedding_cache.OpenAIEmbeddings"), \
         patch("rag.embedding_cache.LocalFileStore"), \
         patch("rag.embedding_cache.CacheBackedEmbeddings"):
        result = make_cached_embeddings()
        assert isinstance(result, CachedEmbeddings)
