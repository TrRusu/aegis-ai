from unittest.mock import MagicMock, patch
from rag.store import ChromaStore


def test_default_uses_cached_embeddings():
    with patch("rag.store.make_cached_embeddings") as mock_factory, patch("rag.store.Chroma"):
        mock_factory.return_value = MagicMock()
        store = ChromaStore()
        mock_factory.assert_called_once()
        assert store._embeddings is mock_factory.return_value


def test_injected_embeddings_used_instead_of_cached_default():
    mock_embeddings = MagicMock()
    with patch("rag.store.make_cached_embeddings") as mock_factory, patch("rag.store.Chroma"):
        store = ChromaStore(embeddings=mock_embeddings)
        mock_factory.assert_not_called()
        assert store._embeddings is mock_embeddings


def test_vectorstore_constructed_with_embeddings():
    mock_embeddings = MagicMock()
    with patch("rag.store.make_cached_embeddings"), patch("rag.store.Chroma") as mock_chroma_cls:
        ChromaStore(embeddings=mock_embeddings)
        assert mock_chroma_cls.call_args[1]["embedding_function"] is mock_embeddings


def test_add_documents_uses_injected_embeddings():
    mock_embeddings = MagicMock()
    with patch("rag.store.make_cached_embeddings"), patch("rag.store.Chroma") as mock_chroma_cls:
        store = ChromaStore(embeddings=mock_embeddings)
        store.add_documents([MagicMock()])
        assert mock_chroma_cls.from_documents.call_args[1]["embedding"] is mock_embeddings
