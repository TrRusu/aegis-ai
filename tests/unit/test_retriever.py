"""
Unit tests for VectorRetriever and HybridRetriever classes in rag/retriever.py (TDD).
"""
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document


def test_vector_retriever_build_returns_retriever():
    from rag.retriever import VectorRetriever
    mock_vectorstore = MagicMock()
    mock_vectorstore.as_retriever.return_value = MagicMock()
    retriever = VectorRetriever(vectorstore=mock_vectorstore)
    result = retriever.build(k=4)
    mock_vectorstore.as_retriever.assert_called_once()
    assert result is not None


def test_vector_retriever_applies_selected_docs_filter():
    from rag.retriever import VectorRetriever
    mock_vectorstore = MagicMock()
    mock_vectorstore.as_retriever.return_value = MagicMock()
    retriever = VectorRetriever(vectorstore=mock_vectorstore)
    retriever.build(k=4, selected_docs=["report.pdf"])
    call_kwargs = mock_vectorstore.as_retriever.call_args[1]
    assert "filter" in call_kwargs["search_kwargs"]


def test_vector_retriever_no_filter_when_no_selected_docs():
    from rag.retriever import VectorRetriever
    mock_vectorstore = MagicMock()
    mock_vectorstore.as_retriever.return_value = MagicMock()
    retriever = VectorRetriever(vectorstore=mock_vectorstore)
    retriever.build(k=4, selected_docs=None)
    call_kwargs = mock_vectorstore.as_retriever.call_args[1]
    assert "filter" not in call_kwargs["search_kwargs"]


def test_hybrid_retriever_build_returns_ensemble_retriever():
    from rag.retriever import HybridRetriever
    from langchain_classic.retrievers import EnsembleRetriever
    mock_vectorstore = MagicMock()
    mock_vectorstore.as_retriever.return_value = MagicMock()
    mock_vectorstore._collection.get.return_value = {
        "documents": ["some text"],
        "metadatas": [{"source": "/kb/report.pdf"}],
    }
    retriever = HybridRetriever(vectorstore=mock_vectorstore)
    with patch("rag.retriever.BM25Retriever") as mock_bm25:
        mock_bm25.from_documents.return_value = MagicMock()
        result = retriever.build(k=4)
    assert isinstance(result, EnsembleRetriever)


def test_hybrid_retriever_filters_docs_by_selected():
    from rag.retriever import HybridRetriever
    import os
    kb_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "knowledge_base"))
    mock_vectorstore = MagicMock()
    mock_vectorstore.as_retriever.return_value = MagicMock()
    mock_vectorstore._collection.get.return_value = {
        "documents": ["text1", "text2"],
        "metadatas": [
            {"source": os.path.join(kb_dir, "report.pdf")},
            {"source": os.path.join(kb_dir, "other.pdf")},
        ],
    }
    retriever = HybridRetriever(vectorstore=mock_vectorstore)
    with patch("rag.retriever.BM25Retriever") as mock_bm25:
        mock_bm25.from_documents.return_value = MagicMock()
        retriever.build(k=4, selected_docs=["report.pdf"])
        docs_passed = mock_bm25.from_documents.call_args[0][0]
    assert len(docs_passed) == 1
