"""
Unit tests for DocumentStore class in rag/ingestion.py (TDD).
"""
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document


def test_get_ingested_documents_returns_sorted_filenames():
    from rag.ingestion import DocumentStore
    mock_chroma = MagicMock()
    mock_chroma._collection.get.return_value = {
        "metadatas": [
            {"source": "/kb/report.pdf"},
            {"source": "/kb/report.pdf"},
            {"source": "/kb/annual.pdf"},
        ]
    }
    store = DocumentStore(vectorstore=mock_chroma)
    result = store.get_ingested_documents()
    assert result == ["annual.pdf", "report.pdf"]


def test_get_ingested_documents_returns_empty_on_failure():
    from rag.ingestion import DocumentStore
    mock_chroma = MagicMock()
    mock_chroma._collection.get.side_effect = Exception("ChromaDB unavailable")
    store = DocumentStore(vectorstore=mock_chroma)
    assert store.get_ingested_documents() == []


def test_ingest_skips_already_ingested_file():
    from rag.ingestion import DocumentStore
    mock_chroma = MagicMock()
    mock_chroma._collection.get.return_value = {"metadatas": [{"source": "/kb/test.pdf"}]}
    store = DocumentStore(vectorstore=mock_chroma)
    result = store.ingest("test.pdf", loader=MagicMock())
    assert "already ingested" in result


def test_ingest_returns_correct_message_format():
    from rag.ingestion import DocumentStore
    mock_chroma = MagicMock()
    mock_chroma._collection.get.return_value = {"metadatas": []}
    mock_loader = MagicMock()
    mock_loader.load.return_value = [
        Document(page_content="chunk one", metadata={}),
        Document(page_content="chunk two", metadata={}),
    ]
    store = DocumentStore(vectorstore=mock_chroma)
    with patch("rag.ingestion.Chroma"), patch("rag.ingestion.OpenAIEmbeddings"):
        result = store.ingest("report.pdf", loader=mock_loader)
    assert "report.pdf" in result
    assert "2 chunks" in result


def test_ingest_stores_chunks_in_vectorstore():
    from rag.ingestion import DocumentStore
    mock_chroma = MagicMock()
    mock_chroma._collection.get.return_value = {"metadatas": []}
    mock_loader = MagicMock()
    mock_loader.load.return_value = [Document(page_content="chunk", metadata={})]
    store = DocumentStore(vectorstore=mock_chroma)
    with patch("rag.ingestion.Chroma") as mock_chroma_cls, patch("rag.ingestion.OpenAIEmbeddings"):
        store.ingest("report.pdf", loader=mock_loader)
        mock_chroma_cls.from_documents.assert_called_once()
