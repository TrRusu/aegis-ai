"""
Unit tests for DocumentStore class in rag/ingestion.py (TDD).
"""
from unittest.mock import MagicMock

from langchain_core.documents import Document


def test_get_ingested_documents_returns_sorted_filenames():
    from rag.ingestion import DocumentStore
    mock_store = MagicMock()
    mock_store.vectorstore._collection.get.return_value = {
        "metadatas": [
            {"source": "/kb/report.pdf"},
            {"source": "/kb/report.pdf"},
            {"source": "/kb/annual.pdf"},
        ]
    }
    store = DocumentStore(store=mock_store)
    result = store.get_ingested_documents()
    assert result == ["annual.pdf", "report.pdf"]


def test_get_ingested_documents_returns_empty_on_failure():
    from rag.ingestion import DocumentStore
    mock_store = MagicMock()
    mock_store.vectorstore._collection.get.side_effect = Exception("ChromaDB unavailable")
    store = DocumentStore(store=mock_store)
    assert store.get_ingested_documents() == []


def test_ingest_skips_already_ingested_file():
    from rag.ingestion import DocumentStore
    mock_store = MagicMock()
    mock_store.vectorstore._collection.get.return_value = {"metadatas": [{"source": "/kb/test.pdf"}]}
    store = DocumentStore(store=mock_store)
    result = store.ingest("test.pdf", loader=MagicMock())
    assert "already ingested" in result


def test_ingest_returns_correct_message_format():
    from rag.ingestion import DocumentStore
    mock_store = MagicMock()
    mock_store.vectorstore._collection.get.return_value = {"metadatas": []}
    mock_loader = MagicMock()
    mock_loader.load.return_value = [
        Document(page_content="chunk one", metadata={}),
        Document(page_content="chunk two", metadata={}),
    ]
    store = DocumentStore(store=mock_store)
    result = store.ingest("report.pdf", loader=mock_loader)
    assert "report.pdf" in result
    assert "2 chunks" in result


def test_ingest_stores_chunks_in_vectorstore():
    from rag.ingestion import DocumentStore
    mock_store = MagicMock()
    mock_store.vectorstore._collection.get.return_value = {"metadatas": []}
    mock_loader = MagicMock()
    mock_loader.load.return_value = [Document(page_content="chunk", metadata={})]
    store = DocumentStore(store=mock_store)
    store.ingest("report.pdf", loader=mock_loader)
    mock_store.add_documents.assert_called_once()


def test_ingest_clears_cache_on_new_ingestion():
    from rag.ingestion import DocumentStore
    mock_store = MagicMock()
    mock_store.vectorstore._collection.get.return_value = {"metadatas": []}
    mock_cache = MagicMock()
    mock_loader = MagicMock()
    mock_loader.load.return_value = [Document(page_content="chunk", metadata={})]
    store = DocumentStore(store=mock_store, cache=mock_cache)
    store.ingest("report.pdf", loader=mock_loader)
    mock_cache.clear.assert_called_once()


def test_ingest_does_not_clear_cache_when_already_ingested():
    from rag.ingestion import DocumentStore
    mock_store = MagicMock()
    mock_store.vectorstore._collection.get.return_value = {"metadatas": [{"source": "/kb/test.pdf"}]}
    mock_cache = MagicMock()
    store = DocumentStore(store=mock_store, cache=mock_cache)
    store.ingest("test.pdf", loader=MagicMock())
    mock_cache.clear.assert_not_called()


def test_ingest_without_injected_cache_does_not_raise():
    from rag.ingestion import DocumentStore
    mock_store = MagicMock()
    mock_store.vectorstore._collection.get.return_value = {"metadatas": []}
    mock_loader = MagicMock()
    mock_loader.load.return_value = [Document(page_content="chunk", metadata={})]
    store = DocumentStore(store=mock_store)
    result = store.ingest("report.pdf", loader=mock_loader)
    assert "report.pdf" in result
