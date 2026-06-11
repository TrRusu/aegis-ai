"""
Approval tests for the ingestion pipeline.
"""
import json
from unittest.mock import MagicMock

from approvaltests import verify

from langchain_core.documents import Document
from rag.ingestion import DocumentStore
from rag.vision import VisionAnalyzer


def test_chart_garbage_detects_ocr_noise():
    """Approval: OCR noise with short average token length is flagged as garbage."""
    verify(str(VisionAnalyzer.looks_like_chart_garbage("A oO A BB A ) A w A £")))

def test_chart_garbage_passes_real_prose():
    """Approval: real prose with normal word lengths is not flagged as garbage."""
    verify(str(VisionAnalyzer.looks_like_chart_garbage(
        "Healthcare industry had the highest breach cost at USD 9.77 million in 2024."
    )))

def test_chart_garbage_ignores_short_texts():
    """Approval: texts with fewer than 4 tokens are never flagged."""
    verify(str(VisionAnalyzer.looks_like_chart_garbage("A B")))

def test_ingest_file_basic_return_message():
    """Approval: ingest basic mode return message format."""
    mock_store = MagicMock()
    mock_store.vectorstore._collection.get.return_value = {"metadatas": []}
    mock_loader = MagicMock()
    mock_loader.load.return_value = [
        Document(page_content="chunk one", metadata={"source": "test.pdf", "page": 1}),
        Document(page_content="chunk two", metadata={"source": "test.pdf", "page": 2}),
    ]
    verify(DocumentStore(store=mock_store).ingest("test.pdf", mock_loader, mode_label="basic"))

def test_ingest_file_enhanced_return_message():
    """Approval: ingest enhanced mode return message format."""
    mock_store = MagicMock()
    mock_store.vectorstore._collection.get.return_value = {"metadatas": []}
    mock_loader = MagicMock()
    mock_loader.load.return_value = [
        Document(page_content="chart summary", metadata={"source": "test.pdf", "page": 1, "category": "ImageSummary"}),
        Document(page_content="table summary", metadata={"source": "test.pdf", "page": 2, "category": "TableSummary"}),
        Document(page_content="narrative text", metadata={"source": "test.pdf", "page": 3, "category": "NarrativeText"}),
    ]
    verify(DocumentStore(store=mock_store).ingest("test.pdf", mock_loader, mode_label="enhanced"))

def test_ingest_file_already_ingested_message():
    """Approval: ingest skips a file already in ChromaDB."""
    mock_store = MagicMock()
    mock_store.vectorstore._collection.get.return_value = {
        "metadatas": [{"source": "/kb/test.pdf"}]
    }
    verify(DocumentStore(store=mock_store).ingest("test.pdf", MagicMock()))

def test_get_ingested_documents_returns_sorted():
    """Approval: get_ingested_documents returns sorted unique filenames."""
    mock_store = MagicMock()
    mock_store.vectorstore._collection.get.return_value = {
        "metadatas": [
            {"source": "/data/knowledge_base/report.pdf"},
            {"source": "/data/knowledge_base/report.pdf"},
            {"source": "/data/knowledge_base/annual.pdf"},
        ]
    }
    verify(json.dumps(DocumentStore(store=mock_store).get_ingested_documents(), indent=2))

def test_get_ingested_documents_on_error():
    """Approval: get_ingested_documents returns empty list if ChromaDB fails."""
    mock_store = MagicMock()
    mock_store.vectorstore._collection.get.side_effect = Exception("ChromaDB unavailable")
    verify(json.dumps(DocumentStore(store=mock_store).get_ingested_documents(), indent=2))
