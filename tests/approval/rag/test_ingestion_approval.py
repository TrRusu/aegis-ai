"""
Approval tests for the ingestion pipeline.
"""
import json
from unittest.mock import MagicMock, patch

from approvaltests import verify

from langchain_core.documents import Document
from rag.ingestion import ingest_file, get_ingested_documents
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

@patch("rag.ingestion.Chroma")
@patch("rag.ingestion.OpenAIEmbeddings")
@patch("rag.ingestion.get_ingested_documents", return_value=[])
@patch("rag.ingestion.BasicPdfLoader")
def test_ingest_file_basic_return_message(mock_loader_cls, mock_get, mock_emb, mock_chroma):
    """Approval: ingest_file basic mode return message format."""
    mock_loader_cls.return_value.load.return_value = [
        Document(page_content="chunk one", metadata={"source": "test.pdf", "page": 1}),
        Document(page_content="chunk two", metadata={"source": "test.pdf", "page": 2}),
    ]
    mock_chroma.from_documents.return_value = MagicMock()
    verify(ingest_file("test.pdf", enhanced=False))

@patch("rag.ingestion.Chroma")
@patch("rag.ingestion.OpenAIEmbeddings")
@patch("rag.ingestion.ChatOpenAI")
@patch("rag.ingestion.get_ingested_documents", return_value=[])
@patch("rag.ingestion.EnhancedPdfLoader")
def test_ingest_file_enhanced_return_message(mock_loader_cls, mock_get, mock_llm, mock_emb, mock_chroma):
    """Approval: ingest_file enhanced mode return message format."""
    mock_loader_cls.return_value.load.return_value = [
        Document(page_content="chart summary", metadata={"source": "test.pdf", "page": 1, "category": "ImageSummary"}),
        Document(page_content="table summary", metadata={"source": "test.pdf", "page": 2, "category": "TableSummary"}),
        Document(page_content="narrative text", metadata={"source": "test.pdf", "page": 3, "category": "NarrativeText"}),
    ]
    mock_chroma.from_documents.return_value = MagicMock()
    verify(ingest_file("test.pdf", enhanced=True))

@patch("rag.ingestion.Chroma")
@patch("rag.ingestion.OpenAIEmbeddings")
@patch("rag.ingestion.get_ingested_documents", return_value=["test.pdf"])
def test_ingest_file_already_ingested_message(mock_get, mock_emb, mock_chroma):
    """Approval: ingest_file skips a file already in ChromaDB."""
    verify(ingest_file("test.pdf", enhanced=False))

@patch("rag.ingestion.Chroma")
@patch("rag.ingestion.OpenAIEmbeddings")
def test_get_ingested_documents_returns_sorted(mock_emb, mock_chroma):
    """Approval: get_ingested_documents returns sorted unique filenames."""
    mock_vs = MagicMock()
    mock_vs._collection.get.return_value = {
        "metadatas": [
            {"source": "/data/knowledge_base/report.pdf"},
            {"source": "/data/knowledge_base/report.pdf"},
            {"source": "/data/knowledge_base/annual.pdf"},
        ]
    }
    mock_chroma.return_value = mock_vs

    verify(json.dumps(get_ingested_documents(), indent=2))

@patch("rag.ingestion.Chroma")
@patch("rag.ingestion.OpenAIEmbeddings")
def test_get_ingested_documents_on_error(mock_emb, mock_chroma):
    """Approval: get_ingested_documents returns empty list if ChromaDB fails."""
    mock_chroma.side_effect = Exception("ChromaDB unavailable")

    verify(json.dumps(get_ingested_documents(), indent=2))