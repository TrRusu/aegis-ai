"""
Approval tests for the ingestion pipeline.
These tests capture the CURRENT behavior before any refactoring.
If refactoring breaks the ingestion interface, these tests will fail.
"""
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from rag.ingestion import _looks_like_chart_garbage, ingest_file, get_ingested_documents


# ── _looks_like_chart_garbage ──────────────────────────────────────────────────

def test_chart_garbage_detects_short_tokens():
    """Approval: OCR noise with average token length < 3.5 is flagged as garbage."""
    assert _looks_like_chart_garbage("A oO A BB A ) A w A £") is True


def test_chart_garbage_passes_real_prose():
    """Approval: real prose with normal word lengths is not flagged as garbage."""
    assert _looks_like_chart_garbage(
        "Healthcare industry had the highest breach cost at USD 9.77 million in 2024."
    ) is False


def test_chart_garbage_ignores_short_texts():
    """Approval: texts with fewer than 4 tokens are never flagged."""
    assert _looks_like_chart_garbage("A B") is False


# ── ingest_file ────────────────────────────────────────────────────────────────

@patch("rag.ingestion.Chroma")
@patch("rag.ingestion.OpenAIEmbeddings")
@patch("rag.ingestion.get_ingested_documents", return_value=[])
@patch("rag.ingestion._load_pdf_basic")
def test_ingest_file_basic_returns_success_message(mock_load, mock_get, mock_emb, mock_chroma):
    """Approval: ingest_file returns a message with filename and chunk count."""
    mock_load.return_value = [
        Document(page_content="chunk one", metadata={"source": "test.pdf", "page": 1}),
        Document(page_content="chunk two", metadata={"source": "test.pdf", "page": 2}),
    ]
    mock_chroma.from_documents.return_value = MagicMock()

    result = ingest_file("test.pdf", enhanced=False)

    assert "test.pdf" in result
    assert "2 chunks" in result
    assert "basic" in result


@patch("rag.ingestion.Chroma")
@patch("rag.ingestion.OpenAIEmbeddings")
@patch("rag.ingestion.get_ingested_documents", return_value=["test.pdf"])
def test_ingest_file_skips_already_ingested(mock_get, mock_emb, mock_chroma):
    """Approval: ingest_file skips a file that is already in ChromaDB."""
    result = ingest_file("test.pdf", enhanced=False)

    assert "already ingested" in result
    mock_chroma.from_documents.assert_not_called()


@patch("rag.ingestion.Chroma")
@patch("rag.ingestion.OpenAIEmbeddings")
@patch("rag.ingestion.get_ingested_documents", return_value=[])
@patch("rag.ingestion._load_and_chunk_pdf")
def test_ingest_file_enhanced_returns_success_message(mock_load, mock_get, mock_emb, mock_chroma):
    """Approval: ingest_file with enhanced=True returns message labelled 'enhanced'."""
    mock_load.return_value = [
        Document(page_content="chart summary", metadata={"source": "test.pdf", "page": 1, "category": "ImageSummary"}),
        Document(page_content="table summary", metadata={"source": "test.pdf", "page": 2, "category": "TableSummary"}),
        Document(page_content="narrative text", metadata={"source": "test.pdf", "page": 3, "category": "NarrativeText"}),
    ]
    mock_chroma.from_documents.return_value = MagicMock()

    result = ingest_file("test.pdf", enhanced=True)

    assert "test.pdf" in result
    assert "3 chunks" in result
    assert "enhanced" in result


# ── get_ingested_documents ─────────────────────────────────────────────────────

@patch("rag.ingestion.Chroma")
@patch("rag.ingestion.OpenAIEmbeddings")
def test_get_ingested_documents_returns_sorted_filenames(mock_emb, mock_chroma):
    """Approval: get_ingested_documents returns sorted list of unique filenames."""
    mock_vs = MagicMock()
    mock_vs._collection.get.return_value = {
        "metadatas": [
            {"source": "/data/knowledge_base/report.pdf"},
            {"source": "/data/knowledge_base/report.pdf"},
            {"source": "/data/knowledge_base/annual.pdf"},
        ]
    }
    mock_chroma.return_value = mock_vs

    result = get_ingested_documents()

    assert result == ["annual.pdf", "report.pdf"]


@patch("rag.ingestion.Chroma")
@patch("rag.ingestion.OpenAIEmbeddings")
def test_get_ingested_documents_returns_empty_on_error(mock_emb, mock_chroma):
    """Approval: get_ingested_documents returns empty list if ChromaDB fails."""
    mock_chroma.side_effect = Exception("ChromaDB unavailable")

    result = get_ingested_documents()

    assert result == []
