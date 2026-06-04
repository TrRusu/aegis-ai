"""
Approval tests for app/rag_chain.py.
Captures the current behavior before any refactoring.
"""
import json
from unittest.mock import MagicMock, patch

from approvaltests import verify
from langchain_core.documents import Document

from observability.fault_tolerance import FALLBACK_MESSAGE
from app.rag_chain import build_rag_response


def _make_mock_docs():
    return [
        Document(page_content="Healthcare breach cost was $9.77M", metadata={"source": "test.pdf", "page": 10}),
        Document(page_content="Average cost per record is $169", metadata={"source": "test.pdf", "page": 5}),
    ]


def _make_mock_chunk(text):
    chunk = MagicMock()
    chunk.content = text
    return chunk


@patch("app.rag_chain.ChatOpenAI")
@patch("app.rag_chain.build_retriever")
def test_build_rag_response_returns_stream_and_docs(mock_retriever, mock_llm):
    """Approval: build_rag_response returns a (stream, list[Document]) tuple."""
    docs = _make_mock_docs()
    mock_retriever.return_value.invoke.return_value = docs
    mock_llm.return_value.stream.return_value = iter([_make_mock_chunk("RAG answer.")])

    stream, source_docs = build_rag_response(
        user_input="What is the healthcare breach cost?",
        history=[],
        temperature=0.2,
        max_tokens=1024,
    )

    verify(json.dumps({
        "source_docs_count": len(source_docs),
        "source_docs_type": type(source_docs[0]).__name__,
        "stream_is_iterable": hasattr(stream, "__iter__"),
    }, indent=2))


@patch("app.rag_chain.ChatOpenAI")
@patch("app.rag_chain.build_retriever")
def test_build_rag_response_returns_correct_docs(mock_retriever, mock_llm):
    """Approval: source docs returned have correct metadata fields."""
    docs = _make_mock_docs()
    mock_retriever.return_value.invoke.return_value = docs
    mock_llm.return_value.stream.return_value = iter([_make_mock_chunk("answer")])

    _, source_docs = build_rag_response(
        user_input="test",
        history=[],
        temperature=0.2,
        max_tokens=1024,
    )

    verify(json.dumps({
        "all_have_source": all("source" in d.metadata for d in source_docs),
        "all_have_page": all("page" in d.metadata for d in source_docs),
    }, indent=2))


@patch("app.rag_chain.build_retriever")
def test_build_rag_response_failure_returns_fallback_stream(mock_retriever):
    """Approval: on failure, build_rag_response returns a stream with fallback message."""
    mock_retriever.side_effect = Exception("Retriever failed")

    stream, source_docs = build_rag_response(
        user_input="test",
        history=[],
        temperature=0.2,
        max_tokens=1024,
    )

    chunks = list(stream)
    verify(json.dumps({
        "source_docs": source_docs,
        "chunk_content": chunks[0].content,
    }, indent=2))
