"""
Approval tests for the retriever output structure.
"""
import json
from unittest.mock import MagicMock, patch

from approvaltests import verify

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from rag.retriever import build_retriever


def _make_mock_vectorstore(docs=None):
    if docs is None:
        docs = [
            Document(page_content="Healthcare breach cost was $9.77M", metadata={"source": "test.pdf", "page": 10}),
            Document(page_content="Average cost per record is $169", metadata={"source": "test.pdf", "page": 5}),
        ]
    mock_vs = MagicMock()
    mock_vs.as_retriever.return_value.invoke.return_value = docs
    mock_vs._collection.get.return_value = {
        "documents": [d.page_content for d in docs],
        "metadatas": [d.metadata for d in docs],
    }
    return mock_vs, docs

@patch("rag.retriever.ChromaStore")
def test_vector_retriever_document_structure(mock_chroma_store):
    """Approval: vector retriever returns documents with correct structure."""
    mock_vs, expected_docs = _make_mock_vectorstore()
    mock_chroma_store.return_value.vectorstore = mock_vs

    retriever = build_retriever(k=2, hybrid=False)
    results = retriever.invoke("healthcare breach cost")

    verify(json.dumps({
        "count": len(results),
        "fields": sorted(results[0].metadata.keys()) if results else [],
        "has_page_content": all(len(d.page_content) > 0 for d in results),
    }, indent=2))

@patch("rag.retriever.ChromaStore")
def test_hybrid_retriever_type(mock_chroma_store):
    """Approval: hybrid=True returns an EnsembleRetriever."""
    docs = [
        Document(page_content="Healthcare breach cost was $9.77M", metadata={"source": "test.pdf", "page": 10}),
        Document(page_content="Average cost per record is $169", metadata={"source": "test.pdf", "page": 5}),
    ]
    mock_vs = MagicMock()
    mock_vs._collection.get.return_value = {
        "documents": [d.page_content for d in docs],
        "metadatas": [d.metadata for d in docs],
    }
    real_vector_retriever = BM25Retriever.from_documents(docs, k=2)
    mock_vs.as_retriever.return_value = real_vector_retriever
    mock_chroma_store.return_value.vectorstore = mock_vs

    retriever = build_retriever(k=2, hybrid=True)

    verify(type(retriever).__name__)

@patch("rag.retriever.ChromaStore")
def test_retrieved_document_metadata_fields(mock_chroma_store):
    """Approval: each retrieved document has required metadata fields."""
    mock_vs, expected_docs = _make_mock_vectorstore()
    mock_chroma_store.return_value.vectorstore = mock_vs

    retriever = build_retriever(k=2, hybrid=False)
    results = retriever.invoke("test query")

    verify(json.dumps({
        "all_have_source": all("source" in d.metadata for d in results),
        "all_have_page": all("page" in d.metadata for d in results),
    }, indent=2))
