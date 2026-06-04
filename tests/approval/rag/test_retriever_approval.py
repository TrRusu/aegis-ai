"""
Approval tests for the retriever output structure.
These tests capture the CURRENT behavior before any refactoring.
If refactoring breaks the retriever interface, these tests will fail.
"""
from unittest.mock import MagicMock, patch

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from rag.retriever import build_retriever


def _make_mock_vectorstore(docs=None):
    """Create a mock ChromaDB vectorstore returning fake documents."""
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


@patch("rag.retriever.Chroma")
@patch("rag.retriever.OpenAIEmbeddings")
def test_vector_retriever_returns_documents(mock_embeddings, mock_chroma):
    """Approval: vector retriever returns a list of Document objects."""
    mock_vs, expected_docs = _make_mock_vectorstore()
    mock_chroma.return_value = mock_vs

    retriever = build_retriever(k=2, hybrid=False)
    results = retriever.invoke("healthcare breach cost")

    assert isinstance(results, list)
    assert len(results) == 2
    assert all(isinstance(doc, Document) for doc in results)
    assert all(hasattr(doc, "page_content") for doc in results)
    assert all(hasattr(doc, "metadata") for doc in results)


@patch("rag.retriever.Chroma")
@patch("rag.retriever.OpenAIEmbeddings")
def test_hybrid_retriever_returns_ensemble(mock_embeddings, mock_chroma):
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
    mock_chroma.return_value = mock_vs

    retriever = build_retriever(k=2, hybrid=True)

    assert isinstance(retriever, EnsembleRetriever)


@patch("rag.retriever.Chroma")
@patch("rag.retriever.OpenAIEmbeddings")
def test_document_structure_has_required_fields(mock_embeddings, mock_chroma):
    """Approval: each retrieved document has page_content and metadata with source."""
    mock_vs, expected_docs = _make_mock_vectorstore()
    mock_chroma.return_value = mock_vs

    retriever = build_retriever(k=2, hybrid=False)
    results = retriever.invoke("test query")

    for doc in results:
        assert "source" in doc.metadata
        assert len(doc.page_content) > 0
