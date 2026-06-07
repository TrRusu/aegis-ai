"""
Unit tests for rag/document_loader.py — BasicPdfLoader and EnhancedPdfLoader classes (TDD).
"""
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document


def test_basic_loader_returns_list_of_documents():
    from rag.document_loader import BasicPdfLoader
    mock_doc = Document(page_content="Some text.", metadata={"source": "file.pdf", "page": 1})
    with patch("rag.document_loader.PyPDFLoader") as mock_loader_cls, \
         patch("rag.document_loader.RecursiveCharacterTextSplitter") as mock_splitter_cls:
        mock_loader_cls.return_value.load.return_value = [mock_doc]
        mock_splitter_cls.return_value.split_documents.return_value = [mock_doc]
        result = BasicPdfLoader().load("report.pdf")
    assert isinstance(result, list)
    assert all(isinstance(d, Document) for d in result)


def test_basic_loader_uses_chunk_size_and_overlap():
    from rag.document_loader import BasicPdfLoader
    with patch("rag.document_loader.PyPDFLoader") as mock_loader_cls, \
         patch("rag.document_loader.RecursiveCharacterTextSplitter") as mock_splitter_cls:
        mock_loader_cls.return_value.load.return_value = []
        mock_splitter_cls.return_value.split_documents.return_value = []
        BasicPdfLoader(chunk_size=500, chunk_overlap=50).load("report.pdf")
        mock_splitter_cls.assert_called_once_with(chunk_size=500, chunk_overlap=50)


def test_enhanced_loader_returns_list_of_documents():
    from rag.document_loader import EnhancedPdfLoader
    mock_llm = MagicMock()
    mock_doc = Document(page_content="Narrative text.", metadata={"source": "file.pdf", "page": 1, "category": "NarrativeText"})
    with patch("rag.document_loader.UnstructuredPDFLoader") as mock_loader_cls, \
         patch("rag.document_loader.RecursiveCharacterTextSplitter") as mock_splitter_cls:
        mock_loader_cls.return_value.load.return_value = [mock_doc]
        mock_splitter_cls.return_value.split_documents.return_value = [mock_doc]
        result = EnhancedPdfLoader(llm=mock_llm).load("report.pdf")
    assert isinstance(result, list)
    assert all(isinstance(d, Document) for d in result)


def test_enhanced_loader_splits_narrative_chunks():
    from rag.document_loader import EnhancedPdfLoader
    mock_llm = MagicMock()
    narrative = Document(page_content="This is a long narrative text about data breaches.", metadata={"category": "NarrativeText", "source": "f.pdf", "page": 1})
    split_result = [Document(page_content="chunk1", metadata={}), Document(page_content="chunk2", metadata={})]
    with patch("rag.document_loader.UnstructuredPDFLoader") as mock_loader_cls, \
         patch("rag.document_loader.RecursiveCharacterTextSplitter") as mock_splitter_cls:
        mock_loader_cls.return_value.load.return_value = [narrative]
        mock_splitter_cls.return_value.split_documents.return_value = split_result
        result = EnhancedPdfLoader(llm=mock_llm).load("report.pdf")
    assert len(result) == 2


def test_enhanced_loader_does_not_split_non_narrative_chunks():
    from rag.document_loader import EnhancedPdfLoader
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Table summary.")
    table_doc = Document(page_content="Col1 Col2", metadata={"category": "Table", "source": "f.pdf", "page": 1})
    with patch("rag.document_loader.UnstructuredPDFLoader") as mock_loader_cls, \
         patch("rag.document_loader.RecursiveCharacterTextSplitter"):
        mock_loader_cls.return_value.load.return_value = [table_doc]
        result = EnhancedPdfLoader(llm=mock_llm).load("report.pdf")
    assert len(result) == 1
