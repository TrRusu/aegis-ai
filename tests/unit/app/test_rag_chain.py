"""
Unit tests for RagChain class in app/rag_chain.py (TDD).
"""
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document


def _make_mock_retriever(docs=None):
    if docs is None:
        docs = [Document(page_content="Healthcare breach cost $9.77M.", metadata={})]
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = docs
    return mock_retriever


def test_rag_chain_returns_tuple():
    from app.rag_chain import RagChain
    mock_llm = MagicMock()
    mock_llm.stream.return_value = iter([MagicMock(content="Answer.")])
    with patch("app.rag_chain.build_retriever", return_value=_make_mock_retriever()):
        result = RagChain(llm=mock_llm).run("What is the cost?", [], temperature=0.5, max_tokens=512)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_rag_chain_delegates_to_injected_llm():
    from app.rag_chain import RagChain
    mock_llm = MagicMock()
    mock_llm.stream.return_value = iter([])
    with patch("app.rag_chain.build_retriever", return_value=_make_mock_retriever()):
        RagChain(llm=mock_llm).run("Query.", [], temperature=0.0, max_tokens=256)
    mock_llm.stream.assert_called_once()


def test_rag_chain_returns_docs():
    from app.rag_chain import RagChain
    docs = [Document(page_content="Doc content.", metadata={})]
    mock_llm = MagicMock()
    mock_llm.stream.return_value = iter([])
    with patch("app.rag_chain.build_retriever", return_value=_make_mock_retriever(docs)):
        _, returned_docs = RagChain(llm=mock_llm).run("Query.", [], temperature=0.0, max_tokens=256)
    assert returned_docs == docs


def test_rag_chain_returns_fallback_on_error():
    from app.rag_chain import RagChain
    from observability.fault_tolerance import FALLBACK_MESSAGE
    mock_llm = MagicMock()
    mock_llm.stream.side_effect = Exception("LLM error")
    with patch("app.rag_chain.build_retriever", return_value=_make_mock_retriever()):
        stream, docs = RagChain(llm=mock_llm).run("Query.", [], temperature=0.0, max_tokens=256)
    assert next(stream).content == FALLBACK_MESSAGE
    assert docs == []


def test_rag_chain_includes_history_in_messages():
    from app.rag_chain import RagChain
    mock_llm = MagicMock()
    mock_llm.stream.return_value = iter([])
    history = [
        {"role": "user", "content": "Previous question."},
        {"role": "assistant", "content": "Previous answer."},
        {"role": "user", "content": "Current question."},
    ]
    with patch("app.rag_chain.build_retriever", return_value=_make_mock_retriever()):
        RagChain(llm=mock_llm).run("Current question.", history, temperature=0.0, max_tokens=256)
    messages = mock_llm.stream.call_args[0][0]
    contents = [m.content for m in messages if hasattr(m, "content") and isinstance(m.content, str)]
    assert any("Previous question." in c for c in contents)
