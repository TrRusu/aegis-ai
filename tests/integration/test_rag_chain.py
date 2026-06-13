import pytest
from unittest.mock import MagicMock
from langchain_chroma import Chroma
from langchain_core.documents import Document
from app.rag_chain import RagChain
from tests.integration.conftest import FakeEmbedder


@pytest.fixture
def store(tmp_path, monkeypatch):
    chroma_dir = str(tmp_path / "chroma")
    monkeypatch.setattr("rag.store.CHROMA_DIR", chroma_dir)
    docs = [
        Document(
            page_content="The average cost of a data breach is $4.45 million.",
            metadata={"source": str(tmp_path / "report.pdf")},
        ),
        Document(
            page_content="GDPR fines for data breaches can reach 4% of global annual turnover.",
            metadata={"source": str(tmp_path / "report.pdf")},
        ),
        Document(
            page_content="Ransomware accounted for 24% of all breaches in 2023.",
            metadata={"source": str(tmp_path / "other.pdf")},
        ),
    ]
    Chroma.from_documents(documents=docs, embedding=FakeEmbedder(), persist_directory=chroma_dir)
    return tmp_path


def _mock_llm():
    llm = MagicMock()
    llm.stream.return_value = iter([])
    return llm


@pytest.mark.integration
def test_retrieves_documents_from_chroma(store, monkeypatch):
    monkeypatch.setattr("rag.store.CHROMA_DIR", str(store / "chroma"))
    _, docs = RagChain(llm=_mock_llm()).run("data breach cost", [], 0.2, 1024, k=2)
    assert len(docs) > 0


@pytest.mark.integration
def test_k_limits_number_of_chunks(store, monkeypatch):
    monkeypatch.setattr("rag.store.CHROMA_DIR", str(store / "chroma"))
    _, docs = RagChain(llm=_mock_llm()).run("breach", [], 0.2, 1024, k=1)
    assert len(docs) == 1


@pytest.mark.integration
def test_context_is_passed_to_llm(store, monkeypatch):
    monkeypatch.setattr("rag.store.CHROMA_DIR", str(store / "chroma"))
    llm = _mock_llm()
    RagChain(llm=llm).run("data breach cost", [], 0.2, 1024, k=2)
    messages = llm.stream.call_args[0][0]
    assert "4.45 million" in messages[0].content or "GDPR" in messages[0].content


@pytest.mark.integration
def test_selected_docs_filter(store, monkeypatch):
    monkeypatch.setattr("rag.store.CHROMA_DIR", str(store / "chroma"))
    _, docs = RagChain(llm=_mock_llm()).run(
        "breach", [], 0.2, 1024, selected_docs=["report.pdf"], k=3
    )
    assert all("other.pdf" not in doc.metadata.get("source", "") for doc in docs)
