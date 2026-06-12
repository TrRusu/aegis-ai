import pytest
from unittest.mock import MagicMock
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from app.config import OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL
from app.rag_chain import RagChain


@pytest.fixture
def store(tmp_path, monkeypatch):
    chroma_dir = str(tmp_path / "chroma")
    monkeypatch.setattr("rag.store.CHROMA_DIR", chroma_dir)
    embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
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
    Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=chroma_dir)
    return tmp_path


def _mock_llm():
    llm = MagicMock()
    llm.stream.return_value = iter([])
    return llm


@pytest.mark.integration
class TestRagChainIntegration:

    def test_retrieves_documents_from_chroma(self, store, monkeypatch):
        monkeypatch.setattr("rag.store.CHROMA_DIR", str(store / "chroma"))
        _, docs = RagChain(llm=_mock_llm()).run("data breach cost", [], 0.2, 1024, k=2)
        assert len(docs) > 0

    def test_k_limits_number_of_chunks(self, store, monkeypatch):
        monkeypatch.setattr("rag.store.CHROMA_DIR", str(store / "chroma"))
        _, docs = RagChain(llm=_mock_llm()).run("breach", [], 0.2, 1024, k=1)
        assert len(docs) == 1

    def test_context_is_passed_to_llm(self, store, monkeypatch):
        monkeypatch.setattr("rag.store.CHROMA_DIR", str(store / "chroma"))
        llm = _mock_llm()
        RagChain(llm=llm).run("data breach cost", [], 0.2, 1024, k=2)
        messages = llm.stream.call_args[0][0]
        system_content = messages[0].content
        assert "4.45 million" in system_content or "GDPR" in system_content

    def test_selected_docs_filter(self, store, monkeypatch):
        monkeypatch.setattr("rag.store.CHROMA_DIR", str(store / "chroma"))
        _, docs = RagChain(llm=_mock_llm()).run(
            "breach",
            [],
            0.2,
            1024,
            selected_docs=["report.pdf"],
            k=3,
        )
        sources = {doc.metadata.get("source", "") for doc in docs}
        assert all("other.pdf" not in s for s in sources)
