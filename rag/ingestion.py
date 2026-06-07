import os

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStore
from app.config import OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL, OPENAI_MODEL
from rag.document_loader import BasicPdfLoader, EnhancedPdfLoader

KNOWLEDGE_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "knowledge_base"))
CHROMA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".chroma"))


class DocumentStore:
    """Stores and retrieves documents in ChromaDB."""

    def __init__(self, vectorstore: VectorStore):
        self._vectorstore = vectorstore

    def get_ingested_documents(self) -> list[str]:
        try:
            results = self._vectorstore._collection.get(include=["metadatas"])
            sources = set()
            for meta in results["metadatas"]:
                source = meta.get("source", "")
                if source:
                    sources.add(os.path.basename(source))
            return sorted(sources)
        except Exception:
            return []

    def ingest(self, filepath: str, loader, mode_label: str = "") -> str:
        filename = os.path.basename(filepath)
        if filename in self.get_ingested_documents():
            return f"{filename} is already ingested."
        chunks = loader.load(filepath)
        embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
        Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=CHROMA_DIR)
        suffix = f" ({mode_label})" if mode_label else ""
        return f"Ingested {filename} — {len(chunks)} chunks{suffix}."


def _make_store() -> DocumentStore:
    embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    return DocumentStore(vectorstore=vectorstore)


def get_ingested_documents() -> list[str]:
    try:
        return _make_store().get_ingested_documents()
    except Exception:
        return []


def ingest_file(
    filepath: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    enhanced: bool = True,
) -> str:
    if enhanced:
        llm = ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY, temperature=0.0, max_tokens=2048)
        loader = EnhancedPdfLoader(llm=llm, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    else:
        loader = BasicPdfLoader(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    mode_label = "enhanced" if enhanced else "basic"
    return _make_store().ingest(filepath, loader, mode_label=mode_label)
