import os

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from app.config import OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL

CHROMA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".chroma"))
_KB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "knowledge_base"))


class VectorRetriever:
    """Builds a vector similarity retriever backed by ChromaDB."""

    def __init__(self, vectorstore: VectorStore):
        self._vectorstore = vectorstore

    def build(self, k: int = 4, selected_docs: list[str] | None = None):
        search_kwargs = {"k": k}
        if selected_docs:
            search_kwargs["filter"] = {"source": {"$in": [
                os.path.join(_KB_DIR, doc) for doc in selected_docs
            ]}}
        return self._vectorstore.as_retriever(search_kwargs=search_kwargs)


class HybridRetriever:
    """Builds a hybrid BM25 + vector retriever backed by ChromaDB."""

    def __init__(self, vectorstore: VectorStore):
        self._vectorstore = vectorstore

    def build(self, k: int = 4, selected_docs: list[str] | None = None):
        search_kwargs = {"k": k}
        if selected_docs:
            search_kwargs["filter"] = {"source": {"$in": [
                os.path.join(_KB_DIR, doc) for doc in selected_docs
            ]}}
        vector_retriever = self._vectorstore.as_retriever(search_kwargs=search_kwargs)

        all_docs = self._load_all_docs()
        if selected_docs:
            selected_paths = {os.path.join(_KB_DIR, doc) for doc in selected_docs}
            all_docs = [d for d in all_docs if d.metadata.get("source") in selected_paths]

        bm25_retriever = BM25Retriever.from_documents(all_docs, k=k)
        return EnsembleRetriever(retrievers=[bm25_retriever, vector_retriever], weights=[0.5, 0.5])

    def _load_all_docs(self) -> list[Document]:
        result = self._vectorstore._collection.get(include=["documents", "metadatas"])
        return [
            Document(page_content=content, metadata=meta)
            for content, meta in zip(result["documents"], result["metadatas"])
        ]


def _make_vectorstore() -> Chroma:
    embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)


def build_retriever(
    k: int = 4,
    selected_docs: list[str] | None = None,
    hybrid: bool = False,
):
    vectorstore = _make_vectorstore()
    if hybrid:
        return HybridRetriever(vectorstore=vectorstore).build(k=k, selected_docs=selected_docs)
    return VectorRetriever(vectorstore=vectorstore).build(k=k, selected_docs=selected_docs)
