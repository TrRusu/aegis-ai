"""Module for building vector stores for RAG.
"""
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from app.config import CHROMA_DIR
from rag.embedding_cache import make_cached_embeddings


class ChromaStore:

    def __init__(self, embeddings: Embeddings | None = None):
        self._embeddings = embeddings or make_cached_embeddings()
        self._vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=self._embeddings)

    @property
    def vectorstore(self) -> Chroma:
        return self._vectorstore

    def add_documents(self, chunks: list) -> None:
        Chroma.from_documents(documents=chunks, embedding=self._embeddings, persist_directory=CHROMA_DIR)
