"""Module for building vector stores for RAG.
"""
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from app.config import OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL, CHROMA_DIR


class ChromaStore:

    def __init__(self):
        self._embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
        self._vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=self._embeddings)

    @property
    def vectorstore(self) -> Chroma:
        return self._vectorstore

    def add_documents(self, chunks: list) -> None:
        Chroma.from_documents(documents=chunks, embedding=self._embeddings, persist_directory=CHROMA_DIR)
