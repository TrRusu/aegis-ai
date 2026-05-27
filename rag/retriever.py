import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from app.config import OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL

CHROMA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".chroma"))

def _load_all_docs_from_chroma(vectorstore: Chroma) -> list[Document]:
    """Load all stored documents from ChromaDB as LangChain Document objects."""
    result = vectorstore._collection.get(include=["documents", "metadatas"])
    return [
        Document(page_content=content, metadata=meta)
        for content, meta in zip(result["documents"], result["metadatas"])
    ]

def build_retriever(
    k: int = 4,
    selected_docs: list[str] | None = None,
    hybrid: bool = False,
):
    embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

    search_kwargs = {"k": k}
    if selected_docs:
        search_kwargs["filter"] = {"source": {"$in": [
            os.path.join(
                os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "knowledge_base")),
                doc
            )
            for doc in selected_docs
        ]}}

    vector_retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)

    if not hybrid:
        return vector_retriever

    all_docs = _load_all_docs_from_chroma(vectorstore)

    if selected_docs:
        kb_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "knowledge_base"))
        selected_paths = {os.path.join(kb_dir, doc) for doc in selected_docs}
        all_docs = [d for d in all_docs if d.metadata.get("source") in selected_paths]

    bm25_retriever = BM25Retriever.from_documents(all_docs, k=k)

    return EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5],
    )
