import os
import re
import sys
import base64
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_chroma import Chroma
from langchain_core.documents import Document
import chromadb
from app.config import OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL, OPENAI_MODEL

_FIGURE_RE = re.compile(r"\bFig(?:ure)?\.?\s*\d+[A-Za-z]?", re.IGNORECASE)


def _looks_like_chart_garbage(text: str) -> bool:
    """Return True if the text looks like OCR noise from a chart rather than real prose."""
    tokens = text.split()
    if len(tokens) < 4:
        return False
    avg_word_len = sum(len(t) for t in tokens) / len(tokens)
    return avg_word_len < 3.5

KNOWLEDGE_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "knowledge_base"))
CHROMA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".chroma"))


def _build_llm() -> ChatOpenAI:
    return ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY, temperature=0.0, max_tokens=2048)


def _summarize_table(table_text: str, llm: ChatOpenAI, doc_name: str = "the document") -> str:
    """Ask GPT-4o to produce clean structured text from a raw extracted table."""
    prompt = (
        f"The following text was extracted from a table in {doc_name}. "
        "Rewrite it as clear, structured plain text that preserves all numbers, "
        "labels, and relationships. Do not add information that isn't present.\n\n"
        f"{table_text}"
    )
    return llm.invoke([HumanMessage(content=prompt)]).content


def _render_page_png(filepath: str, page_index: int) -> str | None:
    """Render a single PDF page to a base64 PNG using pymupdf."""
    try:
        import fitz  # pymupdf
        doc = fitz.open(filepath)
        page = doc[page_index]
        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
        return base64.b64encode(pix.tobytes("png")).decode()
    except Exception:
        return None


def _summarize_page(image_b64: str, page_number: int, llm: ChatOpenAI, doc_name: str = "the document") -> str:
    """Send a rendered page image to GPT-4o vision for chart/figure extraction."""
    message = HumanMessage(content=[
        {
            "type": "text",
            "text": (
                f"This is page {page_number} of {doc_name}. "
                "It contains a chart or figure. Extract every specific number, percentage, "
                "industry name, label, and trend visible. Write a precise description "
                "suitable for information retrieval — someone should be able to answer "
                "factual questions about this chart from your description alone."
            ),
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
        },
    ])
    return llm.invoke([message]).content


def _chunk_by_title(
    elements: list[Document],
    filepath: str,
    llm: ChatOpenAI,
) -> list[Document]:
    doc_name = os.path.splitext(os.path.basename(filepath))[0]
    """Convert a flat list of unstructured elements into title-scoped chunks."""
    chunks: list[Document] = []
    current_title = ""
    current_texts: list[str] = []
    current_page = 1
    current_source = ""
    chart_pages_done: set[int] = set()

    def flush():
        if not current_texts:
            return
        body = "\n\n".join(current_texts).strip()
        if not body:
            return
        heading = f"{current_title}\n\n" if current_title else ""
        chunks.append(Document(
            page_content=(heading + body).strip(),
            metadata={"source": current_source, "page": current_page, "category": "NarrativeText"},
        ))

    for elem in elements:
        category = elem.metadata.get("category", "")
        page = elem.metadata.get("page_number", 1)
        source = elem.metadata.get("source", "")

        if category == "Title":
            flush()
            current_title = elem.page_content.strip()
            current_texts = []
            current_page = page
            current_source = source

        elif category == "Table":
            flush()
            current_texts = []
            summary = _summarize_table(elem.page_content, llm, doc_name)
            chunks.append(Document(
                page_content=summary,
                metadata={"source": source, "page": page, "category": "TableSummary"},
            ))

        elif category == "FigureCaption" or (
            category not in ("Table", "Title")
            and len(elem.page_content) < 300
            and _FIGURE_RE.search(elem.page_content)
        ):

            if page not in chart_pages_done:
                chart_pages_done.add(page)
                img_b64 = _render_page_png(filepath, page - 1)
                if img_b64:
                    summary = _summarize_page(img_b64, page, llm, doc_name)
                    chunks.append(Document(
                        page_content=summary,
                        metadata={"source": source, "page": page, "category": "ImageSummary"},
                    ))

        else:
            
            text = elem.page_content.strip()
            if len(text) >= 20:
                alnum = sum(c.isalnum() or c.isspace() for c in text)
                if alnum / len(text) < 0.5:
                    pass
                elif _looks_like_chart_garbage(text):
                    if page not in chart_pages_done:
                        chart_pages_done.add(page)
                        img_b64 = _render_page_png(filepath, page - 1)
                        if img_b64:
                            summary = _summarize_page(img_b64, page, llm, doc_name)
                            chunks.append(Document(
                                page_content=summary,
                                metadata={"source": source, "page": page, "category": "ImageSummary"},
                            ))
                else:
                    current_texts.append(text)

    flush()
    return chunks


def _load_and_chunk_pdf(
    filepath: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[Document]:
    
    loader = UnstructuredPDFLoader(filepath, mode="elements", strategy="hi_res")
    elements = loader.load()

    llm = _build_llm()
    title_chunks = _chunk_by_title(elements, filepath, llm)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    final = []
    for chunk in title_chunks:
        if chunk.metadata.get("category") == "NarrativeText":
            final.extend(splitter.split_documents([chunk]))
        else:
            final.append(chunk)
    return final


def _load_pdf_basic(
    filepath: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[Document]:
    """Basic ingestion pipeline using PyPDF."""
    from langchain_community.document_loaders import PyPDFLoader
    loader = PyPDFLoader(filepath)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)


# ── Public API ────────────────────────────────────────────────────────────────

def get_ingested_documents() -> list[str]:
    """Return list of document filenames already in ChromaDB."""
    try:
        embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
        vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
        results = vectorstore._collection.get(include=["metadatas"])
        sources = set()
        for meta in results["metadatas"]:
            source = meta.get("source", "")
            if source:
                sources.add(os.path.basename(source))
        return sorted(sources)
    except Exception:
        return []


def ingest_file(
    filepath: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    enhanced: bool = True,
) -> str:
    """Ingest a single PDF into ChromaDB."""
    filename = os.path.basename(filepath)
    if filename in get_ingested_documents():
        return f"{filename} is already ingested."

    if enhanced:
        chunks = _load_and_chunk_pdf(filepath, chunk_size, chunk_overlap)
    else:
        chunks = _load_pdf_basic(filepath, chunk_size, chunk_overlap)

    embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
    Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=CHROMA_DIR)
    mode_label = "enhanced" if enhanced else "basic"
    return f"Ingested {filename} — {len(chunks)} chunks ({mode_label})."
