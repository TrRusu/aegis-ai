"""
PDF loading strategies — BasicPdfLoader and EnhancedPdfLoader.
"""
import os
import re

from langchain_community.document_loaders import UnstructuredPDFLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel

from rag.vision import VisionAnalyzer

_FIGURE_RE = re.compile(r"\bFig(?:ure)?\.?\s*\d+[A-Za-z]?", re.IGNORECASE)


class BasicPdfLoader:
    """Loads a PDF using PyPDF and splits it into chunks."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def load(self, filepath: str) -> list[Document]:
        loader = PyPDFLoader(filepath)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=self._chunk_size, chunk_overlap=self._chunk_overlap)
        return splitter.split_documents(documents)


class EnhancedPdfLoader:
    """Loads a PDF using Unstructured with vision-based chart and table extraction."""

    def __init__(self, llm: BaseChatModel, chunk_size: int = 1000, chunk_overlap: int = 150):
        self._llm = llm
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def load(self, filepath: str) -> list[Document]:
        loader = UnstructuredPDFLoader(filepath, mode="elements", strategy="hi_res")
        elements = loader.load()
        analyzer = VisionAnalyzer(llm=self._llm)
        title_chunks = self._chunk_by_title(elements, filepath, analyzer)
        splitter = RecursiveCharacterTextSplitter(chunk_size=self._chunk_size, chunk_overlap=self._chunk_overlap)
        result = []
        for chunk in title_chunks:
            if chunk.metadata.get("category") == "NarrativeText":
                result.extend(splitter.split_documents([chunk]))
            else:
                result.append(chunk)
        return result

    def _chunk_by_title(self, elements: list[Document], filepath: str, analyzer: VisionAnalyzer) -> list[Document]:
        doc_name = os.path.splitext(os.path.basename(filepath))[0]
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
                summary = analyzer.summarize_table(elem.page_content, doc_name)
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
                    img_b64 = VisionAnalyzer.render_page_png(filepath, page - 1)
                    if img_b64:
                        summary = analyzer.summarize_page(img_b64, page, doc_name)
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
                    elif VisionAnalyzer.looks_like_chart_garbage(text):
                        if page not in chart_pages_done:
                            chart_pages_done.add(page)
                            img_b64 = VisionAnalyzer.render_page_png(filepath, page - 1)
                            if img_b64:
                                summary = analyzer.summarize_page(img_b64, page, doc_name)
                                chunks.append(Document(
                                    page_content=summary,
                                    metadata={"source": source, "page": page, "category": "ImageSummary"},
                                ))
                    else:
                        current_texts.append(text)

        flush()
        return chunks
