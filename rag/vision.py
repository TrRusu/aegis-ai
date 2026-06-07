import base64
import re

import fitz
from langchain_core.messages import HumanMessage
from langchain_core.language_models import BaseChatModel

_FIGURE_RE = re.compile(r"\bFig(?:ure)?\.?\s*\d+[A-Za-z]?", re.IGNORECASE)


class VisionAnalyzer:
    """Handles all vision and LLM-based content extraction from PDF pages."""

    def __init__(self, llm: BaseChatModel):
        self._llm = llm

    def summarize_table(self, table_text: str, doc_name: str = "the document") -> str:
        prompt = (
            f"The following text was extracted from a table in {doc_name}. "
            "Rewrite it as clear, structured plain text that preserves all numbers, "
            "labels, and relationships. Do not add information that isn't present.\n\n"
            f"{table_text}"
        )
        return self._llm.invoke([HumanMessage(content=prompt)]).content

    def summarize_page(self, image_b64: str, page_number: int, doc_name: str = "the document") -> str:
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
        return self._llm.invoke([message]).content

    @staticmethod
    def looks_like_chart_garbage(text: str) -> bool:
        tokens = text.split()
        if len(tokens) < 4:
            return False
        avg_word_len = sum(len(t) for t in tokens) / len(tokens)
        return avg_word_len < 3.5

    @staticmethod
    def render_page_png(filepath: str, page_index: int) -> str | None:
        try:
            doc = fitz.open(filepath)
            page = doc[page_index]
            pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
            return base64.b64encode(pix.tobytes("png")).decode()
        except Exception:
            return None
