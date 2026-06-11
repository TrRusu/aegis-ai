"""Enriches incident descriptions with image analysis using an injected LLM.
"""

import base64
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from observability.logging_setup import logger
from prompts.multimodal_agent import SYSTEM_PROMPT as _SYSTEM_PROMPT


class MultimodalAgent:

    def __init__(self, llm: BaseChatModel):
        self._llm = llm

    def enrich(self, incident: str, image_bytes: bytes | None, mime_type: str = "image/png") -> str:
        if not image_bytes:
            return incident

        logger.info("[Multimodal] Enriching incident description with image analysis")

        image_b64 = base64.b64encode(image_bytes).decode()

        message = HumanMessage(content=[
            {"type": "text", "text": f"Incident description:\n{incident}"},
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}},
        ])

        response = self._llm.invoke([SystemMessage(content=_SYSTEM_PROMPT), message])

        content = response.content
        if isinstance(content, list):
            content = "\n".join(
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )

        logger.info("[Multimodal] Enrichment complete")
        return content