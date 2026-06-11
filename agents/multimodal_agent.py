import base64
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from app.config import OPENAI_API_KEY, OPENAI_MODEL
from observability.logging_setup import logger

_SYSTEM_PROMPT = """You are a cybersecurity incident image analyst.

You will receive a text incident description and optionally an image (screenshot of a security alert,
malware notification, network anomaly dashboard, error log, or similar).

If an image is provided:
- Analyze it carefully for any security-relevant details: error codes, IP addresses, timestamps,
  malware names, alert severities, affected systems, usernames, URLs, or anything else visible.
- Rewrite the incident description to incorporate your visual observations as a single coherent paragraph.
- Do not add a separate "Image Analysis" section — merge everything naturally.

If no image is provided, or the image does not appear to be security-related:
- Return the original incident description exactly as given, with no changes.

Your output must be a single paragraph combining the text and visual evidence."""


class MultimodalAgent:
    """Enriches incident descriptions with image analysis using an injected LLM."""

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


def enrich_with_image(incident: str, image_bytes: bytes | None, mime_type: str = "image/png") -> str:
    llm = ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY, temperature=0.0, max_tokens=1024)
    return MultimodalAgent(llm=llm).enrich(incident, image_bytes, mime_type)
