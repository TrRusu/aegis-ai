"""Scores user input for prompt injection risk using an injected LLM.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from prompts.prompt_injection_guard import SYSTEM_PROMPT as _SYSTEM_PROMPT

INJECTION_THRESHOLD = 0.7


class PromptInjectionGuard:

    def __init__(self, llm: BaseChatModel):
        self._llm = llm

    def check(self, user_input: str) -> tuple[bool, float]:
        response = self._llm.invoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_input),
        ])

        content = response.content
        if isinstance(content, list):
            content = " ".join(
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )

        try:
            score = float(content.strip())
            score = max(0.0, min(1.0, score))
        except ValueError:
            score = 0.0

        return score >= INJECTION_THRESHOLD, score
