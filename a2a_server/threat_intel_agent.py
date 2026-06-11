"""Maps cybersecurity incidents to MITRE ATT&CK TTPs using an injected LLM.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from prompts.threat_intel_agent import SYSTEM_PROMPT


class ThreatIntelAgent:

    def __init__(self, llm: BaseChatModel):
        self._llm = llm

    def analyse(self, incident: str) -> str:
        response = self._llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=incident),
        ])
        content = response.content
        if isinstance(content, list):
            content = "\n".join(
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
        return content
