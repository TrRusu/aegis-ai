"""
Threat Intelligence A2A Server — runs as a standalone service on port 8888.

Implements a simplified Agent-to-Agent protocol:
  GET  /.well-known/agent-card.json  →  describes this agent's capabilities
  POST /run                           →  accepts an incident, returns MITRE ATT&CK analysis

Start with:
    python a2a_server/threat_intel_server.py
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from app.config import OPENAI_API_KEY, OPENAI_MODEL

app = FastAPI(title="Aegis Threat Intelligence Agent", version="1.0.0")

# ── Agent Card ─────────────────────────────────────────────────────────────────
# Published at /.well-known/agent-card.json so clients can discover capabilities.
# Equivalent to the tutorial's @PublicAgentCard / AgentCard builder.

AGENT_CARD = {
    "name": "Threat Intelligence Agent",
    "description": "Maps cybersecurity incidents to MITRE ATT&CK tactics, techniques and procedures. "
                   "Returns structured threat intelligence analysis for a given incident description.",
    "version": "1.0.0",
    "protocol_version": "1.0.0",
    "url": "http://localhost:8888",
    "capabilities": {
        "streaming": False,
        "push_notifications": False,
    },
    "skills": [
        {
            "id": "mitre-mapping",
            "name": "MITRE ATT&CK Mapping",
            "description": "Maps an incident to relevant MITRE ATT&CK tactics, techniques and sub-techniques.",
            "tags": ["threat-intelligence", "mitre", "attack", "ttp"],
        }
    ],
    "input_modes": ["text"],
    "output_modes": ["text"],
    "transport": "http+json",
    "endpoint": "http://localhost:8888/run",
}


@app.get("/.well-known/agent-card.json")
def agent_card():
    return AGENT_CARD


# ── Task endpoint ──────────────────────────────────────────────────────────────
# Equivalent to the tutorial's AgentExecutor.execute().
# Accepts an incident description, runs the ThreatIntelligenceAgent, returns analysis.

class RunRequest(BaseModel):
    incident: str


class RunResponse(BaseModel):
    analysis: str


_SYSTEM_PROMPT = """You are a threat intelligence analyst specialising in the MITRE ATT&CK framework.

Given a cybersecurity incident description, produce a structured threat intelligence briefing:

- **Likely Threat Actor Profile**: nation-state / cybercriminal / insider / unknown
- **MITRE ATT&CK Tactics** (in order of likely execution): list each tactic with its ID (e.g. TA0001 Initial Access)
- **Key Techniques**: for each tactic, list the most likely technique(s) with IDs (e.g. T1566.001 Spearphishing Attachment)
- **Indicators of Compromise (IOCs)**: what to look for in logs and network traffic
- **Threat Intelligence Summary**: 2-3 sentences on the overall threat profile

Be specific and use official MITRE ATT&CK terminology. If information is insufficient for a confident mapping, say so."""


@app.post("/run", response_model=RunResponse)
def run(request: RunRequest) -> RunResponse:
    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        api_key=OPENAI_API_KEY,
        temperature=0.0,
        max_tokens=1024,
    )
    response = llm.invoke([
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=request.incident),
    ])
    content = response.content
    if isinstance(content, list):
        content = "\n".join(
            b.get("text", "") for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return RunResponse(analysis=content)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)
