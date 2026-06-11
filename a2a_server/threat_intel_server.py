"""
Threat Intelligence A2A Server — runs as a standalone service on port 8888.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from app.config import OPENAI_API_KEY, OPENAI_MODEL
from a2a_server.threat_intel_agent import ThreatIntelAgent

app = FastAPI(title="Aegis Threat Intelligence Agent", version="1.0.0")

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


class RunRequest(BaseModel):
    incident: str


class RunResponse(BaseModel):
    analysis: str


@app.get("/.well-known/agent-card.json")
def agent_card():
    return AGENT_CARD


@app.post("/run", response_model=RunResponse)
def run(request: RunRequest) -> RunResponse:
    _llm = ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY, temperature=0.0, max_tokens=1024)
    return RunResponse(analysis=ThreatIntelAgent(llm=_llm).analyse(request.incident))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)
