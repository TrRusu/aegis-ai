import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import httpx
from app.config import A2A_SERVER_URL, A2A_TIMEOUT
from observability.logging_setup import logger

class A2AClient:
    """Calls the remote Threat Intelligence Agent over HTTP."""

    def __init__(self, base_url: str):
        self._base_url = base_url

    def fetch_agent_card(self) -> dict:
        resp = httpx.get(f"{self._base_url}/.well-known/agent-card.json", timeout=A2A_TIMEOUT)
        resp.raise_for_status()
        return resp.json()

    def call_threat_intel_agent(self, incident: str) -> str:
        logger.info(f"[A2A] Calling remote ThreatIntelAgent at {self._base_url}/run")
        resp = httpx.post(
            f"{self._base_url}/run",
            json={"incident": incident},
            timeout=A2A_TIMEOUT,
        )
        resp.raise_for_status()
        analysis = resp.json().get("analysis", "No analysis returned.")
        logger.info("[A2A] Remote agent responded successfully")
        return analysis

    def is_server_available(self) -> bool:
        try:
            httpx.get(f"{self._base_url}/.well-known/agent-card.json", timeout=3)
            return True
        except Exception:
            return False