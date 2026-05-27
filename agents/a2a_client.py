"""
A2A client — calls the remote Threat Intelligence Agent over HTTP.

The client treats the remote agent exactly like a local tool: pass an incident,
get back an analysis string. The fact that it runs in a separate process on a
different port is invisible to the caller.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import httpx
from observability.logging_setup import logger

A2A_SERVER_URL = os.getenv("A2A_SERVER_URL", "http://localhost:8888")
_TIMEOUT = 30


def fetch_agent_card() -> dict:
    """Retrieve the remote agent's capabilities descriptor."""
    resp = httpx.get(f"{A2A_SERVER_URL}/.well-known/agent-card.json", timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def call_threat_intel_agent(incident: str) -> str:
    """
    Send an incident to the remote Threat Intelligence Agent and return its analysis.
    Raises httpx.HTTPError if the server is unreachable.
    """
    logger.info(f"[A2A] Calling remote ThreatIntelAgent at {A2A_SERVER_URL}/run")
    resp = httpx.post(
        f"{A2A_SERVER_URL}/run",
        json={"incident": incident},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    analysis = resp.json().get("analysis", "No analysis returned.")
    logger.info("[A2A] Remote agent responded successfully")
    return analysis


def is_server_available() -> bool:
    """Check if the remote A2A server is reachable."""
    try:
        httpx.get(f"{A2A_SERVER_URL}/.well-known/agent-card.json", timeout=3)
        return True
    except Exception:
        return False
