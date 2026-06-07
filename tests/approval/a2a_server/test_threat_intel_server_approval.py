"""
Approval tests for the A2A threat intel server — /run endpoint via FastAPI TestClient.
"""
import json
from unittest.mock import MagicMock, patch

from approvaltests import verify
from fastapi.testclient import TestClient

from a2a_server.threat_intel_server import app

client = TestClient(app)


def test_agent_card_endpoint_returns_required_keys():

    response = client.get("/.well-known/agent-card.json")

    card = response.json()
    verify(json.dumps({
        "status_code": response.status_code,
        "has_name": "name" in card,
        "has_version": "version" in card,
        "has_skills": "skills" in card,
        "has_capabilities": "capabilities" in card,
    }, indent=2))


def test_agent_card_endpoint_name():
    """Approval: agent card returns the correct agent name."""
    response = client.get("/.well-known/agent-card.json")

    verify(response.json()["name"])


@patch("a2a_server.threat_intel_server.ChatOpenAI")
def test_run_endpoint_returns_analysis(mock_llm):
    """Approval: POST /run returns a JSON response with a non-empty analysis string."""
    mock_llm.return_value.invoke.return_value = MagicMock(
        content="MITRE ATT&CK mapping: T1566 Phishing."
    )

    response = client.post("/run", json={"incident": "Phishing attack detected."})

    verify(json.dumps({
        "status_code": response.status_code,
        "has_analysis_key": "analysis" in response.json(),
        "analysis_type": type(response.json()["analysis"]).__name__,
        "analysis_non_empty": len(response.json()["analysis"]) > 0,
    }, indent=2))


@patch("a2a_server.threat_intel_server.ChatOpenAI")
def test_run_endpoint_returns_llm_content(mock_llm):
    """Approval: POST /run returns the LLM response as the analysis value."""
    mock_llm.return_value.invoke.return_value = MagicMock(
        content="MITRE ATT&CK mapping: T1566 Phishing."
    )

    response = client.post("/run", json={"incident": "Phishing attack detected."})

    verify(response.json()["analysis"])
