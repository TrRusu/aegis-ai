"""
Approval tests for the A2A client.
"""
import json
from unittest.mock import MagicMock, patch

from approvaltests import verify

from agents.a2a_client import A2AClient


def _make_post_response(analysis="Threat intel analysis result."):
    mock = MagicMock()
    mock.json.return_value = {"analysis": analysis}
    mock.raise_for_status = MagicMock()
    return mock


@patch("agents.a2a_client.httpx.post")
def test_call_threat_intel_agent_return_type(mock_post):
    """Approval: call_threat_intel_agent returns a non-empty string."""
    mock_post.return_value = _make_post_response()

    result = A2AClient(base_url="http://test").call_threat_intel_agent("Ransomware attack on healthcare systems.")

    verify(json.dumps({
        "return_type": type(result).__name__,
        "is_non_empty": len(result) > 0,
    }, indent=2))

@patch("agents.a2a_client.httpx.post")
def test_call_threat_intel_agent_returns_analysis_field(mock_post):
    """Approval: call_threat_intel_agent returns the analysis field from the JSON response."""
    mock_post.return_value = _make_post_response(analysis="MITRE ATT&CK: T1566.001 Spearphishing.")

    result = A2AClient(base_url="http://test").call_threat_intel_agent("Phishing attack detected.")

    verify(result)

@patch("agents.a2a_client.httpx.post")
def test_call_threat_intel_agent_missing_analysis_key(mock_post):
    """Approval: call_threat_intel_agent falls back when analysis key is absent."""
    mock = MagicMock()
    mock.json.return_value = {}
    mock.raise_for_status = MagicMock()
    mock_post.return_value = mock

    result = A2AClient(base_url="http://test").call_threat_intel_agent("test incident")

    verify(result)

@patch("agents.a2a_client.httpx.get")
def test_is_server_available_returns_true_when_reachable(mock_get):
    """Approval: is_server_available returns True when server responds."""
    mock_get.return_value = MagicMock()

    result = A2AClient(base_url="http://test").is_server_available()

    verify(json.dumps({"available": result}, indent=2))

@patch("agents.a2a_client.httpx.get")
def test_is_server_available_returns_false_when_unreachable(mock_get):
    """Approval: is_server_available returns False when server is unreachable."""
    mock_get.side_effect = Exception("Connection refused")

    result = A2AClient(base_url="http://test").is_server_available()

    verify(json.dumps({"available": result}, indent=2))

@patch("agents.a2a_client.httpx.get")
def test_fetch_agent_card_return_type(mock_get):
    """Approval: fetch_agent_card returns a dict."""
    mock_get.return_value.json.return_value = {"name": "Threat Intelligence Agent", "version": "1.0.0"}
    mock_get.return_value.raise_for_status = MagicMock()

    result = A2AClient(base_url="http://test").fetch_agent_card()

    verify(json.dumps({"return_type": type(result).__name__}, indent=2))

@patch("agents.a2a_client.httpx.get")
def test_fetch_agent_card_returns_response_json(mock_get):
    """Approval: fetch_agent_card returns the full JSON from the server response."""
    card = {"name": "Threat Intelligence Agent", "version": "1.0.0"}
    mock_get.return_value.json.return_value = card
    mock_get.return_value.raise_for_status = MagicMock()

    result = A2AClient(base_url="http://test").fetch_agent_card()

    verify(json.dumps(result, indent=2))
