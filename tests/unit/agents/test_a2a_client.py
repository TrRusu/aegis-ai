"""
Unit tests for A2AClient class in agents/a2a_client.py (TDD).
"""
from unittest.mock import MagicMock, patch


def test_a2a_client_call_threat_intel_agent_returns_analysis():
    from agents.a2a_client import A2AClient
    mock_response = MagicMock()
    mock_response.json.return_value = {"analysis": "Threat identified."}
    with patch("agents.a2a_client.httpx.post", return_value=mock_response):
        client = A2AClient(base_url="http://localhost:8888")
        result = client.call_threat_intel_agent("Ransomware on hospital network.")
    assert result == "Threat identified."


def test_a2a_client_call_returns_fallback_when_analysis_missing():
    from agents.a2a_client import A2AClient
    mock_response = MagicMock()
    mock_response.json.return_value = {}
    with patch("agents.a2a_client.httpx.post", return_value=mock_response):
        client = A2AClient(base_url="http://localhost:8888")
        result = client.call_threat_intel_agent("Incident.")
    assert result == "No analysis returned."


def test_a2a_client_is_server_available_returns_true_on_success():
    from agents.a2a_client import A2AClient
    with patch("agents.a2a_client.httpx.get", return_value=MagicMock()):
        client = A2AClient(base_url="http://localhost:8888")
        assert client.is_server_available() is True


def test_a2a_client_is_server_available_returns_false_on_exception():
    from agents.a2a_client import A2AClient
    with patch("agents.a2a_client.httpx.get", side_effect=Exception("Connection refused")):
        client = A2AClient(base_url="http://localhost:8888")
        assert client.is_server_available() is False


def test_a2a_client_fetch_agent_card_returns_dict():
    from agents.a2a_client import A2AClient
    mock_response = MagicMock()
    mock_response.json.return_value = {"name": "ThreatIntelAgent", "version": "1.0"}
    with patch("agents.a2a_client.httpx.get", return_value=mock_response):
        client = A2AClient(base_url="http://localhost:8888")
        result = client.fetch_agent_card()
    assert result == {"name": "ThreatIntelAgent", "version": "1.0"}
