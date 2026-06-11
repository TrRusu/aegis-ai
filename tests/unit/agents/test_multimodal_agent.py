"""
Unit tests for MultimodalAgent class in agents/multimodal_agent.py (TDD).
"""
from unittest.mock import MagicMock


def test_multimodal_agent_returns_incident_unchanged_when_no_image():
    from agents.multimodal_agent import MultimodalAgent
    mock_llm = MagicMock()
    agent = MultimodalAgent(llm=mock_llm)
    result = agent.enrich("Ransomware detected.", image_bytes=None)
    assert result == "Ransomware detected."


def test_multimodal_agent_returns_incident_unchanged_when_empty_bytes():
    from agents.multimodal_agent import MultimodalAgent
    mock_llm = MagicMock()
    agent = MultimodalAgent(llm=mock_llm)
    result = agent.enrich("Suspicious login.", image_bytes=b"")
    assert result == "Suspicious login."


def test_multimodal_agent_no_llm_call_when_no_image():
    from agents.multimodal_agent import MultimodalAgent
    mock_llm = MagicMock()
    agent = MultimodalAgent(llm=mock_llm)
    agent.enrich("Incident.", image_bytes=None)
    mock_llm.invoke.assert_not_called()


def test_multimodal_agent_delegates_to_injected_llm_when_image_provided():
    from agents.multimodal_agent import MultimodalAgent
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Enriched description.")
    agent = MultimodalAgent(llm=mock_llm)
    result = agent.enrich("Incident.", image_bytes=b"fake-png")
    mock_llm.invoke.assert_called_once()
    assert result == "Enriched description."
