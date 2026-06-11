"""
Unit tests for ThreatIntelAgent class in a2a_server/threat_intel_server.py (TDD).
"""
from unittest.mock import MagicMock


def test_threat_intel_agent_returns_string():
    from a2a_server.threat_intel_server import ThreatIntelAgent
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="MITRE ATT&CK analysis result.")
    agent = ThreatIntelAgent(llm=mock_llm)
    result = agent.analyse("Ransomware attack on hospital.")
    assert isinstance(result, str)


def test_threat_intel_agent_delegates_to_injected_llm():
    from a2a_server.threat_intel_server import ThreatIntelAgent
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Analysis.")
    agent = ThreatIntelAgent(llm=mock_llm)
    agent.analyse("Some incident.")
    mock_llm.invoke.assert_called_once()


def test_threat_intel_agent_returns_llm_content():
    from a2a_server.threat_intel_server import ThreatIntelAgent
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Threat actor: APT29.")
    agent = ThreatIntelAgent(llm=mock_llm)
    result = agent.analyse("Phishing attack.")
    assert result == "Threat actor: APT29."


def test_threat_intel_agent_handles_list_content_format():
    from a2a_server.threat_intel_server import ThreatIntelAgent
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content=[
        {"type": "text", "text": "MITRE mapping complete."},
    ])
    agent = ThreatIntelAgent(llm=mock_llm)
    result = agent.analyse("Incident description.")
    assert result == "MITRE mapping complete."


def test_threat_intel_agent_passes_incident_to_llm():
    from a2a_server.threat_intel_server import ThreatIntelAgent
    from langchain_core.messages import HumanMessage
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Done.")
    agent = ThreatIntelAgent(llm=mock_llm)
    agent.analyse("Supply chain attack on software vendor.")
    messages = mock_llm.invoke.call_args[0][0]
    human_messages = [m for m in messages if isinstance(m, HumanMessage)]
    assert any("Supply chain attack" in m.content for m in human_messages)
