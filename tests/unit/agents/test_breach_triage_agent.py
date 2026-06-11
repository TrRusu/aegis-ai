"""
Unit tests for BreachTriageAgent class in agents/breach_triage_agent.py (TDD).
"""
from unittest.mock import AsyncMock, MagicMock, patch


def test_breach_triage_agent_run_returns_tuple():
    from agents.breach_triage_agent import BreachTriageAgent
    mock_llm = MagicMock()
    agent = BreachTriageAgent(llm=mock_llm)
    with patch("agents.breach_triage_agent.MultiServerMCPClient") as mock_mcp, \
         patch("agents.breach_triage_agent.create_agent") as mock_react:
        mock_mcp.return_value.get_tools = AsyncMock(return_value=[])
        mock_react.return_value.ainvoke = AsyncMock(return_value={
            "messages": [MagicMock(content="Triage report.", tool_calls=[])]
        })
        result = agent.run("Ransomware on hospital network.")
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_breach_triage_agent_run_returns_string_and_list():
    from agents.breach_triage_agent import BreachTriageAgent
    mock_llm = MagicMock()
    agent = BreachTriageAgent(llm=mock_llm)
    with patch("agents.breach_triage_agent.MultiServerMCPClient") as mock_mcp, \
         patch("agents.breach_triage_agent.create_agent") as mock_react:
        mock_mcp.return_value.get_tools = AsyncMock(return_value=[])
        mock_react.return_value.ainvoke = AsyncMock(return_value={
            "messages": [MagicMock(content="Triage report.", tool_calls=[])]
        })
        response, tool_calls = agent.run("Ransomware on hospital network.")
    assert isinstance(response, str)
    assert isinstance(tool_calls, list)


def test_breach_triage_agent_run_returns_fallback_on_failure():
    from agents.breach_triage_agent import BreachTriageAgent
    from observability.fault_tolerance import FALLBACK_MESSAGE
    mock_llm = MagicMock()
    agent = BreachTriageAgent(llm=mock_llm)
    with patch("agents.breach_triage_agent.MultiServerMCPClient") as mock_mcp:
        mock_mcp.return_value.get_tools = AsyncMock(side_effect=Exception("MCP failed"))
        response, tool_calls = agent.run("test incident")
    assert response == FALLBACK_MESSAGE
    assert tool_calls == []
