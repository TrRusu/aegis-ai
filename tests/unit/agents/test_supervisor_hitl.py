"""
Unit tests for HitlSupervisor class in agents/supervisor_hitl.py (TDD).
"""
from unittest.mock import AsyncMock, MagicMock, patch


def test_hitl_supervisor_phase1_returns_tuple():
    from agents.supervisor_hitl import HitlSupervisor
    mock_llm = MagicMock()
    supervisor = HitlSupervisor(llm=mock_llm)
    with patch("agents.supervisor_hitl.MultiServerMCPClient") as mock_mcp, \
         patch("agents.supervisor_hitl.make_tools", return_value=[]):
        mock_mcp.return_value.get_tools = AsyncMock(return_value=[])
        mock_llm.invoke.return_value = MagicMock(content="Briefing. Severity: Medium")
        result = supervisor.run_phase1("Credential theft.")
    assert isinstance(result, tuple)
    assert len(result) == 3


def test_hitl_supervisor_phase1_returns_fallback_on_failure():
    from agents.supervisor_hitl import HitlSupervisor
    from observability.fault_tolerance import FALLBACK_MESSAGE
    mock_llm = MagicMock()
    supervisor = HitlSupervisor(llm=mock_llm)
    with patch("agents.supervisor_hitl.MultiServerMCPClient") as mock_mcp:
        mock_mcp.return_value.get_tools = AsyncMock(side_effect=Exception("MCP failed"))
        briefing, severity, tool_calls = supervisor.run_phase1("test incident")
    assert briefing == FALLBACK_MESSAGE
    assert tool_calls == []


def test_hitl_supervisor_phase2_returns_string():
    from agents.supervisor_hitl import HitlSupervisor
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Final report.")
    supervisor = HitlSupervisor(llm=mock_llm)
    result = supervisor.run_phase2("Incident.", "Briefing.", decision="APPROVED", reason="Verified.")
    assert isinstance(result, str)


def test_hitl_supervisor_phase2_returns_fallback_on_failure():
    from agents.supervisor_hitl import HitlSupervisor
    from observability.fault_tolerance import FALLBACK_MESSAGE
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = Exception("LLM failed")
    supervisor = HitlSupervisor(llm=mock_llm)
    result = supervisor.run_phase2("Incident.", "Briefing.", decision="APPROVED", reason="Verified.")
    assert result == FALLBACK_MESSAGE


def test_requires_approval_returns_false_for_critical():
    from agents.supervisor_hitl import requires_approval
    assert requires_approval("Critical") is False


def test_requires_approval_returns_false_for_high():
    from agents.supervisor_hitl import requires_approval
    assert requires_approval("High") is False


def test_requires_approval_returns_true_for_medium():
    from agents.supervisor_hitl import requires_approval
    assert requires_approval("Medium") is True


def test_requires_approval_returns_true_for_low():
    from agents.supervisor_hitl import requires_approval
    assert requires_approval("Low") is True
