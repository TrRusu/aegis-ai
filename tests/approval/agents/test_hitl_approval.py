"""
Approval tests for the HITL supervisor workflow.
"""
import json
from unittest.mock import AsyncMock, MagicMock, patch

from approvaltests import verify
from langchain_core.messages import AIMessage, HumanMessage

from agents.supervisor_hitl import HitlSupervisor, requires_approval


def _make_phase1_result(severity_label="Medium"):
    content = f"""**Incident Summary**: Ransomware attack detected.

**Severity Assessment**: Based on the incident details.
Severity: {severity_label}

**Specialist Findings**: No specialists invoked.
**Specialists not invoked**: All — insufficient detail."""
    return {"messages": [
        HumanMessage(content="test incident"),
        AIMessage(content=content),
    ]}

def test_requires_approval_low():
    """Approval: Low severity requires human approval."""
    verify(str(requires_approval("Low")))

def test_requires_approval_medium():
    """Approval: Medium severity requires human approval."""
    verify(str(requires_approval("Medium")))

def test_requires_approval_unknown():
    """Approval: Unknown severity requires human approval."""
    verify(str(requires_approval("Unknown")))

def test_requires_approval_high():
    """Approval: High severity bypasses the approval gate."""
    verify(str(requires_approval("High")))

def test_requires_approval_critical():
    """Approval: Critical severity bypasses the approval gate."""
    verify(str(requires_approval("Critical")))

@patch("agents.supervisor_hitl.MultiServerMCPClient")
@patch("agents.supervisor_hitl.make_tools", return_value=[])
@patch("agents.supervisor_hitl.create_agent")
def test_run_hitl_phase1_return_structure(mock_agent, mock_tools, mock_mcp):
    """Approval: run_phase1 returns a (str, str, list) tuple."""
    mock_mcp.return_value.get_tools = AsyncMock(return_value=[])
    mock_agent.return_value.ainvoke = AsyncMock(return_value=_make_phase1_result("Medium"))
    mock_llm = MagicMock()

    result = HitlSupervisor(llm=mock_llm).run_phase1("An employee lost a laptop.")

    verify(json.dumps({
        "return_type": type(result).__name__,
        "length": len(result),
        "element_types": [type(x).__name__ for x in result],
    }, indent=2))

@patch("agents.supervisor_hitl.MultiServerMCPClient")
@patch("agents.supervisor_hitl.make_tools", return_value=[])
@patch("agents.supervisor_hitl.create_agent")
def test_run_hitl_phase1_severity_extraction(mock_agent, mock_tools, mock_mcp):
    """Approval: severity is correctly extracted from the phase 1 output."""
    mock_mcp.return_value.get_tools = AsyncMock(return_value=[])
    mock_agent.return_value.ainvoke = AsyncMock(return_value=_make_phase1_result("Critical"))
    mock_llm = MagicMock()

    _, severity, _ = HitlSupervisor(llm=mock_llm).run_phase1("Ransomware attack, 500k records exfiltrated.")

    verify(severity)

@patch("agents.supervisor_hitl.MultiServerMCPClient")
@patch("agents.supervisor_hitl.make_tools", return_value=[])
@patch("agents.supervisor_hitl.create_agent")
def test_run_hitl_phase1_failure_behavior(mock_agent, mock_tools, mock_mcp):
    """Approval: run_phase1 returns fallback, Unknown severity, and empty list on failure."""
    mock_mcp.return_value.get_tools = AsyncMock(side_effect=Exception("MCP failed"))
    mock_llm = MagicMock()

    preliminary, severity, tool_calls = HitlSupervisor(llm=mock_llm).run_phase1("test incident")

    verify(json.dumps({
        "preliminary": preliminary,
        "severity": severity,
        "tool_calls": tool_calls,
    }, indent=2))

def test_run_hitl_phase2_return_type():
    """Approval: run_phase2 returns a non-empty string."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Final report content.")

    result = HitlSupervisor(llm=mock_llm).run_phase2(
        incident="test incident",
        preliminary_analysis="Phase 1 briefing.",
        decision="APPROVED",
        reason="Confirmed with SOC team.",
    )

    verify(json.dumps({
        "return_type": type(result).__name__,
        "is_non_empty": len(result) > 0,
    }, indent=2))

def test_run_hitl_phase2_failure_behavior():
    """Approval: run_phase2 returns fallback message on failure."""
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = Exception("LLM failed")

    result = HitlSupervisor(llm=mock_llm).run_phase2(
        incident="test incident",
        preliminary_analysis="Phase 1 briefing.",
        decision="APPROVED",
        reason="test",
    )

    verify(result)
