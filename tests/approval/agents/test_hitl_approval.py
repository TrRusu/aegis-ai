"""
Approval tests for the HITL supervisor workflow.
Captures the current output structure before any refactoring.
"""
import json
from unittest.mock import AsyncMock, MagicMock, patch

from approvaltests import verify
from langchain_core.messages import AIMessage, HumanMessage

from agents.supervisor_hitl import run_hitl_phase1, run_hitl_phase2, requires_approval


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


# ── requires_approval ──────────────────────────────────────────────────────────

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


# ── run_hitl_phase1 ────────────────────────────────────────────────────────────

@patch("agents.supervisor_hitl.ChatOpenAI")
@patch("agents.supervisor_hitl.MultiServerMCPClient")
@patch("agents.supervisor_hitl.make_tools", return_value=[])
@patch("agents.supervisor_hitl.create_react_agent")
def test_run_hitl_phase1_return_structure(mock_agent, mock_tools, mock_mcp, mock_llm):
    """Approval: run_hitl_phase1 returns a (str, str, list) tuple."""
    mock_mcp.return_value.get_tools = AsyncMock(return_value=[])
    mock_agent.return_value.ainvoke = AsyncMock(return_value=_make_phase1_result("Medium"))

    result = run_hitl_phase1("An employee lost a laptop.")

    verify(json.dumps({
        "return_type": type(result).__name__,
        "length": len(result),
        "element_types": [type(x).__name__ for x in result],
    }, indent=2))


@patch("agents.supervisor_hitl.ChatOpenAI")
@patch("agents.supervisor_hitl.MultiServerMCPClient")
@patch("agents.supervisor_hitl.make_tools", return_value=[])
@patch("agents.supervisor_hitl.create_react_agent")
def test_run_hitl_phase1_severity_extraction(mock_agent, mock_tools, mock_mcp, mock_llm):
    """Approval: severity is correctly extracted from the phase 1 output."""
    mock_mcp.return_value.get_tools = AsyncMock(return_value=[])
    mock_agent.return_value.ainvoke = AsyncMock(return_value=_make_phase1_result("Critical"))

    _, severity, _ = run_hitl_phase1("Ransomware attack, 500k records exfiltrated.")

    verify(severity)


@patch("agents.supervisor_hitl.ChatOpenAI")
@patch("agents.supervisor_hitl.MultiServerMCPClient")
@patch("agents.supervisor_hitl.make_tools", return_value=[])
@patch("agents.supervisor_hitl.create_react_agent")
def test_run_hitl_phase1_failure_behavior(mock_agent, mock_tools, mock_mcp, mock_llm):
    """Approval: run_hitl_phase1 returns fallback, Unknown severity, and empty list on failure."""
    mock_mcp.return_value.get_tools = AsyncMock(side_effect=Exception("MCP failed"))

    preliminary, severity, tool_calls = run_hitl_phase1("test incident")

    verify(json.dumps({
        "preliminary": preliminary,
        "severity": severity,
        "tool_calls": tool_calls,
    }, indent=2))


# ── run_hitl_phase2 ────────────────────────────────────────────────────────────

@patch("agents.supervisor_hitl.ChatOpenAI")
def test_run_hitl_phase2_return_type(mock_llm):
    """Approval: run_hitl_phase2 returns a non-empty string."""
    mock_llm.return_value.invoke.return_value = MagicMock(content="Final report content.")

    result = run_hitl_phase2(
        incident="test incident",
        preliminary_analysis="Phase 1 briefing.",
        decision="APPROVED",
        reason="Confirmed with SOC team.",
    )

    verify(json.dumps({
        "return_type": type(result).__name__,
        "is_non_empty": len(result) > 0,
    }, indent=2))


@patch("agents.supervisor_hitl.ChatOpenAI")
def test_run_hitl_phase2_failure_behavior(mock_llm):
    """Approval: run_hitl_phase2 returns fallback message on failure."""
    mock_llm.return_value.invoke.side_effect = Exception("LLM failed")

    result = run_hitl_phase2(
        incident="test incident",
        preliminary_analysis="Phase 1 briefing.",
        decision="APPROVED",
        reason="test",
    )

    verify(result)
