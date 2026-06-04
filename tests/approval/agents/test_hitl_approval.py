"""
Approval tests for the HITL supervisor workflow.
Captures the current output structure before any refactoring.
"""
from unittest.mock import AsyncMock, MagicMock, patch

from observability.fault_tolerance import FALLBACK_MESSAGE
from agents.supervisor_hitl import run_hitl_phase1, run_hitl_phase2, requires_approval


def _make_phase1_result(severity_label="Medium"):
    """Build a fake LangGraph result with a severity line in the final message."""
    from langchain_core.messages import AIMessage, HumanMessage

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

def test_requires_approval_low_severity_needs_approval():
    """Approval: Low severity requires human approval."""
    assert requires_approval("Low") is True


def test_requires_approval_medium_severity_needs_approval():
    """Approval: Medium severity requires human approval."""
    assert requires_approval("Medium") is True


def test_requires_approval_unknown_severity_needs_approval():
    """Approval: Unknown severity requires human approval."""
    assert requires_approval("Unknown") is True


def test_requires_approval_high_severity_skips_approval():
    """Approval: High severity bypasses the approval gate."""
    assert requires_approval("High") is False


def test_requires_approval_critical_severity_skips_approval():
    """Approval: Critical severity bypasses the approval gate."""
    assert requires_approval("Critical") is False


# ── run_hitl_phase1 ────────────────────────────────────────────────────────────

@patch("agents.supervisor_hitl.ChatOpenAI")
@patch("agents.supervisor_hitl.MultiServerMCPClient")
@patch("agents.supervisor_hitl.make_tools", return_value=[])
@patch("agents.supervisor_hitl.create_react_agent")
def test_run_hitl_phase1_returns_three_tuple(mock_agent, mock_tools, mock_mcp, mock_llm):
    """Approval: run_hitl_phase1 returns (str, str, list) tuple."""
    mock_mcp.return_value.get_tools = AsyncMock(return_value=[])
    mock_agent.return_value.ainvoke = AsyncMock(
        return_value=_make_phase1_result("Medium")
    )

    result = run_hitl_phase1("An employee lost a laptop with customer data.")

    assert isinstance(result, tuple)
    assert len(result) == 3
    assert isinstance(result[0], str)  # preliminary analysis
    assert isinstance(result[1], str)  # severity
    assert isinstance(result[2], list)  # tool calls log


@patch("agents.supervisor_hitl.ChatOpenAI")
@patch("agents.supervisor_hitl.MultiServerMCPClient")
@patch("agents.supervisor_hitl.make_tools", return_value=[])
@patch("agents.supervisor_hitl.create_react_agent")
def test_run_hitl_phase1_extracts_severity(mock_agent, mock_tools, mock_mcp, mock_llm):
    """Approval: severity is correctly extracted from the phase 1 output."""
    mock_mcp.return_value.get_tools = AsyncMock(return_value=[])
    mock_agent.return_value.ainvoke = AsyncMock(
        return_value=_make_phase1_result("Critical")
    )

    _, severity, _ = run_hitl_phase1("Ransomware attack, 500k records exfiltrated.")

    assert severity == "Critical"


@patch("agents.supervisor_hitl.ChatOpenAI")
@patch("agents.supervisor_hitl.MultiServerMCPClient")
@patch("agents.supervisor_hitl.make_tools", return_value=[])
@patch("agents.supervisor_hitl.create_react_agent")
def test_run_hitl_phase1_returns_fallback_on_failure(mock_agent, mock_tools, mock_mcp, mock_llm):
    """Approval: run_hitl_phase1 returns fallback, Unknown severity, and empty list on failure."""
    mock_mcp.return_value.get_tools = AsyncMock(side_effect=Exception("MCP failed"))

    preliminary, severity, tool_calls = run_hitl_phase1("test incident")

    assert preliminary == FALLBACK_MESSAGE
    assert severity == "Unknown"
    assert tool_calls == []


# ── run_hitl_phase2 ────────────────────────────────────────────────────────────

@patch("agents.supervisor_hitl.ChatOpenAI")
def test_run_hitl_phase2_returns_string(mock_llm):
    """Approval: run_hitl_phase2 returns a string final report."""
    mock_llm.return_value.invoke.return_value = MagicMock(content="Final report content.")

    result = run_hitl_phase2(
        incident="test incident",
        preliminary_analysis="Phase 1 briefing.",
        decision="APPROVED",
        reason="Confirmed with SOC team.",
    )

    assert isinstance(result, str)
    assert len(result) > 0


@patch("agents.supervisor_hitl.ChatOpenAI")
def test_run_hitl_phase2_returns_fallback_on_failure(mock_llm):
    """Approval: run_hitl_phase2 returns fallback message on failure."""
    mock_llm.return_value.invoke.side_effect = Exception("LLM failed")

    result = run_hitl_phase2(
        incident="test incident",
        preliminary_analysis="Phase 1 briefing.",
        decision="APPROVED",
        reason="test",
    )

    assert result == FALLBACK_MESSAGE
