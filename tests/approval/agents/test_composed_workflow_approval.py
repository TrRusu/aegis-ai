"""
Approval tests for the composed workflow — run_composed_workflow().
"""
import json
from unittest.mock import AsyncMock, MagicMock, patch

from approvaltests import verify

from agents.composed_workflow import run_composed_workflow


@patch("agents.composed_workflow.ChatOpenAI")
@patch("agents.composed_workflow.MultiServerMCPClient")
@patch("agents.composed_workflow.make_tools", return_value=[])
@patch("agents.composed_workflow.critical_response_node", new_callable=AsyncMock)
@patch("agents.composed_workflow.standard_response_node", new_callable=AsyncMock)
def test_run_composed_workflow_return_structure(mock_standard, mock_critical, mock_tools, mock_mcp, mock_llm):
    """Approval: run_composed_workflow returns a (str, list) tuple."""
    mock_mcp.return_value.get_tools = AsyncMock(return_value=[])
    mock_llm.return_value.invoke.return_value = MagicMock(content="Severity: Medium\nThreat analysis.")
    mock_standard.return_value = {"response_plan": "Standard plan.", "tool_calls_log": []}
    mock_critical.return_value = {"response_plan": "Critical plan.", "tool_calls_log": []}

    result = run_composed_workflow("Phishing attack targeting finance employees.")

    verify(json.dumps({
        "return_type": type(result).__name__,
        "length": len(result),
        "element_types": [type(x).__name__ for x in result],
    }, indent=2))


@patch("agents.composed_workflow.ChatOpenAI")
@patch("agents.composed_workflow.MultiServerMCPClient")
@patch("agents.composed_workflow.make_tools", return_value=[])
@patch("agents.composed_workflow.critical_response_node", new_callable=AsyncMock)
@patch("agents.composed_workflow.standard_response_node", new_callable=AsyncMock)
def test_run_composed_workflow_routes_to_critical_on_high_severity(mock_standard, mock_critical, mock_tools, mock_mcp, mock_llm):
    """Approval: critical_response_node is invoked when severity is Critical."""
    mock_mcp.return_value.get_tools = AsyncMock(return_value=[])
    mock_llm.return_value.invoke.return_value = MagicMock(content="Severity: Critical\nThreat analysis.")
    mock_critical.return_value = {"response_plan": "Critical plan.", "tool_calls_log": []}

    run_composed_workflow("Ransomware attack on 500k records.")

    verify(json.dumps({
        "critical_called": mock_critical.called,
        "standard_called": mock_standard.called,
    }, indent=2))


@patch("agents.composed_workflow.ChatOpenAI")
@patch("agents.composed_workflow.MultiServerMCPClient")
@patch("agents.composed_workflow.make_tools", return_value=[])
@patch("agents.composed_workflow.critical_response_node", new_callable=AsyncMock)
@patch("agents.composed_workflow.standard_response_node", new_callable=AsyncMock)
def test_run_composed_workflow_routes_to_standard_on_low_severity(mock_standard, mock_critical, mock_tools, mock_mcp, mock_llm):
    """Approval: standard_response_node is invoked when severity is Medium."""
    mock_mcp.return_value.get_tools = AsyncMock(return_value=[])
    mock_llm.return_value.invoke.return_value = MagicMock(content="Severity: Medium\nThreat analysis.")
    mock_standard.return_value = {"response_plan": "Standard plan.", "tool_calls_log": []}

    run_composed_workflow("Phishing attack, low impact.")

    verify(json.dumps({
        "critical_called": mock_critical.called,
        "standard_called": mock_standard.called,
    }, indent=2))


@patch("agents.composed_workflow.ChatOpenAI")
@patch("agents.composed_workflow.MultiServerMCPClient")
@patch("agents.composed_workflow.make_tools", return_value=[])
@patch("agents.composed_workflow.critical_response_node", new_callable=AsyncMock)
@patch("agents.composed_workflow.standard_response_node", new_callable=AsyncMock)
def test_run_composed_workflow_tool_calls_captured(mock_standard, mock_critical, mock_tools, mock_mcp, mock_llm):
    """Approval: tool calls from the response node are captured in the log."""
    mock_mcp.return_value.get_tools = AsyncMock(return_value=[])
    mock_llm.return_value.invoke.return_value = MagicMock(content="Severity: Critical\nThreat analysis.")
    mock_critical.return_value = {
        "response_plan": "Critical plan.",
        "tool_calls_log": [
            {"tool": "search_knowledge_base", "input": {"query": "ransomware"}, "output": "KB result"},
        ],
    }

    _, tool_calls_log = run_composed_workflow("Ransomware attack with known CVE.")

    verify(json.dumps([c["tool"] for c in tool_calls_log], indent=2))


@patch("agents.composed_workflow.ChatOpenAI")
@patch("agents.composed_workflow.MultiServerMCPClient")
@patch("agents.composed_workflow.make_tools", return_value=[])
@patch("agents.composed_workflow.critical_response_node", new_callable=AsyncMock)
@patch("agents.composed_workflow.standard_response_node", new_callable=AsyncMock)
def test_run_composed_workflow_failure_behavior(mock_standard, mock_critical, mock_tools, mock_mcp, mock_llm):
    """Approval: run_composed_workflow returns fallback message and empty list on failure."""
    mock_mcp.return_value.get_tools = AsyncMock(side_effect=Exception("MCP failed"))

    report, tool_calls = run_composed_workflow("test incident")

    verify(json.dumps({"report": report, "tool_calls": tool_calls}, indent=2))
