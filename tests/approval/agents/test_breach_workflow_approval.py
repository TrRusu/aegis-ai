"""
Approval tests for the breach workflow — run_workflow().
"""
import json
from unittest.mock import AsyncMock, MagicMock, patch

from approvaltests import verify

from agents.breach_workflow import run_workflow


@patch("agents.breach_workflow.ChatOpenAI")
@patch("agents.breach_workflow.MultiServerMCPClient")
@patch("agents.breach_workflow.make_tools", return_value=[])
@patch("agents.breach_workflow._research_node_async", new_callable=AsyncMock)
def test_run_workflow_return_structure(mock_research, mock_tools, mock_mcp, mock_llm):
    """Approval: run_workflow returns a (str, list) tuple."""
    mock_mcp.return_value.get_tools = AsyncMock(return_value=[])
    mock_research.return_value = {"research": "Research findings.", "tool_calls_log": []}
    mock_llm.return_value.invoke.return_value = MagicMock(content="LLM output.")

    result = run_workflow("Ransomware attack on 500k healthcare records.")

    verify(json.dumps({
        "return_type": type(result).__name__,
        "length": len(result),
        "element_types": [type(x).__name__ for x in result],
    }, indent=2))

@patch("agents.breach_workflow.ChatOpenAI")
@patch("agents.breach_workflow.MultiServerMCPClient")
@patch("agents.breach_workflow.make_tools", return_value=[])
@patch("agents.breach_workflow._research_node_async", new_callable=AsyncMock)
def test_run_workflow_tool_calls_captured(mock_research, mock_tools, mock_mcp, mock_llm):
    """Approval: tool calls from the research node are captured in the log."""
    mock_mcp.return_value.get_tools = AsyncMock(return_value=[])
    mock_research.return_value = {
        "research": "Research findings.",
        "tool_calls_log": [
            {"tool": "search_knowledge_base", "input": {"query": "ransomware"}, "output": "KB result"},
            {"tool": "lookup_cve", "input": {"cve_id": "CVE-2021-44228"}, "output": "CVE result"},
        ],
    }
    mock_llm.return_value.invoke.return_value = MagicMock(content="LLM output.")

    _, tool_calls_log = run_workflow("Ransomware attack with CVE-2021-44228.")

    verify(json.dumps([c["tool"] for c in tool_calls_log], indent=2))

@patch("agents.breach_workflow.ChatOpenAI")
@patch("agents.breach_workflow.MultiServerMCPClient")
@patch("agents.breach_workflow.make_tools", return_value=[])
@patch("agents.breach_workflow._research_node_async", new_callable=AsyncMock)
def test_run_workflow_empty_tool_calls_when_no_tools_used(mock_research, mock_tools, mock_mcp, mock_llm):
    """Approval: tool_calls_log is empty when the research agent uses no tools."""
    mock_mcp.return_value.get_tools = AsyncMock(return_value=[])
    mock_research.return_value = {"research": "No tools needed.", "tool_calls_log": []}
    mock_llm.return_value.invoke.return_value = MagicMock(content="LLM output.")

    _, tool_calls_log = run_workflow("Vague incident description.")

    verify(json.dumps(tool_calls_log, indent=2))

@patch("agents.breach_workflow.ChatOpenAI")
@patch("agents.breach_workflow.MultiServerMCPClient")
@patch("agents.breach_workflow.make_tools", return_value=[])
@patch("agents.breach_workflow._research_node_async", new_callable=AsyncMock)
def test_run_workflow_failure_behavior(mock_research, mock_tools, mock_mcp, mock_llm):
    """Approval: run_workflow returns fallback message and empty list on failure."""
    mock_mcp.return_value.get_tools = AsyncMock(side_effect=Exception("MCP failed"))

    report, tool_calls = run_workflow("test incident")

    verify(json.dumps({"report": report, "tool_calls": tool_calls}, indent=2))
