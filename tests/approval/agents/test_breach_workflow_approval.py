"""
Approval tests for the breach workflow — BreachWorkflow.run().
"""
import json
from unittest.mock import AsyncMock, MagicMock, patch

from approvaltests import verify
from langchain_core.messages import AIMessage, ToolMessage

from agents.breach_workflow import BreachWorkflow


def _make_research_result(tool_entries=None):
    messages = []
    if tool_entries:
        ai_msg = AIMessage(content="")
        ai_msg.tool_calls = [
            {"name": e["tool"], "args": e["input"], "id": f"call_{i}"}
            for i, e in enumerate(tool_entries)
        ]
        messages.append(ai_msg)
        for i, e in enumerate(tool_entries):
            tm = ToolMessage(content=e["output"], tool_call_id=f"call_{i}")
            tm.name = e["tool"]
            messages.append(tm)
    messages.append(AIMessage(content="Research findings."))
    return {"messages": messages}


@patch("agents.breach_workflow.create_agent")
@patch("agents.breach_workflow.MultiServerMCPClient")
@patch("agents.breach_workflow.make_tools", return_value=[])
def test_run_workflow_return_structure(mock_tools, mock_mcp, mock_create_agent):
    """Approval: BreachWorkflow.run returns a (str, list) tuple."""
    mock_mcp.return_value.get_tools = AsyncMock(return_value=[])
    mock_create_agent.return_value.ainvoke = AsyncMock(return_value=_make_research_result())
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="LLM output.")

    result = BreachWorkflow(llm=mock_llm).run("Ransomware attack on 500k healthcare records.")

    verify(json.dumps({
        "return_type": type(result).__name__,
        "length": len(result),
        "element_types": [type(x).__name__ for x in result],
    }, indent=2))

@patch("agents.breach_workflow.create_agent")
@patch("agents.breach_workflow.MultiServerMCPClient")
@patch("agents.breach_workflow.make_tools", return_value=[])
def test_run_workflow_tool_calls_captured(mock_tools, mock_mcp, mock_create_agent):
    """Approval: tool calls from the research node are captured in the log."""
    mock_mcp.return_value.get_tools = AsyncMock(return_value=[])
    tool_entries = [
        {"tool": "search_knowledge_base", "input": {"query": "ransomware"}, "output": "KB result"},
        {"tool": "lookup_cve", "input": {"cve_id": "CVE-2021-44228"}, "output": "CVE result"},
    ]
    mock_create_agent.return_value.ainvoke = AsyncMock(return_value=_make_research_result(tool_entries))
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="LLM output.")

    _, tool_calls_log = BreachWorkflow(llm=mock_llm).run("Ransomware attack with CVE-2021-44228.")

    verify(json.dumps([c["tool"] for c in tool_calls_log], indent=2))

@patch("agents.breach_workflow.create_agent")
@patch("agents.breach_workflow.MultiServerMCPClient")
@patch("agents.breach_workflow.make_tools", return_value=[])
def test_run_workflow_empty_tool_calls_when_no_tools_used(mock_tools, mock_mcp, mock_create_agent):
    """Approval: tool_calls_log is empty when the research agent uses no tools."""
    mock_mcp.return_value.get_tools = AsyncMock(return_value=[])
    mock_create_agent.return_value.ainvoke = AsyncMock(return_value=_make_research_result())
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="LLM output.")

    _, tool_calls_log = BreachWorkflow(llm=mock_llm).run("Vague incident description.")

    verify(json.dumps(tool_calls_log, indent=2))

@patch("agents.breach_workflow.MultiServerMCPClient")
@patch("agents.breach_workflow.make_tools", return_value=[])
def test_run_workflow_failure_behavior(mock_tools, mock_mcp):
    """Approval: BreachWorkflow.run returns fallback message and empty list on failure."""
    mock_mcp.return_value.get_tools = AsyncMock(side_effect=Exception("MCP failed"))
    mock_llm = MagicMock()

    report, tool_calls = BreachWorkflow(llm=mock_llm).run("test incident")

    verify(json.dumps({"report": report, "tool_calls": tool_calls}, indent=2))
