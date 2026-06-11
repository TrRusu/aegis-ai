"""
Approval tests for the composed workflow — ComposedWorkflow.run().
"""
import json
from unittest.mock import AsyncMock, MagicMock, patch

from approvaltests import verify
from langchain_core.messages import AIMessage, ToolMessage

from agents.composed_workflow import ComposedWorkflow
from prompts.composed_workflow import CRITICAL_RESPONSE_PROMPT, STANDARD_RESPONSE_PROMPT


def _make_response_result(tool_entries=None):
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
    messages.append(AIMessage(content="Response plan."))
    return {"messages": messages}


def _routing_agent_factory():
    critical_called = False
    standard_called = False

    def factory(llm, tools, prompt):
        nonlocal critical_called, standard_called
        m = MagicMock()
        m.ainvoke = AsyncMock(return_value=_make_response_result())
        if prompt == CRITICAL_RESPONSE_PROMPT:
            critical_called = True
        elif prompt == STANDARD_RESPONSE_PROMPT:
            standard_called = True
        return m

    return factory, lambda: (critical_called, standard_called)


@patch("agents.composed_workflow.create_agent")
@patch("agents.composed_workflow.MultiServerMCPClient")
@patch("agents.composed_workflow.make_tools", return_value=[])
def test_run_composed_workflow_return_structure(mock_tools, mock_mcp, mock_create_agent):
    """Approval: ComposedWorkflow.run returns a (str, list) tuple."""
    mock_mcp.return_value.get_tools = AsyncMock(return_value=[])
    mock_create_agent.return_value.ainvoke = AsyncMock(return_value=_make_response_result())
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Severity: Medium\nThreat analysis.")

    result = ComposedWorkflow(llm=mock_llm).run("Phishing attack targeting finance employees.")

    verify(json.dumps({
        "return_type": type(result).__name__,
        "length": len(result),
        "element_types": [type(x).__name__ for x in result],
    }, indent=2))

@patch("agents.composed_workflow.create_agent")
@patch("agents.composed_workflow.MultiServerMCPClient")
@patch("agents.composed_workflow.make_tools", return_value=[])
def test_run_composed_workflow_routes_to_critical_on_high_severity(mock_tools, mock_mcp, mock_create_agent):
    """Approval: critical path is taken when severity is Critical."""
    mock_mcp.return_value.get_tools = AsyncMock(return_value=[])
    factory, get_flags = _routing_agent_factory()
    mock_create_agent.side_effect = factory
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Severity: Critical\nThreat analysis.")

    ComposedWorkflow(llm=mock_llm).run("Ransomware attack on 500k records.")
    critical_called, standard_called = get_flags()

    verify(json.dumps({
        "critical_called": critical_called,
        "standard_called": standard_called,
    }, indent=2))

@patch("agents.composed_workflow.create_agent")
@patch("agents.composed_workflow.MultiServerMCPClient")
@patch("agents.composed_workflow.make_tools", return_value=[])
def test_run_composed_workflow_routes_to_standard_on_low_severity(mock_tools, mock_mcp, mock_create_agent):
    """Approval: standard path is taken when severity is Medium."""
    mock_mcp.return_value.get_tools = AsyncMock(return_value=[])
    factory, get_flags = _routing_agent_factory()
    mock_create_agent.side_effect = factory
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Severity: Medium\nThreat analysis.")

    ComposedWorkflow(llm=mock_llm).run("Phishing attack, low impact.")
    critical_called, standard_called = get_flags()

    verify(json.dumps({
        "critical_called": critical_called,
        "standard_called": standard_called,
    }, indent=2))

@patch("agents.composed_workflow.create_agent")
@patch("agents.composed_workflow.MultiServerMCPClient")
@patch("agents.composed_workflow.make_tools", return_value=[])
def test_run_composed_workflow_tool_calls_captured(mock_tools, mock_mcp, mock_create_agent):
    """Approval: tool calls from the response node are captured in the log."""
    mock_mcp.return_value.get_tools = AsyncMock(return_value=[])
    tool_entries = [
        {"tool": "search_knowledge_base", "input": {"query": "ransomware"}, "output": "KB result"},
    ]
    mock_create_agent.return_value.ainvoke = AsyncMock(return_value=_make_response_result(tool_entries))
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Severity: Critical\nThreat analysis.")

    _, tool_calls_log = ComposedWorkflow(llm=mock_llm).run("Ransomware attack with known CVE.")

    verify(json.dumps([c["tool"] for c in tool_calls_log], indent=2))

@patch("agents.composed_workflow.MultiServerMCPClient")
@patch("agents.composed_workflow.make_tools", return_value=[])
def test_run_composed_workflow_failure_behavior(mock_tools, mock_mcp):
    """Approval: ComposedWorkflow.run returns fallback message and empty list on failure."""
    mock_mcp.return_value.get_tools = AsyncMock(side_effect=Exception("MCP failed"))
    mock_llm = MagicMock()

    report, tool_calls = ComposedWorkflow(llm=mock_llm).run("test incident")

    verify(json.dumps({"report": report, "tool_calls": tool_calls}, indent=2))
