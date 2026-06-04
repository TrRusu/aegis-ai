"""
Approval tests for the supervisor workflow.
Captures the current output structure before any refactoring.
"""
from unittest.mock import AsyncMock, MagicMock, patch

from observability.fault_tolerance import FALLBACK_MESSAGE
from agents.supervisor_workflow import run_supervisor


def _make_supervisor_result(specialists=None):
    """Build a fake LangGraph result simulating supervisor with specialist tool calls."""
    from langchain_core.messages import AIMessage, ToolMessage, HumanMessage

    messages = [HumanMessage(content="test incident")]

    if specialists:
        ai_msg = AIMessage(content="")
        ai_msg.tool_calls = [
            {"name": name, "args": {"incident_summary": "test"}, "id": f"call_{i}"}
            for i, name in enumerate(specialists)
        ]
        messages.append(ai_msg)

        for i, name in enumerate(specialists):
            tool_msg = ToolMessage(content=f"Analysis from {name}", tool_call_id=f"call_{i}")
            tool_msg.name = name
            messages.append(tool_msg)

    messages.append(AIMessage(content="Supervisor final report."))
    return {"messages": messages}


@patch("agents.supervisor_workflow.MultiServerMCPClient")
@patch("agents.supervisor_workflow.make_tools", return_value=[])
@patch("agents.supervisor_workflow.create_react_agent")
def test_run_supervisor_returns_tuple(mock_agent, mock_tools, mock_mcp):
    """Approval: run_supervisor returns a (str, list) tuple."""
    mock_mcp.return_value.get_tools = AsyncMock(return_value=[])
    mock_agent.return_value.ainvoke = AsyncMock(
        return_value=_make_supervisor_result()
    )

    result = run_supervisor("A ransomware attack on healthcare systems.")

    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], str)
    assert isinstance(result[1], list)


@patch("agents.supervisor_workflow.MultiServerMCPClient")
@patch("agents.supervisor_workflow.make_tools", return_value=[])
@patch("agents.supervisor_workflow.create_react_agent")
def test_run_supervisor_captures_specialist_calls(mock_agent, mock_tools, mock_mcp):
    """Approval: specialist agents invoked by supervisor are captured in tool_calls_log."""
    mock_mcp.return_value.get_tools = AsyncMock(return_value=[])
    mock_agent.return_value.ainvoke = AsyncMock(
        return_value=_make_supervisor_result(
            specialists=["cve_analyst", "cost_analyst", "compliance_analyst"]
        )
    )

    _, tool_calls_log = run_supervisor("A ransomware attack on healthcare systems.")

    assert len(tool_calls_log) == 3
    assert tool_calls_log[0]["tool"] == "cve_analyst"
    assert tool_calls_log[1]["tool"] == "cost_analyst"
    assert tool_calls_log[2]["tool"] == "compliance_analyst"


@patch("agents.supervisor_workflow.MultiServerMCPClient")
@patch("agents.supervisor_workflow.make_tools", return_value=[])
@patch("agents.supervisor_workflow.create_react_agent")
def test_run_supervisor_no_specialists_when_vague_incident(mock_agent, mock_tools, mock_mcp):
    """Approval: supervisor invokes no specialists for a vague incident."""
    mock_mcp.return_value.get_tools = AsyncMock(return_value=[])
    mock_agent.return_value.ainvoke = AsyncMock(
        return_value=_make_supervisor_result(specialists=None)
    )

    _, tool_calls_log = run_supervisor("We may have had some kind of security incident.")

    assert tool_calls_log == []


@patch("agents.supervisor_workflow.MultiServerMCPClient")
@patch("agents.supervisor_workflow.make_tools", return_value=[])
@patch("agents.supervisor_workflow.create_react_agent")
def test_run_supervisor_returns_fallback_on_failure(mock_agent, mock_tools, mock_mcp):
    """Approval: run_supervisor returns fallback message and empty list on failure."""
    mock_mcp.return_value.get_tools = AsyncMock(side_effect=Exception("MCP failed"))

    response, tool_calls = run_supervisor("test incident")

    assert response == FALLBACK_MESSAGE
    assert tool_calls == []
