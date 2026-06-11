"""
Approval tests for the supervisor workflow.
"""
import json
from unittest.mock import AsyncMock, MagicMock, patch

from approvaltests import verify
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agents.supervisor_workflow import SupervisorWorkflow


def _make_supervisor_result(specialists=None):
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
@patch("agents.supervisor_workflow.create_agent")
def test_run_supervisor_return_structure(mock_agent, mock_tools, mock_mcp):
    """Approval: SupervisorWorkflow.run returns a (str, list) tuple."""
    mock_mcp.return_value.get_tools = AsyncMock(return_value=[])
    mock_agent.return_value.ainvoke = AsyncMock(return_value=_make_supervisor_result())
    mock_llm = MagicMock()

    result = SupervisorWorkflow(llm=mock_llm).run("A ransomware attack on healthcare systems.")

    verify(json.dumps({
        "return_type": type(result).__name__,
        "length": len(result),
        "element_types": [type(x).__name__ for x in result],
    }, indent=2))

@patch("agents.supervisor_workflow.MultiServerMCPClient")
@patch("agents.supervisor_workflow.make_tools", return_value=[])
@patch("agents.supervisor_workflow.create_agent")
def test_run_supervisor_specialist_sequence(mock_agent, mock_tools, mock_mcp):
    """Approval: specialist agents invoked are captured in order."""
    mock_mcp.return_value.get_tools = AsyncMock(return_value=[])
    mock_agent.return_value.ainvoke = AsyncMock(
        return_value=_make_supervisor_result(
            specialists=["cve_analyst", "cost_analyst", "compliance_analyst"]
        )
    )
    mock_llm = MagicMock()

    _, tool_calls_log = SupervisorWorkflow(llm=mock_llm).run("A ransomware attack on healthcare systems.")

    verify(json.dumps([c["tool"] for c in tool_calls_log], indent=2))

@patch("agents.supervisor_workflow.MultiServerMCPClient")
@patch("agents.supervisor_workflow.make_tools", return_value=[])
@patch("agents.supervisor_workflow.create_agent")
def test_run_supervisor_no_specialists_behavior(mock_agent, mock_tools, mock_mcp):
    """Approval: supervisor invokes no specialists for a vague incident."""
    mock_mcp.return_value.get_tools = AsyncMock(return_value=[])
    mock_agent.return_value.ainvoke = AsyncMock(return_value=_make_supervisor_result(specialists=None))
    mock_llm = MagicMock()

    _, tool_calls_log = SupervisorWorkflow(llm=mock_llm).run("We may have had some kind of security incident.")

    verify(json.dumps(tool_calls_log, indent=2))

@patch("agents.supervisor_workflow.MultiServerMCPClient")
@patch("agents.supervisor_workflow.make_tools", return_value=[])
def test_run_supervisor_failure_behavior(mock_tools, mock_mcp):
    """Approval: SupervisorWorkflow.run returns fallback message and empty list on failure."""
    mock_mcp.return_value.get_tools = AsyncMock(side_effect=Exception("MCP failed"))
    mock_llm = MagicMock()

    response, tool_calls = SupervisorWorkflow(llm=mock_llm).run("test incident")

    verify(json.dumps({"response": response, "tool_calls": tool_calls}, indent=2))
