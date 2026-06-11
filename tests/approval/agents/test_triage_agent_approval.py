"""
Approval tests for the breach triage agent.
"""
import json
from unittest.mock import AsyncMock, MagicMock, patch

from approvaltests import verify
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agents.breach_triage_agent import BreachTriageAgent


def _make_agent_result(tool_names=None):
    messages = [HumanMessage(content="test incident")]

    if tool_names:
        ai_msg = AIMessage(content="")
        ai_msg.tool_calls = [
            {"name": name, "args": {"query": "test"}, "id": f"call_{i}"}
            for i, name in enumerate(tool_names)
        ]
        messages.append(ai_msg)

        for i, name in enumerate(tool_names):
            tool_msg = ToolMessage(content=f"Result from {name}", tool_call_id=f"call_{i}")
            tool_msg.name = name
            messages.append(tool_msg)

    messages.append(AIMessage(content="Final triage report."))
    return {"messages": messages}

@patch("agents.breach_triage_agent.MultiServerMCPClient")
@patch("agents.breach_triage_agent.make_tools", return_value=[])
@patch("agents.breach_triage_agent.create_agent")
def test_run_agent_return_structure(mock_agent, mock_tools, mock_mcp):
    """Approval: BreachTriageAgent.run returns a (str, list) tuple."""
    mock_mcp.return_value.get_tools = AsyncMock(return_value=[])
    mock_agent.return_value.ainvoke = AsyncMock(return_value=_make_agent_result())
    mock_llm = MagicMock()

    result = BreachTriageAgent(llm=mock_llm).run("A ransomware attack on healthcare systems.")

    verify(json.dumps({
        "return_type": type(result).__name__,
        "length": len(result),
        "element_types": [type(x).__name__ for x in result],
    }, indent=2))

@patch("agents.breach_triage_agent.MultiServerMCPClient")
@patch("agents.breach_triage_agent.make_tools", return_value=[])
@patch("agents.breach_triage_agent.create_agent")
def test_run_agent_tool_call_entry_keys(mock_agent, mock_tools, mock_mcp):
    """Approval: each tool call entry has tool, input, and output keys."""
    mock_mcp.return_value.get_tools = AsyncMock(return_value=[])
    mock_agent.return_value.ainvoke = AsyncMock(
        return_value=_make_agent_result(tool_names=["search_knowledge_base"])
    )
    mock_llm = MagicMock()

    _, tool_calls_log = BreachTriageAgent(llm=mock_llm).run("A ransomware attack on healthcare systems.")

    verify(json.dumps([sorted(entry.keys()) for entry in tool_calls_log], indent=2))

@patch("agents.breach_triage_agent.MultiServerMCPClient")
@patch("agents.breach_triage_agent.make_tools", return_value=[])
@patch("agents.breach_triage_agent.create_agent")
def test_run_agent_tool_call_sequence(mock_agent, mock_tools, mock_mcp):
    """Approval: all tool calls are captured in order."""
    mock_mcp.return_value.get_tools = AsyncMock(return_value=[])
    mock_agent.return_value.ainvoke = AsyncMock(
        return_value=_make_agent_result(
            tool_names=["search_knowledge_base", "lookup_cve", "calculate_breach_cost"]
        )
    )
    mock_llm = MagicMock()

    _, tool_calls_log = BreachTriageAgent(llm=mock_llm).run("A ransomware attack on healthcare systems.")

    verify(json.dumps([c["tool"] for c in tool_calls_log], indent=2))

@patch("agents.breach_triage_agent.MultiServerMCPClient")
@patch("agents.breach_triage_agent.make_tools", return_value=[])
@patch("agents.breach_triage_agent.create_agent")
def test_run_agent_no_tools_behavior(mock_agent, mock_tools, mock_mcp):
    """Approval: tool_calls_log is empty when the agent responds without calling any tools."""
    mock_mcp.return_value.get_tools = AsyncMock(return_value=[])
    mock_agent.return_value.ainvoke = AsyncMock(return_value=_make_agent_result(tool_names=None))
    mock_llm = MagicMock()

    _, tool_calls_log = BreachTriageAgent(llm=mock_llm).run("What is a data breach?")

    verify(json.dumps(tool_calls_log, indent=2))

@patch("agents.breach_triage_agent.MultiServerMCPClient")
@patch("agents.breach_triage_agent.make_tools", return_value=[])
def test_run_agent_failure_behavior(mock_tools, mock_mcp):
    """Approval: BreachTriageAgent.run returns fallback message and empty list on failure."""
    mock_mcp.return_value.get_tools = AsyncMock(side_effect=Exception("MCP failed"))
    mock_llm = MagicMock()

    response, tool_calls = BreachTriageAgent(llm=mock_llm).run("test incident")

    verify(json.dumps({"response": response, "tool_calls": tool_calls}, indent=2))
