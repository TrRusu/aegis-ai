"""
Approval tests for the breach triage agent.
Captures the current output structure before any refactoring.
"""
from unittest.mock import AsyncMock, patch

from observability.fault_tolerance import FALLBACK_MESSAGE
from agents.breach_triage_agent import run_agent


def _make_agent_result(tool_names=None):
    """Build a fake LangGraph result with tool call messages."""
    from langchain_core.messages import AIMessage, ToolMessage, HumanMessage

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


@patch("agents.breach_triage_agent.ChatOpenAI")
@patch("agents.breach_triage_agent.MultiServerMCPClient")
@patch("agents.breach_triage_agent.make_tools", return_value=[])
@patch("agents.breach_triage_agent.create_react_agent")
def test_run_agent_returns_tuple(mock_agent, mock_tools, mock_mcp, mock_llm):
    """Approval: run_agent returns a (str, list) tuple."""
    mock_mcp.return_value.get_tools = AsyncMock(return_value=[])
    mock_agent.return_value.ainvoke = AsyncMock(
        return_value=_make_agent_result()
    )

    result = run_agent("A ransomware attack on healthcare systems.")

    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], str)
    assert isinstance(result[1], list)


@patch("agents.breach_triage_agent.ChatOpenAI")
@patch("agents.breach_triage_agent.MultiServerMCPClient")
@patch("agents.breach_triage_agent.make_tools", return_value=[])
@patch("agents.breach_triage_agent.create_react_agent")
def test_run_agent_tool_calls_have_required_keys(mock_agent, mock_tools, mock_mcp, mock_llm):
    """Approval: each tool call entry has tool, input, and output keys."""
    mock_mcp.return_value.get_tools = AsyncMock(return_value=[])
    mock_agent.return_value.ainvoke = AsyncMock(
        return_value=_make_agent_result(tool_names=["search_knowledge_base"])
    )

    _, tool_calls_log = run_agent("A ransomware attack on healthcare systems.")

    assert len(tool_calls_log) == 1
    entry = tool_calls_log[0]
    assert "tool" in entry
    assert "input" in entry
    assert "output" in entry
    assert entry["tool"] == "search_knowledge_base"


@patch("agents.breach_triage_agent.ChatOpenAI")
@patch("agents.breach_triage_agent.MultiServerMCPClient")
@patch("agents.breach_triage_agent.make_tools", return_value=[])
@patch("agents.breach_triage_agent.create_react_agent")
def test_run_agent_captures_multiple_tool_calls(mock_agent, mock_tools, mock_mcp, mock_llm):
    """Approval: all tool calls are captured when the agent calls multiple tools."""
    mock_mcp.return_value.get_tools = AsyncMock(return_value=[])
    mock_agent.return_value.ainvoke = AsyncMock(
        return_value=_make_agent_result(
            tool_names=["search_knowledge_base", "lookup_cve", "calculate_breach_cost"]
        )
    )

    _, tool_calls_log = run_agent("A ransomware attack on healthcare systems.")

    assert len(tool_calls_log) == 3
    assert tool_calls_log[0]["tool"] == "search_knowledge_base"
    assert tool_calls_log[1]["tool"] == "lookup_cve"
    assert tool_calls_log[2]["tool"] == "calculate_breach_cost"


@patch("agents.breach_triage_agent.ChatOpenAI")
@patch("agents.breach_triage_agent.MultiServerMCPClient")
@patch("agents.breach_triage_agent.make_tools", return_value=[])
@patch("agents.breach_triage_agent.create_react_agent")
def test_run_agent_empty_tool_calls_when_no_tools_used(mock_agent, mock_tools, mock_mcp, mock_llm):
    """Approval: tool_calls_log is empty when the agent responds without calling any tools."""
    mock_mcp.return_value.get_tools = AsyncMock(return_value=[])
    mock_agent.return_value.ainvoke = AsyncMock(
        return_value=_make_agent_result(tool_names=None)
    )

    _, tool_calls_log = run_agent("What is a data breach?")

    assert tool_calls_log == []


@patch("agents.breach_triage_agent.MultiServerMCPClient")
@patch("agents.breach_triage_agent.make_tools", return_value=[])
@patch("agents.breach_triage_agent.create_react_agent")
def test_run_agent_returns_fallback_on_failure(mock_agent, mock_tools, mock_mcp):
    """Approval: run_agent returns fallback message and empty list on failure."""
    mock_mcp.return_value.get_tools = AsyncMock(side_effect=Exception("MCP failed"))

    response, tool_calls = run_agent("test incident")

    assert response == FALLBACK_MESSAGE
    assert tool_calls == []
