"""
Unit tests for ToolChain class in tools/chain.py (TDD).
"""
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage


def test_tool_chain_run_returns_tuple():
    from tools.chain import ToolChain
    mock_llm = MagicMock()
    ai_msg = AIMessage(content="Final answer.")
    ai_msg.tool_calls = []
    mock_llm.bind_tools.return_value.ainvoke = AsyncMock(return_value=ai_msg)
    chain = ToolChain(llm=mock_llm)
    with patch("tools.chain.MultiServerMCPClient") as mock_mcp:
        mock_mcp.return_value.get_tools = AsyncMock(return_value=[])
        result = chain.run("What is the average breach cost?", [], tools=[])
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_tool_chain_run_returns_string_and_list():
    from tools.chain import ToolChain
    mock_llm = MagicMock()
    ai_msg = AIMessage(content="The average breach cost is $4.88M.")
    ai_msg.tool_calls = []
    mock_llm.bind_tools.return_value.ainvoke = AsyncMock(return_value=ai_msg)
    chain = ToolChain(llm=mock_llm)
    with patch("tools.chain.MultiServerMCPClient") as mock_mcp:
        mock_mcp.return_value.get_tools = AsyncMock(return_value=[])
        response, tool_calls_log = chain.run("What is the average breach cost?", [], tools=[])
    assert isinstance(response, str)
    assert isinstance(tool_calls_log, list)


def test_tool_chain_run_empty_tool_calls_when_no_tools_used():
    from tools.chain import ToolChain
    mock_llm = MagicMock()
    ai_msg = AIMessage(content="Direct answer.")
    ai_msg.tool_calls = []
    mock_llm.bind_tools.return_value.ainvoke = AsyncMock(return_value=ai_msg)
    chain = ToolChain(llm=mock_llm)
    with patch("tools.chain.MultiServerMCPClient") as mock_mcp:
        mock_mcp.return_value.get_tools = AsyncMock(return_value=[])
        _, tool_calls_log = chain.run("Hello", [], tools=[])
    assert tool_calls_log == []


def test_tool_chain_run_failure_returns_fallback():
    from tools.chain import ToolChain
    from observability.fault_tolerance import FALLBACK_MESSAGE
    mock_llm = MagicMock()
    chain = ToolChain(llm=mock_llm)
    with patch("tools.chain.MultiServerMCPClient") as mock_mcp:
        mock_mcp.return_value.get_tools = AsyncMock(side_effect=Exception("MCP failed"))
        response, tool_calls_log = chain.run("test", [], tools=[])
    assert response == FALLBACK_MESSAGE
    assert tool_calls_log == []
