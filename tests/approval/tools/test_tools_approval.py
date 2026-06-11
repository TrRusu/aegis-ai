"""
Approval tests for the tools module.
"""
import json
from unittest.mock import AsyncMock, MagicMock, patch

from approvaltests import verify
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from tools.tools import make_tools
from tools.chain import ToolChain


@patch("tools.tools.build_retriever")
def test_calculate_breach_cost_return_message(mock_retriever):
    """Approval: calculate_breach_cost returns a formatted cost string."""
    tools = make_tools()
    calculate = next(t for t in tools if t.name == "calculate_breach_cost")

    verify(calculate.invoke({"records_lost": 80000, "cost_per_record": 169.0}))

@patch("tools.tools.build_retriever")
def test_calculate_breach_cost_small_numbers(mock_retriever):
    """Approval: calculate_breach_cost formats small numbers correctly."""
    tools = make_tools()
    calculate = next(t for t in tools if t.name == "calculate_breach_cost")

    verify(calculate.invoke({"records_lost": 100, "cost_per_record": 9.99}))

@patch("tools.tools.build_retriever")
def test_calculate_breach_cost_large_numbers(mock_retriever):
    """Approval: calculate_breach_cost formats large numbers with commas."""
    tools = make_tools()
    calculate = next(t for t in tools if t.name == "calculate_breach_cost")

    verify(calculate.invoke({"records_lost": 1000000, "cost_per_record": 169.0}))

@patch("tools.tools.build_retriever")
def test_search_knowledge_base_formats_results(mock_retriever):
    """Approval: search_knowledge_base formats retrieved chunks with page numbers."""
    mock_retriever.return_value.invoke.return_value = [
        Document(page_content="Healthcare breach cost was $9.77M", metadata={"page": 10}),
        Document(page_content="Average cost per record is $169", metadata={"page": 5}),
    ]

    tools = make_tools(k=2)
    search = next(t for t in tools if t.name == "search_knowledge_base")

    verify(search.invoke({"query": "healthcare breach cost"}))

@patch("tools.tools.build_retriever")
def test_search_knowledge_base_no_results_message(mock_retriever):
    """Approval: search_knowledge_base returns a specific message when nothing is found."""
    mock_retriever.return_value.invoke.return_value = []

    tools = make_tools()
    search = next(t for t in tools if t.name == "search_knowledge_base")

    verify(search.invoke({"query": "nonexistent topic"}))

@patch("tools.tools.build_retriever")
def test_make_tools_returns_two_tools(mock_retriever):
    """Approval: make_tools returns exactly two tools with the correct names."""
    tools = make_tools()

    verify(json.dumps([t.name for t in tools], indent=2))

@patch("tools.chain.MultiServerMCPClient")
def test_run_with_tools_returns_tuple(mock_mcp):
    """Approval: ToolChain.run returns a (str, list) tuple."""
    mock_mcp.return_value.get_tools = AsyncMock(return_value=[])
    mock_llm = MagicMock()
    ai_msg = AIMessage(content="Final answer.")
    ai_msg.tool_calls = []
    mock_llm.bind_tools.return_value.ainvoke = AsyncMock(return_value=ai_msg)

    result = ToolChain(llm=mock_llm).run("What is the average breach cost?", [], [])

    verify(json.dumps({
        "return_type": type(result).__name__,
        "length": len(result),
        "element_types": [type(x).__name__ for x in result],
    }, indent=2))

@patch("tools.chain.MultiServerMCPClient")
def test_run_with_tools_empty_tool_calls_when_no_tools_used(mock_mcp):
    """Approval: tool_calls_log is empty when the LLM responds without calling any tools."""
    mock_mcp.return_value.get_tools = AsyncMock(return_value=[])
    mock_llm = MagicMock()
    ai_msg = AIMessage(content="Direct answer.")
    ai_msg.tool_calls = []
    mock_llm.bind_tools.return_value.ainvoke = AsyncMock(return_value=ai_msg)

    _, tool_calls_log = ToolChain(llm=mock_llm).run("Hello", [], [])

    verify(json.dumps(tool_calls_log, indent=2))

@patch("tools.chain.MultiServerMCPClient")
def test_run_with_tools_failure_behavior(mock_mcp):
    """Approval: ToolChain.run returns fallback and empty list on failure."""
    mock_mcp.return_value.get_tools = AsyncMock(side_effect=Exception("MCP failed"))
    mock_llm = MagicMock()

    response, tool_calls = ToolChain(llm=mock_llm).run("test", [], [])

    verify(json.dumps({"response": response, "tool_calls": tool_calls}, indent=2))
