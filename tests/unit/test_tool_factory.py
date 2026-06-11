"""
Unit tests for ToolFactory class in tools/tools.py (TDD).
"""
from unittest.mock import MagicMock

from langchain_core.documents import Document


def test_tool_factory_builds_two_tools():
    from tools.tools import ToolFactory
    mock_retriever = MagicMock()
    factory = ToolFactory(retriever=mock_retriever)
    tools = factory.build()
    assert len(tools) == 2


def test_tool_factory_tool_names():
    from tools.tools import ToolFactory
    mock_retriever = MagicMock()
    tools = ToolFactory(retriever=mock_retriever).build()
    names = {t.name for t in tools}
    assert names == {"search_knowledge_base", "calculate_breach_cost"}


def test_search_knowledge_base_delegates_to_injected_retriever():
    from tools.tools import ToolFactory
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = []
    tools = ToolFactory(retriever=mock_retriever).build()
    search = next(t for t in tools if t.name == "search_knowledge_base")
    search.invoke({"query": "healthcare"})
    mock_retriever.invoke.assert_called_once_with("healthcare")


def test_search_knowledge_base_formats_chunks_with_page_numbers():
    from tools.tools import ToolFactory
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [
        Document(page_content="Healthcare breach cost was $9.77M", metadata={"page": 10}),
        Document(page_content="Average cost per record is $169", metadata={"page": 5}),
    ]
    tools = ToolFactory(retriever=mock_retriever).build()
    search = next(t for t in tools if t.name == "search_knowledge_base")
    result = search.invoke({"query": "healthcare"})
    assert "Chunk 1" in result
    assert "page 10" in result
    assert "Chunk 2" in result
    assert "page 5" in result


def test_search_knowledge_base_returns_no_results_message_when_empty():
    from tools.tools import ToolFactory
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = []
    tools = ToolFactory(retriever=mock_retriever).build()
    search = next(t for t in tools if t.name == "search_knowledge_base")
    result = search.invoke({"query": "nonexistent"})
    assert "No relevant information" in result


def test_calculate_breach_cost_formats_result():
    from tools.tools import ToolFactory
    tools = ToolFactory(retriever=MagicMock()).build()
    calculate = next(t for t in tools if t.name == "calculate_breach_cost")
    result = calculate.invoke({"records_lost": 80000, "cost_per_record": 169.0})
    assert "13,520,000.00" in result
    assert "80,000" in result
