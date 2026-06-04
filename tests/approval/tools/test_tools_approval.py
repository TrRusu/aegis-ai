"""
Approval tests for the tools module.
Captures the current behavior before any refactoring.
"""
import json
from unittest.mock import MagicMock, patch

from approvaltests import verify

from tools.tools import make_tools


# ── calculate_breach_cost ──────────────────────────────────────────────────────

def test_calculate_breach_cost_return_message():
    """Approval: calculate_breach_cost returns a formatted cost string."""
    tools = make_tools()
    calculate = next(t for t in tools if t.name == "calculate_breach_cost")

    verify(calculate.invoke({"records_lost": 80000, "cost_per_record": 169.0}))


def test_calculate_breach_cost_small_numbers():
    """Approval: calculate_breach_cost formats small numbers correctly."""
    tools = make_tools()
    calculate = next(t for t in tools if t.name == "calculate_breach_cost")

    verify(calculate.invoke({"records_lost": 100, "cost_per_record": 9.99}))


def test_calculate_breach_cost_large_numbers():
    """Approval: calculate_breach_cost formats large numbers with commas."""
    tools = make_tools()
    calculate = next(t for t in tools if t.name == "calculate_breach_cost")

    verify(calculate.invoke({"records_lost": 1000000, "cost_per_record": 169.0}))


# ── search_knowledge_base ──────────────────────────────────────────────────────

@patch("tools.tools.build_retriever")
def test_search_knowledge_base_formats_results(mock_retriever):
    """Approval: search_knowledge_base formats retrieved chunks with page numbers."""
    from langchain_core.documents import Document

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


def test_make_tools_returns_two_tools():
    """Approval: make_tools returns exactly two tools with the correct names."""
    tools = make_tools()

    verify(json.dumps([t.name for t in tools], indent=2))
