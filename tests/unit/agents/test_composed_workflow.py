"""
Unit tests for ComposedWorkflow class in agents/composed_workflow.py (TDD).
"""
from unittest.mock import AsyncMock, MagicMock, patch


def test_composed_workflow_run_returns_tuple():
    from agents.composed_workflow import ComposedWorkflow
    mock_llm = MagicMock()
    workflow = ComposedWorkflow(llm=mock_llm)
    with patch("agents.composed_workflow.MultiServerMCPClient") as mock_mcp, \
         patch("agents.composed_workflow.make_tools", return_value=[]):
        mock_mcp.return_value.get_tools = AsyncMock(return_value=[])
        mock_llm.invoke.return_value = MagicMock(content="Severity: Medium\nReport.")
        result = workflow.run("Phishing attack.")
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_composed_workflow_run_returns_fallback_on_failure():
    from agents.composed_workflow import ComposedWorkflow
    from observability.fault_tolerance import FALLBACK_MESSAGE
    mock_llm = MagicMock()
    workflow = ComposedWorkflow(llm=mock_llm)
    with patch("agents.composed_workflow.MultiServerMCPClient") as mock_mcp:
        mock_mcp.return_value.get_tools = AsyncMock(side_effect=Exception("MCP failed"))
        response, tool_calls = workflow.run("test incident")
    assert response == FALLBACK_MESSAGE
    assert tool_calls == []
