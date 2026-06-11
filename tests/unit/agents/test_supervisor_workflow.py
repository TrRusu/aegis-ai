"""
Unit tests for SupervisorWorkflow class in agents/supervisor_workflow.py (TDD).
"""
from unittest.mock import AsyncMock, MagicMock, patch


def test_supervisor_workflow_run_returns_tuple():
    from agents.supervisor_workflow import SupervisorWorkflow
    mock_llm = MagicMock()
    workflow = SupervisorWorkflow(llm=mock_llm)
    with patch("agents.supervisor_workflow.MultiServerMCPClient") as mock_mcp, \
         patch("agents.supervisor_workflow.make_tools", return_value=[]):
        mock_mcp.return_value.get_tools = AsyncMock(return_value=[])
        mock_llm.invoke.return_value = MagicMock(content="Report.")
        result = workflow.run("Phishing attack.")
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_supervisor_workflow_run_returns_fallback_on_failure():
    from agents.supervisor_workflow import SupervisorWorkflow
    from observability.fault_tolerance import FALLBACK_MESSAGE
    mock_llm = MagicMock()
    workflow = SupervisorWorkflow(llm=mock_llm)
    with patch("agents.supervisor_workflow.MultiServerMCPClient") as mock_mcp:
        mock_mcp.return_value.get_tools = AsyncMock(side_effect=Exception("MCP failed"))
        response, tool_calls = workflow.run("test incident")
    assert response == FALLBACK_MESSAGE
    assert tool_calls == []
