import pytest
from unittest.mock import MagicMock, AsyncMock
from langchain_core.messages import AIMessage
from tools.chain import ToolChain


def _make_llm(side_effect: list) -> MagicMock:
    mock_llm = MagicMock()
    mock_bound = MagicMock()
    mock_bound.ainvoke = AsyncMock(side_effect=side_effect)
    mock_llm.bind_tools.return_value = mock_bound
    return mock_llm


def _ai_with_cve_call(cve_id: str = "CVE-2021-44228") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"id": "call_1", "name": "lookup_cve", "args": {"cve_id": cve_id}}],
    )


def _ai_final(text: str = "Analysis complete.") -> AIMessage:
    return AIMessage(content=text)


@pytest.mark.integration
def test_mcp_lookup_cve_tool_is_available():
    llm = _make_llm([_ai_final()])
    ToolChain(llm=llm).run("hello", [], tools=[])
    bound_tool_names = [t.name for t in llm.bind_tools.call_args[0][0]]
    assert "lookup_cve" in bound_tool_names


@pytest.mark.integration
def test_invokes_mcp_cve_tool_returns_nvd_data():
    llm = _make_llm([_ai_with_cve_call(), _ai_final()])
    _, tool_calls_log = ToolChain(llm=llm).run("What is Log4Shell?", [], tools=[])
    assert len(tool_calls_log) == 1
    assert tool_calls_log[0]["tool"] == "lookup_cve"
    assert "CVE-2021-44228" in tool_calls_log[0]["output"]
    assert "10.0" in tool_calls_log[0]["output"]


@pytest.mark.integration
def test_invalid_cve_returns_not_found_message():
    llm = _make_llm([_ai_with_cve_call("CVE-9999-99999"), _ai_final()])
    _, tool_calls_log = ToolChain(llm=llm).run("Look up CVE-9999-99999", [], tools=[])
    assert "No record found" in tool_calls_log[0]["output"]


@pytest.mark.integration
def test_no_tool_call_returns_llm_response():
    llm = _make_llm([_ai_final("No CVE needed here.")])
    response, tool_calls_log = ToolChain(llm=llm).run("Hello", [], tools=[])
    assert response == "No CVE needed here."
    assert tool_calls_log == []
