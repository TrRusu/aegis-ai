import pytest
from unittest.mock import MagicMock, patch
from approvaltests import set_default_reporter
from approvaltests.reporters.python_native_reporter import PythonNativeReporter

# Use PythonNativeReporter in all environments — raises AssertionError with diff, no GUI
set_default_reporter(PythonNativeReporter())


@pytest.fixture(autouse=True)
def mock_openai(request):
    """Mock all OpenAI API calls globally. Applied to every test automatically.
    Tests that need real API calls can opt out with @pytest.mark.live_api."""
    if request.node.get_closest_marker("live_api"):
        yield
        return

    mock_response = MagicMock()
    mock_response.content = "Mocked LLM response for testing."

    with patch("langchain_openai.ChatOpenAI.invoke", return_value=mock_response), \
         patch("langchain_openai.ChatOpenAI.stream", return_value=iter([mock_response])), \
         patch("langchain_openai.OpenAIEmbeddings.embed_documents", return_value=[[0.1] * 1536]), \
         patch("langchain_openai.OpenAIEmbeddings.embed_query", return_value=[0.1] * 1536):
        yield
