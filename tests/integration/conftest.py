import pytest


@pytest.fixture(autouse=True)
def fix_embedding_mock(monkeypatch):
    """Override the global embed_documents mock to return one vector per document.

    The root conftest patches embed_documents with return_value=[[0.1]*1536], which
    always returns a single embedding regardless of input size. ChromaDB requires
    exactly one embedding per document, so we replace it with a side_effect here.
    """
    monkeypatch.setattr(
        "langchain_openai.OpenAIEmbeddings.embed_documents",
        lambda self, texts: [[0.1] * 1536 for _ in texts],
    )
