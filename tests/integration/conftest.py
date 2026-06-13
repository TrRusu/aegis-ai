import pytest


class FakeEmbedder:
    def embed_documents(self, texts):
        return [[0.1] * 1536 for _ in texts]

    def embed_query(self, text):
        return [0.1] * 1536


@pytest.fixture(autouse=True)
def fix_embedding_mock(monkeypatch):
    monkeypatch.setattr("rag.store.OpenAIEmbeddings", lambda **kwargs: FakeEmbedder())
