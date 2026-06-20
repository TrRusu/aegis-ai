from unittest.mock import MagicMock
from rag.semantic_cache import SemanticCache


def _make_embedder(vectors: dict) -> MagicMock:
    embedder = MagicMock()
    embedder.embed_query.side_effect = lambda text: vectors[text]
    return embedder


def test_check_returns_none_when_cache_is_empty():
    cache = SemanticCache(embeddings=_make_embedder({"q": [1.0, 0.0]}), threshold=0.9)
    assert cache.check("q") is None


def test_check_returns_cached_response_for_identical_query():
    embedder = _make_embedder({"What is the breach cost?": [1.0, 0.0]})
    cache = SemanticCache(embeddings=embedder, threshold=0.9)
    cache.store("What is the breach cost?", "It is $4.45 million.")
    assert cache.check("What is the breach cost?") == "It is $4.45 million."


def test_check_returns_response_for_similar_query_above_threshold():
    embedder = _make_embedder({
        "What is the breach cost?": [1.0, 0.0],
        "How much does a breach cost?": [0.99, 0.141],
    })
    cache = SemanticCache(embeddings=embedder, threshold=0.9)
    cache.store("What is the breach cost?", "It is $4.45 million.")
    assert cache.check("How much does a breach cost?") == "It is $4.45 million."


def test_check_returns_none_for_dissimilar_query_below_threshold():
    embedder = _make_embedder({
        "What is the breach cost?": [1.0, 0.0],
        "What is the weather today?": [0.0, 1.0],
    })
    cache = SemanticCache(embeddings=embedder, threshold=0.9)
    cache.store("What is the breach cost?", "It is $4.45 million.")
    assert cache.check("What is the weather today?") is None


def test_check_returns_best_match_among_multiple_entries():
    embedder = _make_embedder({
        "a": [1.0, 0.0],
        "b": [0.0, 1.0],
        "query": [0.9, 0.1],
    })
    cache = SemanticCache(embeddings=embedder, threshold=0.5)
    cache.store("a", "response-a")
    cache.store("b", "response-b")
    assert cache.check("query") == "response-a"


def test_threshold_boundary_is_inclusive():
    embedder = _make_embedder({"a": [1.0, 0.0], "query": [1.0, 0.0]})
    cache = SemanticCache(embeddings=embedder, threshold=1.0)
    cache.store("a", "response-a")
    assert cache.check("query") == "response-a"


def test_store_does_not_raise_and_is_retrievable():
    embedder = _make_embedder({"a": [1.0, 0.0]})
    cache = SemanticCache(embeddings=embedder, threshold=0.9)
    cache.store("a", "response-a")
    assert cache.check("a") == "response-a"


def test_default_threshold_is_within_recommended_range():
    cache = SemanticCache(embeddings=MagicMock())
    assert 0.90 <= cache._threshold <= 0.95
