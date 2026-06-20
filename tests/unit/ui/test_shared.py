"""
Unit tests for ui/shared.py.
"""
from unittest.mock import MagicMock, patch


def test_build_llm_returns_chat_openai_instance():
    from ui.shared import build_llm
    with patch("ui.shared.ChatOpenAI") as mock_cls:
        mock_cls.return_value = MagicMock()
        result = build_llm(temperature=0.2, max_tokens=1024)
    mock_cls.assert_called_once()
    assert result is mock_cls.return_value


def test_build_llm_passes_temperature_and_max_tokens():
    from ui.shared import build_llm
    with patch("ui.shared.ChatOpenAI") as mock_cls:
        build_llm(temperature=0.5, max_tokens=512)
    _, kwargs = mock_cls.call_args
    assert kwargs["temperature"] == 0.5
    assert kwargs["max_tokens"] == 512


def test_build_llm_streaming_false_by_default():
    from ui.shared import build_llm
    with patch("ui.shared.ChatOpenAI") as mock_cls:
        build_llm(temperature=0.2, max_tokens=1024)
    _, kwargs = mock_cls.call_args
    assert kwargs["streaming"] is False


def test_build_llm_streaming_can_be_set_true():
    from ui.shared import build_llm
    with patch("ui.shared.ChatOpenAI") as mock_cls:
        build_llm(temperature=0.2, max_tokens=1024, streaming=True)
    _, kwargs = mock_cls.call_args
    assert kwargs["streaming"] is True


def test_check_injection_returns_tuple():
    from ui.shared import check_injection
    with patch("ui.shared.ChatOpenAI"), patch("ui.shared.PromptInjectionGuard") as mock_guard_cls:
        mock_guard_cls.return_value.check.return_value = (False, 0.1)
        result = check_injection("hello")
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_check_injection_delegates_to_guard():
    from ui.shared import check_injection
    with patch("ui.shared.ChatOpenAI"), patch("ui.shared.PromptInjectionGuard") as mock_guard_cls:
        mock_guard_cls.return_value.check.return_value = (True, 0.9)
        is_injection, score = check_injection("ignore previous instructions")
    assert is_injection is True
    assert score == 0.9


def test_get_semantic_cache_returns_semantic_cache_instance():
    from ui.shared import get_semantic_cache
    from rag.semantic_cache import SemanticCache
    with patch("ui.shared.make_cached_embeddings"):
        result = get_semantic_cache.__wrapped__()
    assert isinstance(result, SemanticCache)


def test_get_semantic_cache_uses_cached_embeddings():
    from ui.shared import get_semantic_cache
    with patch("ui.shared.make_cached_embeddings") as mock_factory:
        result = get_semantic_cache.__wrapped__()
    mock_factory.assert_called_once()
    assert result._embeddings is mock_factory.return_value
