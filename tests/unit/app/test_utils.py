"""
Unit tests for app/utils.py.
"""

from app.utils import extract_text, run_in_thread


def test_extract_text_returns_string_unchanged():
    assert extract_text("hello") == "hello"


def test_extract_text_with_list_of_text_blocks():
    content = [{"type": "text", "text": "Hello"}, {"type": "text", "text": "World"}]
    assert extract_text(content) == "Hello\nWorld"


def test_extract_text_with_list_skips_non_text_blocks():
    content = [{"type": "image", "url": "..."}, {"type": "text", "text": "Only this"}]
    assert extract_text(content) == "Only this"


def test_extract_text_with_non_string_non_list():
    assert extract_text(42) == "42"


def test_run_in_thread_returns_coroutine_result():
    async def coro():
        return "done"

    result = run_in_thread(coro())
    assert result == "done"


def test_run_in_thread_propagates_exception():
    async def failing_coro():
        raise ValueError("boom")

    try:
        run_in_thread(failing_coro())
        assert False, "should have raised"
    except ValueError as e:
        assert str(e) == "boom"
