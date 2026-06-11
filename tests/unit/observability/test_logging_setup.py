"""
Unit tests for CallLogger class in observability/logging_setup.py (TDD).
"""
import asyncio
from unittest.mock import MagicMock


def test_call_logger_passes_through_return_value():
    from observability.logging_setup import CallLogger
    mock_logger = MagicMock()
    cl = CallLogger(logger=mock_logger)

    @cl.log_llm_call("Test")
    def fn():
        return "result"

    assert fn() == "result"


def test_call_logger_logs_request_started():
    from observability.logging_setup import CallLogger
    mock_logger = MagicMock()
    cl = CallLogger(logger=mock_logger)

    @cl.log_llm_call("Test")
    def fn():
        return "ok"

    fn()
    mock_logger.info.assert_any_call("[Test] Request started")


def test_call_logger_logs_completion():
    from observability.logging_setup import CallLogger
    mock_logger = MagicMock()
    cl = CallLogger(logger=mock_logger)

    @cl.log_llm_call("Test")
    def fn():
        return "ok"

    fn()
    calls = [str(c) for c in mock_logger.info.call_args_list]
    assert any("Request completed" in c for c in calls)


def test_call_logger_logs_error_and_reraises():
    from observability.logging_setup import CallLogger
    import pytest
    mock_logger = MagicMock()
    cl = CallLogger(logger=mock_logger)

    @cl.log_llm_call("Test")
    def fn():
        raise ValueError("boom")

    with pytest.raises(ValueError):
        fn()

    mock_logger.error.assert_called_once()


def test_call_logger_preserves_function_name():
    from observability.logging_setup import CallLogger
    mock_logger = MagicMock()
    cl = CallLogger(logger=mock_logger)

    @cl.log_llm_call("Test")
    def my_function():
        return "ok"

    assert my_function.__name__ == "my_function"


def test_call_logger_wraps_async_functions():
    from observability.logging_setup import CallLogger
    mock_logger = MagicMock()
    cl = CallLogger(logger=mock_logger)

    @cl.log_llm_call("Test")
    async def async_fn():
        return "async result"

    result = asyncio.run(async_fn())
    assert result == "async result"


def test_call_logger_logs_async_completion():
    from observability.logging_setup import CallLogger
    mock_logger = MagicMock()
    cl = CallLogger(logger=mock_logger)

    @cl.log_llm_call("Async")
    async def async_fn():
        return "ok"

    asyncio.run(async_fn())
    calls = [str(c) for c in mock_logger.info.call_args_list]
    assert any("Request completed" in c for c in calls)


def test_call_logger_logs_async_error_and_reraises():
    from observability.logging_setup import CallLogger
    import pytest
    mock_logger = MagicMock()
    cl = CallLogger(logger=mock_logger)

    @cl.log_llm_call("Async")
    async def async_fn():
        raise RuntimeError("async boom")

    with pytest.raises(RuntimeError):
        asyncio.run(async_fn())

    mock_logger.error.assert_called_once()
