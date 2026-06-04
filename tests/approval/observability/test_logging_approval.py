"""
Approval tests for observability/logging_setup.py.
Captures the current behavior before any refactoring.
"""

from approvaltests import verify

from observability.logging_setup import log_llm_call


# ── log_llm_call ───────────────────────────────────────────────────────────────

def test_log_llm_call_sync_passes_return_value():
    """Approval: log_llm_call decorator passes through the return value of a sync function."""
    @log_llm_call("Test")
    def sync_fn():
        return "sync result"

    verify(sync_fn())


def test_log_llm_call_async_passes_return_value():
    """Approval: log_llm_call decorator passes through the return value of an async function."""
    import asyncio

    @log_llm_call("Test")
    async def async_fn():
        return "async result"

    verify(asyncio.run(async_fn()))


def test_log_llm_call_preserves_function_name():
    """Approval: log_llm_call preserves the original function name via functools.wraps."""
    @log_llm_call("Test")
    def my_function():
        return "result"

    verify(my_function.__name__)


def test_log_llm_call_sync_reraises_exception():
    """Approval: log_llm_call re-raises exceptions from the wrapped sync function."""
    import pytest

    @log_llm_call("Test")
    def failing_fn():
        raise ValueError("test error")

    with pytest.raises(ValueError, match="test error"):
        failing_fn()

    verify("ValueError re-raised as expected")


def test_log_llm_call_async_reraises_exception():
    """Approval: log_llm_call re-raises exceptions from the wrapped async function."""
    import asyncio
    import pytest

    @log_llm_call("Test")
    async def async_failing_fn():
        raise ValueError("async test error")

    with pytest.raises(ValueError, match="async test error"):
        asyncio.run(async_failing_fn())

    verify("ValueError re-raised from async function as expected")
