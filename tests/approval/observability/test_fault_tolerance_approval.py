"""
Approval tests for observability/fault_tolerance.py.
"""
import asyncio
import json

import pytest
from approvaltests import verify

from observability.fault_tolerance import (
    FALLBACK_MESSAGE,
    MAX_RETRIES,
    TIMEOUT_SECONDS,
    invoke_with_timeout,
    with_retry,
)

def test_fault_tolerance_constants():
    """Approval: verify fault tolerance constants."""
    verify(json.dumps({
        "TIMEOUT_SECONDS": TIMEOUT_SECONDS,
        "MAX_RETRIES": MAX_RETRIES,
        "FALLBACK_MESSAGE": FALLBACK_MESSAGE,
    }, indent=2))

def test_with_retry_returns_result_on_success():
    """Approval: with_retry passes through the return value on first success."""
    @with_retry()
    def always_succeeds():
        return "success"

    verify(always_succeeds())

def test_with_retry_retries_on_failure_then_succeeds():
    """Approval: with_retry retries and returns result when function eventually succeeds."""
    call_count = {"n": 0}

    @with_retry()
    def fails_twice_then_succeeds():
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise ValueError("temporary failure")
        return "recovered"

    result = fails_twice_then_succeeds()

    verify(json.dumps({
        "result": result,
        "total_calls": call_count["n"],
    }, indent=2))

def test_with_retry_raises_after_max_retries():
    """Approval: with_retry raises the original exception after MAX_RETRIES attempts."""
    call_count = {"n": 0}

    @with_retry()
    def always_fails():
        call_count["n"] += 1
        raise ValueError("permanent failure")

    with pytest.raises(ValueError):
        always_fails()

    verify(json.dumps({
        "total_calls": call_count["n"],
        "equals_max_retries": call_count["n"] == MAX_RETRIES,
    }, indent=2))

def test_invoke_with_timeout_returns_result():
    """Approval: invoke_with_timeout returns the coroutine result on success."""
    async def fast_coro():
        return "fast result"

    result = asyncio.run(invoke_with_timeout(fast_coro()))

    verify(result)

def test_invoke_with_timeout_raises_on_timeout():
    """Approval: invoke_with_timeout raises asyncio.TimeoutError when coro exceeds timeout."""
    async def slow_coro():
        await asyncio.sleep(10)
        return "too slow"

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(invoke_with_timeout(slow_coro(), timeout=0.01))

    verify("asyncio.TimeoutError raised as expected")