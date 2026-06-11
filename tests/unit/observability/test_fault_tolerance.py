"""
Unit tests for FaultTolerance class in observability/fault_tolerance.py (TDD).
"""
import asyncio


def test_fault_tolerance_invoke_with_timeout_returns_result():
    from observability.fault_tolerance import FaultTolerance

    async def fast_coro():
        return "done"

    result = asyncio.run(FaultTolerance().invoke_with_timeout(fast_coro()))
    assert result == "done"


def test_fault_tolerance_invoke_with_timeout_raises_on_timeout():
    from observability.fault_tolerance import FaultTolerance
    import pytest

    async def slow_coro():
        await asyncio.sleep(10)

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(FaultTolerance(timeout=0.01).invoke_with_timeout(slow_coro()))


def test_fault_tolerance_uses_injected_timeout():
    from observability.fault_tolerance import FaultTolerance
    import pytest

    async def slow_coro():
        await asyncio.sleep(10)

    ft = FaultTolerance(timeout=0.01)
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(ft.invoke_with_timeout(slow_coro()))


def test_fault_tolerance_retry_passes_through_on_success():
    from observability.fault_tolerance import FaultTolerance

    ft = FaultTolerance()

    @ft.retry()
    def always_succeeds():
        return "ok"

    assert always_succeeds() == "ok"


def test_fault_tolerance_retry_raises_after_max_retries():
    from observability.fault_tolerance import FaultTolerance
    import pytest

    ft = FaultTolerance(max_retries=2)
    call_count = {"n": 0}

    @ft.retry()
    def always_fails():
        call_count["n"] += 1
        raise ValueError("boom")

    with pytest.raises(ValueError):
        always_fails()

    assert call_count["n"] == 2


def test_fault_tolerance_uses_injected_max_retries():
    from observability.fault_tolerance import FaultTolerance

    ft = FaultTolerance(max_retries=3)
    call_count = {"n": 0}

    @ft.retry()
    def always_fails():
        call_count["n"] += 1
        raise ValueError("boom")

    try:
        always_fails()
    except ValueError:
        pass

    assert call_count["n"] == 3


def test_fault_tolerance_retry_returns_result_after_transient_failure():
    from observability.fault_tolerance import FaultTolerance

    ft = FaultTolerance(max_retries=3)
    attempts = {"n": 0}

    @ft.retry()
    def fails_once():
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise ValueError("transient")
        return "recovered"

    result = fails_once()
    assert result == "recovered"
    assert attempts["n"] == 2
