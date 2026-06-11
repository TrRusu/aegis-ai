"""Logs timing for sync and async LLM calls using an injectable logger.
"""

import asyncio
import logging
import time
import functools
from typing import Callable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger("aegis")


class CallLogger:
    
    def __init__(self, logger: logging.Logger = None):
        self._logger = logger or logging.getLogger("aegis")

    def log_llm_call(self, mode: str) -> Callable:
        _logger = self._logger

        def decorator(fn: Callable) -> Callable:
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                start = time.perf_counter()
                _logger.info(f"[{mode}] Request started")
                try:
                    result = fn(*args, **kwargs)
                    elapsed = time.perf_counter() - start
                    _logger.info(f"[{mode}] Request completed in {elapsed:.2f}s")
                    return result
                except Exception as exc:
                    elapsed = time.perf_counter() - start
                    _logger.error(f"[{mode}] Request failed after {elapsed:.2f}s — {type(exc).__name__}: {exc}")
                    raise
            return wrapper

        def async_decorator(fn: Callable) -> Callable:
            @functools.wraps(fn)
            async def wrapper(*args, **kwargs):
                start = time.perf_counter()
                _logger.info(f"[{mode}] Request started")
                try:
                    result = await fn(*args, **kwargs)
                    elapsed = time.perf_counter() - start
                    _logger.info(f"[{mode}] Request completed in {elapsed:.2f}s")
                    return result
                except Exception as exc:
                    elapsed = time.perf_counter() - start
                    _logger.error(f"[{mode}] Request failed after {elapsed:.2f}s — {type(exc).__name__}: {exc}")
                    raise
            return wrapper

        def smart_decorator(fn: Callable) -> Callable:
            if asyncio.iscoroutinefunction(fn):
                return async_decorator(fn)
            return decorator(fn)

        return smart_decorator