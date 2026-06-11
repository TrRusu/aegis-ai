"""Configurable retry and timeout utilities with injectable config.
"""

import asyncio
import logging
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

logger = logging.getLogger("aegis.fault_tolerance")

TIMEOUT_SECONDS = 60
MAX_RETRIES = 3
FALLBACK_MESSAGE = (
    "Aegis is temporarily unavailable — the AI service did not respond in time. "
    "Please try again in a moment."
)


class FaultTolerance:

    def __init__(self, max_retries: int = MAX_RETRIES, timeout: int = TIMEOUT_SECONDS):
        self._max_retries = max_retries
        self._timeout = timeout

    def retry(self):
        return retry(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(Exception),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )

    async def invoke_with_timeout(self, coro, timeout: int = None):
        return await asyncio.wait_for(coro, timeout=timeout if timeout is not None else self._timeout)
