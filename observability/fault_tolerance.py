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


def with_retry():
    """Tenacity retry decorator: 3 attempts, exponential backoff 1s → 10s."""
    return retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )


async def invoke_with_timeout(coro, timeout: int = TIMEOUT_SECONDS):
    """Run an awaitable with a hard timeout. Raises asyncio.TimeoutError on breach."""
    return await asyncio.wait_for(coro, timeout=timeout)
