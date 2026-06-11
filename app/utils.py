import asyncio
import concurrent.futures


def run_in_thread(coro, timeout: int = 45):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(asyncio.run, coro)
        return future.result(timeout=timeout)


def extract_text(content) -> str:
    """Normalise LLM response content to a plain string.
    Newer OpenAI SDK versions return a list of typed content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            b.get("text", "") for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return str(content)