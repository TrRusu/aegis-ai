"""
Approval tests for app/llm.py.
Captures the current behavior before any refactoring.
"""
import json
from unittest.mock import patch

from approvaltests import verify
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.llm import build_llm, build_messages


# ── build_llm ──────────────────────────────────────────────────────────────────

@patch("app.llm.ChatOpenAI")
def test_build_llm_returns_chat_openai(mock_llm):
    """Approval: build_llm returns a ChatOpenAI instance."""
    result = build_llm()

    verify(json.dumps({
        "return_type": type(result).__name__,
    }, indent=2))


@patch("app.llm.ChatOpenAI")
def test_build_llm_passes_correct_params(mock_llm):
    """Approval: build_llm passes temperature, max_tokens, and streaming to ChatOpenAI."""
    build_llm(temperature=0.5, max_tokens=512)

    call_kwargs = mock_llm.call_args.kwargs
    verify(json.dumps({
        "temperature": call_kwargs.get("temperature"),
        "max_tokens": call_kwargs.get("max_tokens"),
        "streaming": call_kwargs.get("streaming"),
    }, indent=2))


# ── build_messages ─────────────────────────────────────────────────────────────

def test_build_messages_empty_history():
    """Approval: build_messages with empty history returns only system message."""
    messages = build_messages([])

    verify(json.dumps({
        "count": len(messages),
        "types": [type(m).__name__ for m in messages],
    }, indent=2))


def test_build_messages_with_history():
    """Approval: build_messages converts history to correct message types."""
    history = [
        {"role": "user", "content": "What is a breach?"},
        {"role": "assistant", "content": "A breach is..."},
        {"role": "user", "content": "How much does it cost?"},
    ]

    messages = build_messages(history)

    verify(json.dumps({
        "count": len(messages),
        "types": [type(m).__name__ for m in messages],
    }, indent=2))


def test_build_messages_system_prompt_is_first():
    """Approval: first message is always a SystemMessage."""
    messages = build_messages([{"role": "user", "content": "test"}])

    verify(type(messages[0]).__name__)
