import streamlit as st
from app.llm import build_llm, build_messages


def handle(prompt: str, history: list, temperature: float, max_tokens: int) -> str:
    llm = build_llm(temperature=temperature, max_tokens=max_tokens)
    messages = build_messages(history)
    return st.write_stream(chunk.content for chunk in llm.stream(messages))
