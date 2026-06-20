import streamlit as st
from app.config import OPENAI_API_KEY, OPENAI_MODEL
from langchain_openai import ChatOpenAI
from guardrails.prompt_injection import PromptInjectionGuard
from rag.semantic_cache import SemanticCache
from rag.embedding_cache import make_cached_embeddings


def build_llm(temperature: float, max_tokens: int, streaming: bool = False) -> ChatOpenAI:
    return ChatOpenAI(
        model=OPENAI_MODEL,
        api_key=OPENAI_API_KEY,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=streaming,
    )


def check_injection(prompt: str) -> tuple[bool, float]:
    llm = ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY, temperature=0.0, max_tokens=5)
    return PromptInjectionGuard(llm=llm).check(prompt)


@st.cache_resource
def get_semantic_cache() -> SemanticCache:
    return SemanticCache(embeddings=make_cached_embeddings())
