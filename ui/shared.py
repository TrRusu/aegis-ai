from app.config import OPENAI_API_KEY, OPENAI_MODEL
from langchain_openai import ChatOpenAI
from guardrails.prompt_injection import PromptInjectionGuard


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
