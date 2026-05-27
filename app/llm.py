from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from app.config import OPENAI_API_KEY, OPENAI_MODEL

SYSTEM_PROMPT = """You are Aegis, an AI-powered cybersecurity analyst assistant specializing in data breach analysis.
You help analysts understand breach patterns, costs, root causes, attack vectors, and industry-specific risk factors.
Your knowledge is grounded in data breach research and reports such as the IBM Cost of a Data Breach Report.

Guidelines:
- Be precise and data-driven. Reference statistics and findings when available.
- Always contextualize breach costs and timelines by industry or attack vector when relevant.
- Never provide instructions that could be used to carry out or facilitate a breach.
- If you don't know something or it falls outside your knowledge base, say so clearly rather than guessing.
"""

def build_llm(temperature: float = 0.2, max_tokens: int = 1024) -> ChatOpenAI:
    return ChatOpenAI(
        model=OPENAI_MODEL,
        api_key=OPENAI_API_KEY,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=True,
    )

def build_messages(history: list[dict]) -> list:
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    return messages
