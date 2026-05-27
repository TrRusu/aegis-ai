from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from rag.retriever import build_retriever
from app.config import OPENAI_API_KEY, OPENAI_MODEL
from observability.logging_setup import log_llm_call
from observability.fault_tolerance import FALLBACK_MESSAGE

RAG_SYSTEM_PROMPT = """You are Aegis, an AI-powered cybersecurity analyst assistant specializing in data breach analysis.
Answer the user's question using ONLY the context excerpts provided below.
If the context does not contain enough information to answer, say so clearly — do not fall back on general knowledge.
Always be precise and cite specific figures or findings from the context when available.

Context:
{context}
"""


@log_llm_call("RAG")
def build_rag_response(
    user_input: str,
    history: list[dict],
    temperature: float,
    max_tokens: int,
    selected_docs: list[str] | None = None,
    k: int = 4,
    hybrid: bool = False,
):
    try:
        retriever = build_retriever(k=k, selected_docs=selected_docs, hybrid=hybrid)
        docs = retriever.invoke(user_input)
        context = "\n\n---\n\n".join(doc.page_content for doc in docs)

        messages = [SystemMessage(content=RAG_SYSTEM_PROMPT.format(context=context))]
        for msg in history[:-1]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        messages.append(HumanMessage(content=user_input))

        llm = ChatOpenAI(
            model=OPENAI_MODEL,
            api_key=OPENAI_API_KEY,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=True,
        )
        return llm.stream(messages), docs

    except Exception:
        return iter([type("_Chunk", (), {"content": FALLBACK_MESSAGE})()]), []
