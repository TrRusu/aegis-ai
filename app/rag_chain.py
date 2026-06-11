"""RAG pipeline with an injected LLM and retriever.
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from rag.retriever import build_retriever
from observability.logging_setup import CallLogger
from observability.fault_tolerance import FALLBACK_MESSAGE
from prompts.rag_chain import SYSTEM_PROMPT

_call_logger = CallLogger()


class RagChain:

    def __init__(self, llm: BaseChatModel):
        self._llm = llm

    @_call_logger.log_llm_call("RAG")
    def run(
        self,
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

            messages = [SystemMessage(content=SYSTEM_PROMPT.format(context=context))]
            for msg in history[:-1]:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))
            messages.append(HumanMessage(content=user_input))

            return self._llm.stream(messages), docs

        except Exception:
            return iter([type("_Chunk", (), {"content": FALLBACK_MESSAGE})()]), []
