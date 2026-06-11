from langchain_core.tools import tool
from rag.retriever import build_retriever


class ToolSet:
    """Builds LangChain tools with an injected retriever."""

    def __init__(self, retriever):
        self._retriever = retriever

    def build(self):
        retriever = self._retriever

        @tool
        def search_knowledge_base(query: str) -> str:
            """Search the selected file for information relevant to the query.
               Use this when you need specific statistics, findings, or details from the report.
            """
            docs = retriever.invoke(query)
            if not docs:
                return "No relevant information found in the knowledge base."
            results = []
            for i, doc in enumerate(docs, 1):
                page = doc.metadata.get("page", "?")
                results.append(f"[Chunk {i} — page {page}]\n{doc.page_content}")
            return "\n\n---\n\n".join(results)

        @tool
        def calculate_breach_cost(records_lost: int, cost_per_record: float) -> str:
            """Estimate a breach cost by multiplying records lost by a per-record cost."""
            total = records_lost * cost_per_record
            return (
                f"Estimated breach cost: USD {total:,.2f} "
                f"({records_lost:,} records x USD {cost_per_record:.2f}/record). "
                f"Note: IBM cautions against this multiplication method for estimating real breach costs."
            )

        return [search_knowledge_base, calculate_breach_cost]


def make_tools(k: int = 4):
    retriever = build_retriever(k=k, hybrid=True)
    return ToolSet(retriever=retriever).build()
