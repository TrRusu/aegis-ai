import os
import streamlit as st
from app.rag_chain import RagChain
from ui.shared import build_llm
from ui.components.tool_calls import render_tool_calls


def handle(prompt: str, history: list, temperature: float, max_tokens: int, selected_docs: list, k: int, hybrid: bool) -> str:
    if not selected_docs:
        st.warning("No documents selected. Please upload and select a document in the sidebar.")
        return "No documents selected. Please upload and select a document in the sidebar."

    _llm = build_llm(temperature=temperature, max_tokens=max_tokens, streaming=True)
    stream, source_docs = RagChain(llm=_llm).run(
        user_input=prompt,
        history=history,
        temperature=temperature,
        max_tokens=max_tokens,
        selected_docs=selected_docs,
        k=k,
        hybrid=hybrid,
    )
    response = st.write_stream(chunk.content for chunk in stream)
    with st.expander("Sources retrieved from knowledge base"):
        for i, doc in enumerate(source_docs, 1):
            page = doc.metadata.get("page", "?")
            source = os.path.basename(doc.metadata.get("source", "unknown"))
            st.markdown(f"**Chunk {i} — {source}, page {page}**")
            st.caption(doc.page_content[:400] + "...")
    return response
