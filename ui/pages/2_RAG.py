import os
import streamlit as st
from langchain_openai import ChatOpenAI
from app.config import APP_NAME, KNOWLEDGE_BASE_DIR, OPENAI_API_KEY, OPENAI_MODEL
from app.rag_chain import RagChain
from rag.ingestion import make_store
from rag.document_loader import BasicPdfLoader, EnhancedPdfLoader
from ui.components.chat import render_chat_history, append_message
from ui.components.sidebar import render_model_params, render_sidebar_footer
from ui.shared import check_injection, build_llm

st.set_page_config(page_title="RAG — Aegis", page_icon="🛡️", layout="centered")
st.title("🛡️ RAG")

with st.sidebar:
    temperature, max_tokens = render_model_params()
    st.divider()

    ingested = make_store().get_ingested_documents()
    st.header("Knowledge Base")

    ingestion_mode = st.radio(
        "Ingestion mode",
        ["Basic (PyPDF)", "Enhanced (Unstructured + GPT-4o)"],
        help="Enhanced reads charts and tables via GPT-4o vision. Basic is faster but misses image-based data.",
    )
    enhanced = ingestion_mode == "Enhanced (Unstructured + GPT-4o)"

    if not enhanced:
        chunk_size = st.slider("Chunk size", min_value=200, max_value=2000, value=1000, step=100)
        chunk_overlap = st.slider("Chunk overlap", min_value=0, max_value=400, value=150, step=25)
    else:
        chunk_size, chunk_overlap = 1000, 150

    if "processed_uploads" not in st.session_state:
        st.session_state.processed_uploads = set()

    uploaded = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded and uploaded.name not in st.session_state.processed_uploads:
        save_path = os.path.join(KNOWLEDGE_BASE_DIR, uploaded.name)
        with open(save_path, "wb") as f:
            f.write(uploaded.getbuffer())
        spinner_msg = f"Ingesting {uploaded.name} (Unstructured + GPT-4o, may take several minutes)..." if enhanced else f"Ingesting {uploaded.name}..."
        with st.spinner(spinner_msg):
            if enhanced:
                _ingest_llm = ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY, temperature=0.0, max_tokens=2048)
                loader = EnhancedPdfLoader(llm=_ingest_llm, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            else:
                loader = BasicPdfLoader(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            mode_label = "enhanced" if enhanced else "basic"
            msg = make_store().ingest(save_path, loader, mode_label=mode_label)
        st.session_state.processed_uploads.add(uploaded.name)
        st.success(msg)
        st.rerun()

    if ingested:
        selected_docs = st.multiselect("Query against", options=ingested, default=ingested)
    else:
        selected_docs = []
        st.caption("No documents ingested yet. Upload a PDF above.")

    st.divider()
    st.header("Retrieval Settings")
    retrieval_mode = st.radio("Retrieval mode", ["Vector only", "Hybrid (BM25 + Vector)"])
    hybrid = retrieval_mode == "Hybrid (BM25 + Vector)"
    k = st.slider("Chunks to retrieve (k)", min_value=1, max_value=10, value=4)

    render_sidebar_footer()

if "messages" not in st.session_state:
    st.session_state.messages = []

render_chat_history()

if prompt := st.chat_input("Ask Aegis about a threat, CVE, or incident..."):
    append_message("user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        is_injection, injection_score = check_injection(prompt)

        if is_injection:
            response = f"Message blocked — potential prompt injection detected (risk score: {injection_score:.2f}). Please rephrase your question."
            st.warning(response)
        elif not selected_docs:
            response = "No documents selected. Please upload and select a document in the sidebar."
            st.warning(response)
        else:
            _llm = build_llm(temperature=temperature, max_tokens=max_tokens, streaming=True)
            stream, source_docs = RagChain(llm=_llm).run(
                user_input=prompt,
                history=st.session_state.messages,
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

    append_message("assistant", response)
