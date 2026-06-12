import os
import streamlit as st
from langchain_openai import ChatOpenAI
from app.config import APP_NAME, KNOWLEDGE_BASE_DIR, OPENAI_API_KEY, OPENAI_MODEL, A2A_SERVER_URL
from agents.a2a_client import A2AClient
from rag.ingestion import make_store
from rag.document_loader import BasicPdfLoader, EnhancedPdfLoader
from tools.tools import make_tools
from ui.shared import check_injection
from ui.components.chat import render_chat_history, append_message
from ui.components.sidebar import render_model_params, render_k_slider, render_sidebar_footer
import ui.modes.chat as mode_chat
import ui.modes.rag as mode_rag
import ui.modes.tools as mode_tools
import ui.modes.agent as mode_agent
import ui.modes.workflow as mode_workflow
import ui.modes.composed as mode_composed
import ui.modes.supervisor as mode_supervisor
import ui.modes.hitl as mode_hitl
import ui.modes.multimodal as mode_multimodal
import ui.modes.a2a as mode_a2a

_a2a_client = A2AClient(base_url=A2A_SERVER_URL)

@st.cache_resource
def _get_tools():
    return make_tools()

st.set_page_config(page_title=APP_NAME, page_icon="🛡️", layout="centered")
st.title(f"🛡️ {APP_NAME}")
st.caption("AI-Powered Data Breach Analyst Assistant")
st.divider()

with st.sidebar:
    temperature, max_tokens = render_model_params()
    st.divider()
    st.header("Mode")
    mode = st.radio("", ["Chat", "RAG", "Tools", "Agent", "Workflow", "Composed", "Supervisor", "HITL", "Multimodal", "A2A"], label_visibility="collapsed")

    selected_docs = []
    k = 4
    hybrid = False
    tools_k = 4
    agent_k = 6
    workflow_k = 6
    composed_k = 6
    supervisor_k = 6
    hitl_k = 6
    multimodal_k = 6
    uploaded_image = None

    if mode == "RAG":
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
                msg = make_store().ingest(save_path, loader, mode_label="enhanced" if enhanced else "basic")
            st.session_state.processed_uploads.add(uploaded.name)
            st.success(msg)
            st.rerun()
        if ingested:
            selected_docs = st.multiselect("Query against", options=ingested, default=ingested)
        else:
            st.caption("No documents ingested yet. Upload a PDF above.")
        st.divider()
        st.header("Retrieval Settings")
        retrieval_mode = st.radio("Retrieval mode", ["Vector only", "Hybrid (BM25 + Vector)"])
        hybrid = retrieval_mode == "Hybrid (BM25 + Vector)"
        k = st.slider("Chunks to retrieve (k)", min_value=1, max_value=10, value=4)

    elif mode == "Tools":
        st.header("Retrieval Settings")
        tools_k = render_k_slider(default=4, help_text="How many document chunks the search tool retrieves per query.")
        st.divider()
        st.header("Local Tools")
        for tool in _get_tools():
            st.markdown(f"**`{tool.name}`**")
            st.caption(tool.description)
        st.divider()
        st.header("MCP Tools")
        st.markdown("**`lookup_cve`**")
        st.caption("Looks up a CVE by ID from the National Vulnerability Database (NVD). Returns severity, CVSS score and description.")

    elif mode == "Agent":
        st.header("Triage Agent Settings")
        agent_k = render_k_slider(default=6, help_text="How many document chunks the agent retrieves per search.")
        st.divider()
        st.info("The Breach Triage Agent autonomously investigates incidents — searching the knowledge base, looking up CVEs, and estimating costs without further direction.")

    elif mode == "Workflow":
        st.header("Workflow Settings")
        workflow_k = render_k_slider(default=6, help_text="How many chunks the Research Agent retrieves per search.")
        st.divider()
        st.info("Sequential workflow: Assessment Agent → Research Agent → Report Agent. Each agent has a specific role and passes its output to the next.")

    elif mode == "Composed":
        st.header("Composed Workflow Settings")
        composed_k = render_k_slider(default=6, help_text="How many chunks the response agents retrieve per search.")
        st.divider()
        st.info("Parallel analysis (Threat + Compliance) → Conditional routing (Critical or Standard response) → Synthesis.")

    elif mode == "Supervisor":
        st.header("Supervisor Settings")
        supervisor_k = render_k_slider(default=6, help_text="How many chunks the specialist agents retrieve per search.")
        st.divider()
        st.info("An LLM supervisor autonomously decides which specialist agents to invoke — CVE Analyst, Cost Analyst, Compliance Analyst — based on what the incident actually requires. No hardcoded routing.")

    elif mode == "HITL":
        st.header("HITL Supervisor Settings")
        hitl_k = render_k_slider(default=6)
        st.divider()
        st.info("Critical/High severity incidents require human approval before the final report is produced. The workflow pauses after Phase 1 and resumes only after you approve or reject.")

    elif mode == "A2A":
        st.header("A2A — Remote Agent")
        st.divider()
        if _a2a_client.is_server_available():
            st.success("Threat Intelligence Agent online at localhost:8888")
            try:
                card = _a2a_client.fetch_agent_card()
                st.caption(f"Agent: **{card['name']}** v{card['version']}")
                for skill in card.get("skills", []):
                    st.markdown(f"**Skill:** {skill['name']}")
                    st.caption(skill["description"])
            except Exception:
                pass
        else:
            st.error("Remote agent offline. Start it with: `python a2a_server/threat_intel_server.py`")

    elif mode == "Multimodal":
        st.header("Multimodal Settings")
        multimodal_k = render_k_slider(default=6)
        st.divider()
        uploaded_image = st.file_uploader(
            "Upload a screenshot (optional)",
            type=["png", "jpg", "jpeg", "webp"],
            help="Upload a screenshot of a security alert, malware warning, or anomaly dashboard. The image is analysed first and its observations are merged into your incident description.",
        )
        st.info("Image analysis enriches the incident description before it reaches the triage agent. If no image is uploaded, the text description is used as-is.")

    render_sidebar_footer()

if "messages" not in st.session_state:
    st.session_state.messages = []

render_chat_history()

if mode == "HITL":
    mode_hitl.render_approval_card()

if prompt := st.chat_input("Ask Aegis about a threat, CVE, or incident..."):
    append_message("user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        is_injection, injection_score = check_injection(prompt)

        if is_injection:
            response = f"Message blocked — potential prompt injection detected (risk score: {injection_score:.2f}). Please rephrase your question."
            st.warning(response)
        elif mode == "Chat":
            response = mode_chat.handle(prompt, st.session_state.messages, temperature, max_tokens)
        elif mode == "RAG":
            response = mode_rag.handle(prompt, st.session_state.messages, temperature, max_tokens, selected_docs, k, hybrid)
        elif mode == "Tools":
            response = mode_tools.handle(prompt, st.session_state.messages, temperature, max_tokens, tools_k)
        elif mode == "Agent":
            response = mode_agent.handle(prompt, temperature, max_tokens, agent_k)
        elif mode == "Workflow":
            response = mode_workflow.handle(prompt, temperature, max_tokens, workflow_k)
        elif mode == "Composed":
            response = mode_composed.handle(prompt, temperature, max_tokens, composed_k)
        elif mode == "Supervisor":
            response = mode_supervisor.handle(prompt, temperature, max_tokens, supervisor_k)
        elif mode == "HITL":
            response = mode_hitl.handle(prompt, temperature, max_tokens, hitl_k)
        elif mode == "Multimodal":
            response = mode_multimodal.handle(prompt, temperature, max_tokens, multimodal_k, uploaded_image)
        elif mode == "A2A":
            response = mode_a2a.handle(prompt, _a2a_client)

    append_message("assistant", response)
    if st.session_state.pop("hitl_needs_rerun", False):
        st.rerun()
