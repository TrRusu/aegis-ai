import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from app.llm import build_llm, build_messages
from app.rag_chain import build_rag_response
from tools.chain import run_with_tools
from tools.tools import make_tools
TOOLS = make_tools()
from app.config import APP_NAME
from rag.ingestion import ingest_file, get_ingested_documents, KNOWLEDGE_BASE_DIR
from guardrails.prompt_injection import check_prompt_injection
from agents.breach_triage_agent import BreachTriageAgent
from agents.breach_workflow import run_workflow
from agents.composed_workflow import run_composed_workflow
from agents.supervisor_workflow import run_supervisor
from agents.supervisor_hitl import run_hitl_phase1, run_hitl_phase2, requires_approval
from agents.multimodal_agent import enrich_with_image
from agents.a2a_client import A2AClient
from app.config import A2A_SERVER_URL

_a2a_client = A2AClient(base_url=A2A_SERVER_URL)

st.set_page_config(page_title=APP_NAME, page_icon="🛡️", layout="centered")

st.title(f"🛡️ {APP_NAME}")
st.caption("AI-Powered Data Breach Analyst Assistant")
st.divider()

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Model Parameters")
    temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.2, step=0.1)
    max_tokens = st.slider("Max Tokens", min_value=128, max_value=4096, value=1024, step=128)

    st.divider()
    st.header("Mode")
    mode = st.radio("", ["Chat", "RAG", "Tools", "Agent", "Workflow", "Composed", "Supervisor", "HITL", "Multimodal", "A2A"], label_visibility="collapsed")

    if mode == "RAG":
        ingested = get_ingested_documents()
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
                msg = ingest_file(save_path, enhanced=enhanced)
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

    elif mode == "Tools":
        st.header("Retrieval Settings")
        tools_k = st.slider(
            "Chunks to retrieve (k)", min_value=1, max_value=10, value=4,
            help="How many document chunks the search tool retrieves per query."
        )
        st.divider()
        st.header("Local Tools")
        for tool in TOOLS:
            st.markdown(f"**`{tool.name}`**")
            st.caption(tool.description)
        st.divider()
        st.header("MCP Tools")
        st.markdown("**`lookup_cve`**")
        st.caption("Looks up a CVE by ID from the National Vulnerability Database (NVD). Returns severity, CVSS score and description.")

    elif mode == "Agent":
        st.header("Triage Agent Settings")
        agent_k = st.slider(
            "Chunks to retrieve (k)", min_value=1, max_value=10, value=6,
            help="How many document chunks the agent retrieves per search."
        )
        st.divider()
        st.info("The Breach Triage Agent autonomously investigates incidents — searching the knowledge base, looking up CVEs, and estimating costs without further direction.")

    elif mode == "Workflow":
        st.header("Workflow Settings")
        workflow_k = st.slider(
            "Chunks to retrieve (k)", min_value=1, max_value=10, value=6,
            help="How many chunks the Research Agent retrieves per search."
        )
        st.divider()
        st.info("Sequential workflow: Assessment Agent → Research Agent → Report Agent. Each agent has a specific role and passes its output to the next.")

    elif mode == "Composed":
        st.header("Composed Workflow Settings")
        composed_k = st.slider(
            "Chunks to retrieve (k)", min_value=1, max_value=10, value=6,
            help="How many chunks the response agents retrieve per search."
        )
        st.divider()
        st.info("Parallel analysis (Threat + Compliance) → Conditional routing (Critical or Standard response) → Synthesis.")

    elif mode == "Supervisor":
        st.header("Supervisor Settings")
        supervisor_k = st.slider(
            "Chunks to retrieve (k)", min_value=1, max_value=10, value=6,
            help="How many chunks the specialist agents retrieve per search."
        )
        st.divider()
        st.info("An LLM supervisor autonomously decides which specialist agents to invoke — CVE Analyst, Cost Analyst, Compliance Analyst — based on what the incident actually requires. No hardcoded routing.")

    elif mode == "HITL":
        st.header("HITL Supervisor Settings")
        hitl_k = st.slider(
            "Chunks to retrieve (k)", min_value=1, max_value=10, value=6,
        )
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
        multimodal_k = st.slider(
            "Chunks to retrieve (k)", min_value=1, max_value=10, value=6,
        )
        st.divider()
        uploaded_image = st.file_uploader(
            "Upload a screenshot (optional)",
            type=["png", "jpg", "jpeg", "webp"],
            help="Upload a screenshot of a security alert, malware warning, or anomaly dashboard. The image is analysed first and its observations are merged into your incident description.",
        )
        st.info("Image analysis enriches the incident description before it reaches the triage agent. If no image is uploaded, the text description is used as-is.")

    st.divider()
    st.caption(f"Model: `{os.getenv('OPENAI_MODEL', 'gpt-4o')}`")
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if mode != "Tools":
    tools_k = 4
if mode != "Agent":
    agent_k = 6
if mode != "Workflow":
    workflow_k = 6
if mode != "Composed":
    composed_k = 6
if mode != "Supervisor":
    supervisor_k = 6
if mode != "HITL":
    hitl_k = 6
if mode != "Multimodal":
    multimodal_k = 6
    uploaded_image = None

# ── CHAT ───────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── HITL approval card — rendered BEFORE st.chat_input so it appears on screen ──
if mode == "HITL" and "hitl_pending" in st.session_state:
    pending = st.session_state["hitl_pending"]
    st.divider()
    st.subheader("Pending Approval")
    st.caption(f"Severity: **{pending['severity']}** — review the Phase 1 briefing above then decide.")
    reason = st.text_input("Your reason (optional)", key="hitl_reason", placeholder="e.g. Confirmed with SOC team")
    col1, col2 = st.columns(2)
    if col1.button("Approve — proceed with full response", use_container_width=True, type="primary"):
        with st.spinner("Phase 2 — compiling approved response plan..."):
            final = run_hitl_phase2(
                incident=pending["incident"],
                preliminary_analysis=pending["preliminary"],
                decision="APPROVED",
                reason=reason or "Approved by analyst.",
            )
        st.session_state.messages.append({"role": "assistant", "content": final})
        del st.session_state["hitl_pending"]
        st.rerun()
    if col2.button("Reject — use conservative fallback", use_container_width=True):
        with st.spinner("Phase 2 — compiling rejection response..."):
            final = run_hitl_phase2(
                incident=pending["incident"],
                preliminary_analysis=pending["preliminary"],
                decision="REJECTED",
                reason=reason or "Rejected by analyst.",
            )
        st.session_state.messages.append({"role": "assistant", "content": final})
        del st.session_state["hitl_pending"]
        st.rerun()
    st.divider()

if prompt := st.chat_input("Ask Aegis about a threat, CVE, or incident..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        is_injection, injection_score = check_prompt_injection(prompt)

        if is_injection:
            response = f"Message blocked — potential prompt injection detected (risk score: {injection_score:.2f}). Please rephrase your question."
            st.warning(response)

        elif mode == "Chat":
            llm = build_llm(temperature=temperature, max_tokens=max_tokens)
            messages = build_messages(st.session_state.messages)
            response = st.write_stream(chunk.content for chunk in llm.stream(messages))

        elif mode == "RAG":
            if not selected_docs:
                response = "No documents selected. Please upload and select a document in the sidebar."
                st.warning(response)
            else:
                stream, source_docs = build_rag_response(
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

        elif mode == "Tools":
            with st.spinner("Aegis is thinking..."):
                response, tool_calls_log = run_with_tools(
                    user_input=prompt,
                    history=st.session_state.messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    k=tools_k,
                )
            st.markdown(response.replace("$", r"\$"))

            if tool_calls_log:
                with st.expander("Tool calls"):
                    for call in tool_calls_log:
                        st.markdown(f"**Tool:** `{call['tool']}`")
                        st.markdown(f"**Input:** {call['input']}")
                        st.markdown(f"**Output:** {call['output']}")
                        st.divider()

        elif mode == "Agent":
            with st.spinner("Aegis Triage Agent is investigating..."):
                from langchain_openai import ChatOpenAI
                from app.config import OPENAI_API_KEY, OPENAI_MODEL
                _llm = ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY, temperature=temperature, max_tokens=max_tokens)
                response, tool_calls_log = BreachTriageAgent(llm=_llm).run(prompt, k=agent_k)
            st.markdown(response.replace("$", r"\$"))

            if tool_calls_log:
                with st.expander("Agent tool calls"):
                    for call in tool_calls_log:
                        st.markdown(f"**Tool:** `{call['tool']}`")
                        st.markdown(f"**Input:** {call['input']}")
                        st.markdown(f"**Output:** {call['output']}")
                        st.divider()

        elif mode == "Workflow":
            with st.spinner("Running breach workflow: Assess → Research → Report..."):
                response, tool_calls_log = run_workflow(
                    incident=prompt,
                    k=workflow_k,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            st.markdown(response.replace("$", r"\$"))

            if tool_calls_log:
                with st.expander("Workflow tool calls (Research Agent)"):
                    for call in tool_calls_log:
                        st.markdown(f"**Tool:** `{call['tool']}`")
                        st.markdown(f"**Input:** {call['input']}")
                        st.markdown(f"**Output:** {call['output']}")
                        st.divider()

        elif mode == "Supervisor":
            with st.spinner("Supervisor is deciding which specialists to invoke..."):
                response, tool_calls_log = run_supervisor(
                    incident=prompt,
                    k=supervisor_k,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            st.markdown(response.replace("$", r"\$"))

            with st.expander("Supervisor — specialist agents invoked"):
                if tool_calls_log:
                    for call in tool_calls_log:
                        st.markdown(f"**Specialist:** `{call['tool']}`")
                        st.markdown(f"**Input:** {call['input']}")
                        st.markdown(f"**Output:** {call['output']}")
                        st.divider()
                else:
                    st.caption("Supervisor determined no specialist agents were needed for this incident.")

        elif mode == "Composed":
            with st.spinner("Running composed workflow: Parallel → Conditional → Synthesize..."):
                response, tool_calls_log = run_composed_workflow(
                    incident=prompt,
                    k=composed_k,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            st.markdown(response.replace("$", r"\$"))

            with st.expander("Composed workflow tool calls"):
                if tool_calls_log:
                    for call in tool_calls_log:
                        st.markdown(f"**Tool:** `{call['tool']}`")
                        st.markdown(f"**Input:** {call['input']}")
                        st.markdown(f"**Output:** {call['output']}")
                        st.divider()
                else:
                    st.caption("No tool calls — agent determined the knowledge base was not needed for this incident.")

        elif mode == "HITL":
            with st.spinner("Phase 1 — gathering intelligence..."):
                preliminary, severity, tool_calls_log = run_hitl_phase1(
                    incident=prompt,
                    k=hitl_k,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

            if requires_approval(severity):
                response = f"**Phase 1 complete — Severity: {severity}**\n\nThis incident requires human approval before proceeding.\n\n---\n\n{preliminary}"
                st.warning(f"Severity **{severity}** — human approval required before final report.")
                st.markdown(preliminary.replace("$", r"\$"))
                st.session_state["hitl_pending"] = {
                    "incident": prompt,
                    "preliminary": preliminary,
                    "tool_calls_log": tool_calls_log,
                    "severity": severity,
                }
                st.session_state["hitl_needs_rerun"] = True
            else:
                with st.spinner("Phase 2 — compiling final report (no approval needed)..."):
                    final = run_hitl_phase2(
                        incident=prompt,
                        preliminary_analysis=preliminary,
                        decision="AUTO-APPROVED",
                        reason=f"Severity {severity} is below approval threshold.",
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                response = final
                st.markdown(final.replace("$", r"\$"))

            with st.expander("Phase 1 specialist agents"):
                if tool_calls_log:
                    for call in tool_calls_log:
                        st.markdown(f"**Specialist:** `{call['tool']}`")
                        st.markdown(f"**Input:** {call['input']}")
                        st.markdown(f"**Output:** {call['output']}")
                        st.divider()
                else:
                    st.caption("No specialist agents invoked.")

        elif mode == "A2A":
            if not _a2a_client.is_server_available():
                response = "Remote Threat Intelligence Agent is offline. Start it with: `python a2a_server/threat_intel_server.py`"
                st.error(response)
            else:
                with st.spinner("Calling remote Threat Intelligence Agent at localhost:8888..."):
                    try:
                        analysis = _a2a_client.call_threat_intel_agent(prompt)
                        response = analysis
                        st.markdown(response)
                        st.caption("Analysis provided by remote ThreatIntelAgent via A2A protocol.")
                    except Exception as exc:
                        response = f"A2A call failed: {exc}"
                        st.error(response)

        elif mode == "Multimodal":
            image_bytes = uploaded_image.read() if uploaded_image else None
            mime_type = uploaded_image.type if uploaded_image else "image/png"

            if image_bytes:
                with st.spinner("Analysing uploaded image..."):
                    enriched = enrich_with_image(prompt, image_bytes, mime_type)
                with st.expander("Enriched incident description"):
                    st.caption("Original: " + prompt)
                    st.markdown("**After image analysis:**")
                    st.write(enriched)
            else:
                enriched = prompt

            with st.spinner("Triage agent investigating..."):
                from langchain_openai import ChatOpenAI
                from app.config import OPENAI_API_KEY, OPENAI_MODEL
                _llm = ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY, temperature=temperature, max_tokens=max_tokens)
                response, tool_calls_log = BreachTriageAgent(llm=_llm).run(enriched, k=multimodal_k)
            st.markdown(response.replace("$", r"\$"))

            if tool_calls_log:
                with st.expander("Agent tool calls"):
                    for call in tool_calls_log:
                        st.markdown(f"**Tool:** `{call['tool']}`")
                        st.markdown(f"**Input:** {call['input']}")
                        st.markdown(f"**Output:** {call['output']}")
                        st.divider()

    st.session_state.messages.append({"role": "assistant", "content": response})
    if st.session_state.pop("hitl_needs_rerun", False):
        st.rerun()
