import streamlit as st
from agents.a2a_client import A2AClient
from app.config import A2A_SERVER_URL
from ui.components.chat import render_chat_history, append_message
from ui.components.sidebar import render_sidebar_footer
from ui.shared import check_injection

st.set_page_config(page_title="A2A — Aegis", page_icon="🛡️", layout="centered")
st.title("🛡️ A2A — Remote Agent")

_a2a_client = A2AClient(base_url=A2A_SERVER_URL)

with st.sidebar:
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
        elif not _a2a_client.is_server_available():
            response = "Remote Threat Intelligence Agent is offline. Start it with: `python a2a_server/threat_intel_server.py`"
            st.error(response)
        else:
            with st.spinner("Calling remote Threat Intelligence Agent at localhost:8888..."):
                try:
                    response = _a2a_client.call_threat_intel_agent(prompt)
                    st.markdown(response)
                    st.caption("Analysis provided by remote ThreatIntelAgent via A2A protocol.")
                except Exception as exc:
                    response = f"A2A call failed: {exc}"
                    st.error(response)

    append_message("assistant", response)
