import streamlit as st
from agents.breach_triage_agent import BreachTriageAgent
from ui.components.chat import render_chat_history, append_message, render_response
from ui.components.sidebar import render_model_params, render_k_slider, render_sidebar_footer
from ui.components.tool_calls import render_tool_calls
from ui.shared import check_injection, build_llm

st.set_page_config(page_title="Agent — Aegis", page_icon="🛡️", layout="centered")
st.title("🛡️ Agent")

with st.sidebar:
    temperature, max_tokens = render_model_params()
    st.divider()
    st.header("Triage Agent Settings")
    agent_k = render_k_slider(default=6, help_text="How many document chunks the agent retrieves per search.")
    st.divider()
    st.info("The Breach Triage Agent autonomously investigates incidents — searching the knowledge base, looking up CVEs, and estimating costs without further direction.")
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
        else:
            with st.spinner("Aegis Triage Agent is investigating..."):
                _llm = build_llm(temperature=temperature, max_tokens=max_tokens)
                response, tool_calls_log = BreachTriageAgent(llm=_llm).run(prompt, k=agent_k)
            render_response(response)
            render_tool_calls(tool_calls_log, label="Agent tool calls")

    append_message("assistant", response)
