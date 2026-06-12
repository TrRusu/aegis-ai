import streamlit as st
from agents.supervisor_workflow import SupervisorWorkflow
from ui.components.chat import render_chat_history, append_message, render_response
from ui.components.sidebar import render_model_params, render_k_slider, render_sidebar_footer
from ui.components.tool_calls import render_tool_calls
from ui.shared import check_injection, build_llm

st.set_page_config(page_title="Supervisor — Aegis", page_icon="🛡️", layout="centered")
st.title("🛡️ Supervisor")

with st.sidebar:
    temperature, max_tokens = render_model_params()
    st.divider()
    st.header("Supervisor Settings")
    supervisor_k = render_k_slider(default=6, help_text="How many chunks the specialist agents retrieve per search.")
    st.divider()
    st.info("An LLM supervisor autonomously decides which specialist agents to invoke — CVE Analyst, Cost Analyst, Compliance Analyst — based on what the incident actually requires. No hardcoded routing.")
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
            with st.spinner("Supervisor is deciding which specialists to invoke..."):
                _llm = build_llm(temperature=temperature, max_tokens=max_tokens)
                response, tool_calls_log = SupervisorWorkflow(llm=_llm).run(prompt, k=supervisor_k)
            render_response(response)
            render_tool_calls(tool_calls_log, label="Supervisor — specialist agents invoked")

    append_message("assistant", response)
