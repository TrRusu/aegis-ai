import streamlit as st
from app.llm import build_llm, build_messages
from ui.components.chat import render_chat_history, append_message, render_response
from ui.components.sidebar import render_model_params, render_sidebar_footer
from ui.shared import check_injection

st.set_page_config(page_title="Chat — Aegis", page_icon="🛡️", layout="centered")
st.title("🛡️ Chat")

with st.sidebar:
    temperature, max_tokens = render_model_params()
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
            llm = build_llm(temperature=temperature, max_tokens=max_tokens)
            messages = build_messages(st.session_state.messages)
            response = st.write_stream(chunk.content for chunk in llm.stream(messages))

    append_message("assistant", response)
