import streamlit as st
from tools.chain import ToolChain
from tools.tools import make_tools
from ui.components.chat import render_chat_history, append_message, render_response
from ui.components.sidebar import render_model_params, render_k_slider, render_sidebar_footer
from ui.components.tool_calls import render_tool_calls
from ui.shared import check_injection, build_llm

st.set_page_config(page_title="Tools — Aegis", page_icon="🛡️", layout="centered")
st.title("🛡️ Tools")

@st.cache_resource
def _get_tools():
    return make_tools()

with st.sidebar:
    temperature, max_tokens = render_model_params()
    st.divider()
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
            with st.spinner("Aegis is thinking..."):
                _llm = build_llm(temperature=temperature, max_tokens=max_tokens)
                response, tool_calls_log = ToolChain(llm=_llm).run(prompt, st.session_state.messages, tools=make_tools(k=tools_k))
            render_response(response)
            render_tool_calls(tool_calls_log, label="Tool calls")

    append_message("assistant", response)
