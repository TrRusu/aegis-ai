import streamlit as st
from agents.multimodal_agent import MultimodalAgent
from agents.breach_triage_agent import BreachTriageAgent
from ui.components.chat import render_chat_history, append_message, render_response
from ui.components.sidebar import render_model_params, render_k_slider, render_sidebar_footer
from ui.components.tool_calls import render_tool_calls
from ui.shared import check_injection, build_llm

st.set_page_config(page_title="Multimodal — Aegis", page_icon="🛡️", layout="centered")
st.title("🛡️ Multimodal")

with st.sidebar:
    temperature, max_tokens = render_model_params()
    st.divider()
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
            image_bytes = uploaded_image.read() if uploaded_image else None
            mime_type = uploaded_image.type if uploaded_image else "image/png"

            if image_bytes:
                with st.spinner("Analysing uploaded image..."):
                    _llm = build_llm(temperature=0.0, max_tokens=1024)
                    enriched = MultimodalAgent(llm=_llm).enrich(prompt, image_bytes, mime_type)
                with st.expander("Enriched incident description"):
                    st.caption("Original: " + prompt)
                    st.markdown("**After image analysis:**")
                    st.write(enriched)
            else:
                enriched = prompt

            with st.spinner("Triage agent investigating..."):
                _llm = build_llm(temperature=temperature, max_tokens=max_tokens)
                response, tool_calls_log = BreachTriageAgent(llm=_llm).run(enriched, k=multimodal_k)
            render_response(response)
            render_tool_calls(tool_calls_log, label="Agent tool calls")

    append_message("assistant", response)
