import streamlit as st
from agents.multimodal_agent import MultimodalAgent
from agents.breach_triage_agent import BreachTriageAgent
from ui.shared import build_llm
from ui.components.chat import render_response
from ui.components.tool_calls import render_tool_calls


def handle(prompt: str, temperature: float, max_tokens: int, multimodal_k: int, uploaded_image) -> str:
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
    return response
