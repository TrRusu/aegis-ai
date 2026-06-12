import os
import streamlit as st


def render_model_params() -> tuple[float, int]:
    st.header("Model Parameters")
    temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.2, step=0.1)
    max_tokens = st.slider("Max Tokens", min_value=128, max_value=4096, value=1024, step=128)
    return temperature, max_tokens


def render_k_slider(default: int = 4, help_text: str = "How many document chunks to retrieve per query.") -> int:
    return st.slider("Chunks to retrieve (k)", min_value=1, max_value=10, value=default, help=help_text)


def render_sidebar_footer() -> None:
    st.divider()
    st.caption(f"Model: `{os.getenv('OPENAI_MODEL', 'gpt-4o')}`")
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
