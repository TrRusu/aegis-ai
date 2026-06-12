import streamlit as st


def render_chat_history() -> None:
    for msg in st.session_state.get("messages", []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def append_message(role: str, content: str) -> None:
    st.session_state.messages.append({"role": role, "content": content})


def render_response(response: str) -> None:
    st.markdown(response.replace("$", r"\$"))
