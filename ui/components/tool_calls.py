import streamlit as st


def render_tool_calls(tool_calls_log: list[dict], label: str = "Tool calls") -> None:
    with st.expander(label):
        if not tool_calls_log:
            st.caption("No tool calls made.")
            return
        for call in tool_calls_log:
            st.markdown(f"**Tool:** `{call['tool']}`")
            st.markdown(f"**Input:** {call['input']}")
            st.markdown(f"**Output:** {call['output']}")
            st.divider()
