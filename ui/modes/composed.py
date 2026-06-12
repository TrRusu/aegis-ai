import streamlit as st
from agents.composed_workflow import ComposedWorkflow
from ui.shared import build_llm
from ui.components.chat import render_response
from ui.components.tool_calls import render_tool_calls


def handle(prompt: str, temperature: float, max_tokens: int, composed_k: int) -> str:
    with st.spinner("Running composed workflow: Parallel → Conditional → Synthesize..."):
        _llm = build_llm(temperature=temperature, max_tokens=max_tokens)
        response, tool_calls_log = ComposedWorkflow(llm=_llm).run(prompt, k=composed_k)
    render_response(response)
    render_tool_calls(tool_calls_log, label="Composed workflow tool calls")
    return response
