import streamlit as st
from agents.supervisor_workflow import SupervisorWorkflow
from ui.shared import build_llm
from ui.components.chat import render_response
from ui.components.tool_calls import render_tool_calls


def handle(prompt: str, temperature: float, max_tokens: int, supervisor_k: int) -> str:
    with st.spinner("Supervisor is deciding which specialists to invoke..."):
        _llm = build_llm(temperature=temperature, max_tokens=max_tokens)
        response, tool_calls_log = SupervisorWorkflow(llm=_llm).run(prompt, k=supervisor_k)
    render_response(response)
    render_tool_calls(tool_calls_log, label="Supervisor — specialist agents invoked")
    return response
