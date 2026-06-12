import streamlit as st
from agents.breach_workflow import BreachWorkflow
from ui.shared import build_llm
from ui.components.chat import render_response
from ui.components.tool_calls import render_tool_calls


def handle(prompt: str, temperature: float, max_tokens: int, workflow_k: int) -> str:
    with st.spinner("Running breach workflow: Assess → Research → Report..."):
        _llm = build_llm(temperature=temperature, max_tokens=max_tokens)
        response, tool_calls_log = BreachWorkflow(llm=_llm).run(prompt, k=workflow_k)
    render_response(response)
    render_tool_calls(tool_calls_log, label="Workflow tool calls (Research Agent)")
    return response
