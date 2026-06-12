import streamlit as st
from agents.breach_triage_agent import BreachTriageAgent
from ui.shared import build_llm
from ui.components.chat import render_response
from ui.components.tool_calls import render_tool_calls


def handle(prompt: str, temperature: float, max_tokens: int, agent_k: int) -> str:
    with st.spinner("Aegis Triage Agent is investigating..."):
        _llm = build_llm(temperature=temperature, max_tokens=max_tokens)
        response, tool_calls_log = BreachTriageAgent(llm=_llm).run(prompt, k=agent_k)
    render_response(response)
    render_tool_calls(tool_calls_log, label="Agent tool calls")
    return response
