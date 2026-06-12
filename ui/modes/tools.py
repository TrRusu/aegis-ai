import streamlit as st
from tools.chain import ToolChain
from tools.tools import make_tools
from ui.shared import build_llm
from ui.components.chat import render_response
from ui.components.tool_calls import render_tool_calls


def handle(prompt: str, history: list, temperature: float, max_tokens: int, tools_k: int) -> str:
    with st.spinner("Aegis is thinking..."):
        _llm = build_llm(temperature=temperature, max_tokens=max_tokens)
        response, tool_calls_log = ToolChain(llm=_llm).run(prompt, history, tools=make_tools(k=tools_k))
    render_response(response)
    render_tool_calls(tool_calls_log, label="Tool calls")
    return response
