"""Parallel + conditional + synthesis workflow with an injected LLM.
"""

import asyncio
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import TypedDict, Literal
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langchain_mcp_adapters.client import MultiServerMCPClient
from tools.tools import make_tools
from app.config import CVE_SERVER_PATH
from observability.logging_setup import CallLogger, logger

_call_logger = CallLogger()
from observability.fault_tolerance import FALLBACK_MESSAGE
from app.utils import extract_text
from langchain.agents import create_agent
from prompts.composed_workflow import THREAT_PROMPT, COMPLIANCE_PROMPT, CRITICAL_RESPONSE_PROMPT, STANDARD_RESPONSE_PROMPT, SYNTHESIS_PROMPT


class ComposedWorkflowState(TypedDict):
    """State schema for the composed workflow."""
    incident: str
    threat_analysis: str
    compliance_analysis: str
    severity: str
    response_plan: str
    report: str
    tool_calls_log: list


class ComposedWorkflow:
    """Parallel + conditional + synthesis workflow with an injected LLM."""

    def __init__(self, llm: BaseChatModel):
        self._llm = llm

    @_call_logger.log_llm_call("ComposedWorkflow")
    def run(self, incident: str, k: int = 6) -> tuple[str, list[dict]]:
        try:
            return asyncio.run(self._run_async(incident, k))
        except Exception as exc:
            logger.error(f"[ComposedWorkflow] Failed — {type(exc).__name__}: {exc}")
            return FALLBACK_MESSAGE, []

    async def _run_async(self, incident: str, k: int) -> tuple[str, list[dict]]:
        local_tools = make_tools(k=k)
        mcp_client = MultiServerMCPClient({
            "cve": {"command": "python", "args": [CVE_SERVER_PATH], "transport": "stdio"}
        })
        mcp_tools = await mcp_client.get_tools()
        all_tools = local_tools + mcp_tools

        llm = self._llm

        async def parallel_analysis_node(state: ComposedWorkflowState) -> dict:
            logger.info("[Composed] Node 1/4 — Parallel: ThreatAgent + ComplianceAgent")
            threat_resp, compliance_resp = await asyncio.gather(
                asyncio.to_thread(llm.invoke, [SystemMessage(content=THREAT_PROMPT), HumanMessage(content=state["incident"])]),
                asyncio.to_thread(llm.invoke, [SystemMessage(content=COMPLIANCE_PROMPT), HumanMessage(content=state["incident"])]),
            )
            threat = extract_text(threat_resp.content)
            compliance = extract_text(compliance_resp.content)
            severity = "Medium"
            for line in threat.splitlines():
                if line.lower().startswith("severity:"):
                    severity = line.split(":", 1)[1].strip()
                    break
            return {"threat_analysis": threat, "compliance_analysis": compliance, "severity": severity}

        def severity_router(state: ComposedWorkflowState) -> Literal["critical", "standard"]:
            s = state["severity"].lower()
            if "critical" in s or "high" in s:
                logger.info("[Composed] Router → critical response path")
                return "critical"
            logger.info("[Composed] Router → standard response path")
            return "standard"

        async def _response_node(state: ComposedWorkflowState, prompt: str, label: str) -> dict:
            logger.info(f"[Composed] {label}")
            agent = create_agent(llm, all_tools, prompt=prompt)
            task = (
                f"Incident:\n{state['incident']}\n\n"
                f"Threat Analysis:\n{state['threat_analysis']}\n\n"
                f"Compliance Analysis:\n{state['compliance_analysis']}"
            )
            result = await agent.ainvoke({"messages": [HumanMessage(content=task)]})
            response = extract_text(result["messages"][-1].content)
            log = list(state.get("tool_calls_log", []))
            for msg in result["messages"]:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        log.append({"tool": tc["name"], "input": tc["args"], "output": ""})
                if hasattr(msg, "name") and msg.name:
                    for entry in reversed(log):
                        if entry["tool"] == msg.name and entry["output"] == "":
                            entry["output"] = extract_text(msg.content)
                            break
            return {"response_plan": response, "tool_calls_log": log}

        async def critical_response_node(state): return await _response_node(state, CRITICAL_RESPONSE_PROMPT, "Node 2/4 — CriticalResponseAgent")
        async def standard_response_node(state): return await _response_node(state, STANDARD_RESPONSE_PROMPT, "Node 2/4 — StandardResponseAgent")

        def synthesis_node(state: ComposedWorkflowState) -> dict:
            logger.info("[Composed] Node 4/4 — SynthesisAgent")
            task = (
                f"Incident:\n{state['incident']}\n\n"
                f"Threat Analysis:\n{state['threat_analysis']}\n\n"
                f"Compliance Analysis:\n{state['compliance_analysis']}\n\n"
                f"Response Plan (severity: {state['severity']}):\n{state['response_plan']}"
            )
            response = llm.invoke([SystemMessage(content=SYNTHESIS_PROMPT), HumanMessage(content=task)])
            return {"report": extract_text(response.content)}

        graph = StateGraph(ComposedWorkflowState)
        graph.add_node("parallel_analysis", parallel_analysis_node)
        graph.add_node("critical_response", critical_response_node)
        graph.add_node("standard_response", standard_response_node)
        graph.add_node("synthesize", synthesis_node)
        graph.add_edge(START, "parallel_analysis")
        graph.add_conditional_edges(
            "parallel_analysis", severity_router,
            {"critical": "critical_response", "standard": "standard_response"},
        )
        graph.add_edge("critical_response", "synthesize")
        graph.add_edge("standard_response", "synthesize")
        graph.add_edge("synthesize", END)

        workflow = graph.compile()
        initial_state: ComposedWorkflowState = {
            "incident": incident, "threat_analysis": "", "compliance_analysis": "",
            "severity": "", "response_plan": "", "report": "", "tool_calls_log": [],
        }
        final_state = await workflow.ainvoke(initial_state)
        return final_state["report"], final_state["tool_calls_log"]