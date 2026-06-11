"""Three-node sequential breach analysis workflow with an injected LLM.
"""

import asyncio
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import TypedDict
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langchain_mcp_adapters.client import MultiServerMCPClient
from tools.tools import make_tools
from app.config import CVE_SERVER_PATH
from observability.logging_setup import log_llm_call, logger
from observability.fault_tolerance import FALLBACK_MESSAGE
from app.utils import extract_text
from langchain.agents import create_agent
from prompts.breach_workflow import ASSESSMENT_PROMPT, RESEARCH_PROMPT, REPORT_PROMPT


class BreachWorkflowState(TypedDict):
    """State schema for the breach analysis workflow."""

    incident: str
    assessment: str
    research: str
    report: str
    tool_calls_log: list


class BreachWorkflow:
    """Three-node sequential breach analysis workflow with an injected LLM."""

    def __init__(self, llm: BaseChatModel):
        self._llm = llm

    @log_llm_call("Workflow")
    def run(self, incident: str, k: int = 6) -> tuple[str, list[dict]]:
        try:
            return asyncio.run(self._run_async(incident, k))
        except Exception as exc:
            logger.error(f"[Workflow] Failed — {type(exc).__name__}: {exc}")
            return FALLBACK_MESSAGE, []

    async def _run_async(self, incident: str, k: int) -> tuple[str, list[dict]]:
        local_tools = make_tools(k=k)
        mcp_client = MultiServerMCPClient({
            "cve": {"command": "python", "args": [CVE_SERVER_PATH], "transport": "stdio"}
        })
        mcp_tools = await mcp_client.get_tools()
        all_tools = local_tools + mcp_tools

        llm = self._llm

        def assessment_node(state: BreachWorkflowState) -> dict:
            logger.info("[Workflow] Node 1/3 — BreachAssessmentAgent")
            response = llm.invoke([
                SystemMessage(content=ASSESSMENT_PROMPT),
                HumanMessage(content=state["incident"]),
            ])
            return {"assessment": extract_text(response.content)}

        def report_node(state: BreachWorkflowState) -> dict:
            logger.info("[Workflow] Node 3/3 — ReportCompilerAgent")
            prompt = f"Incident:\n{state['incident']}\n\nAssessment:\n{state['assessment']}\n\nResearch findings:\n{state['research']}"
            response = llm.invoke([
                SystemMessage(content=REPORT_PROMPT),
                HumanMessage(content=prompt),
            ])
            return {"report": extract_text(response.content)}

        async def research_node(state: BreachWorkflowState) -> dict:
            logger.info("[Workflow] Node 2/3 — ThreatResearchAgent")
            agent = create_agent(llm, all_tools, prompt=RESEARCH_PROMPT)
            task = f"Assessment:\n{state['assessment']}\n\nOriginal incident:\n{state['incident']}"
            result = await agent.ainvoke({"messages": [HumanMessage(content=task)]})
            final = extract_text(result["messages"][-1].content)
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
            return {"research": final, "tool_calls_log": log}

        graph = StateGraph(BreachWorkflowState)
        graph.add_node("assess", assessment_node)
        graph.add_node("research", research_node)
        graph.add_node("compile", report_node)
        graph.add_edge(START, "assess")
        graph.add_edge("assess", "research")
        graph.add_edge("research", "compile")
        graph.add_edge("compile", END)

        workflow = graph.compile()
        initial_state: BreachWorkflowState = {
            "incident": incident, "assessment": "", "research": "",
            "report": "", "tool_calls_log": [],
        }
        final_state = await workflow.ainvoke(initial_state)
        return final_state["report"], final_state["tool_calls_log"]