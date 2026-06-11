"""Two-phase human-in-the-loop supervisor with an injected LLM.
"""

import asyncio
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from tools.tools import make_tools
from app.config import CVE_SERVER_PATH
from observability.logging_setup import CallLogger, logger

_call_logger = CallLogger()
from observability.fault_tolerance import FALLBACK_MESSAGE
from app.utils import extract_text, run_in_thread
from prompts.supervisor_hitl import PHASE1_PROMPT, PHASE2_PROMPT, CVE_ANALYST_PROMPT, COST_ANALYST_PROMPT, COMPLIANCE_ANALYST_PROMPT


APPROVAL_SEVERITIES = {"critical", "high"}


def _extract_severity(text: str) -> str:
    import re
    match = re.search(r"severity[:\s*_]+([a-z]+)", text, re.IGNORECASE)
    if match:
        level = match.group(1).strip().lower().capitalize()
        if level in ("Critical", "High", "Medium", "Low"):
            return level
    lower = text.lower()
    for level in ("critical", "high", "medium", "low"):
        if level in lower:
            return level.capitalize()
    return "Unknown"

def requires_approval(severity: str) -> bool:
    return severity.strip().lower() not in ("critical", "high")


class HitlSupervisor:

    def __init__(self, llm: BaseChatModel):
        self._llm = llm

    @_call_logger.log_llm_call("HITL-Phase1")
    def run_phase1(self, incident: str, k: int = 6) -> tuple[str, str, list[dict]]:
        try:
            return asyncio.run(self._phase1_async(incident, k))
        except Exception as exc:
            logger.error(f"[HITL] Phase 1 failed — {type(exc).__name__}: {exc}")
            return FALLBACK_MESSAGE, "Unknown", []

    @_call_logger.log_llm_call("HITL-Phase2")
    def run_phase2(
        self,
        incident: str,
        preliminary_analysis: str,
        decision: str,
        reason: str,
    ) -> str:
        try:
            prompt = PHASE2_PROMPT.format(decision=decision, reason=reason)
            task = f"Incident:\n{incident}\n\nPreliminary briefing:\n{preliminary_analysis}"
            response = self._llm.invoke([SystemMessage(content=prompt), HumanMessage(content=task)])
            return extract_text(response.content)
        except Exception as exc:
            logger.error(f"[HITL] Phase 2 failed — {type(exc).__name__}: {exc}")
            return FALLBACK_MESSAGE

    async def _phase1_async(self, incident: str, k: int) -> tuple[str, str, list[dict]]:
        local_tools = make_tools(k=k)
        mcp_client = MultiServerMCPClient({
            "cve": {"command": "python", "args": [CVE_SERVER_PATH], "transport": "stdio"}
        })
        mcp_tools = await mcp_client.get_tools()
        specialist_tools = self._make_specialist_tools(local_tools, mcp_tools)

        supervisor = create_agent(self._llm, specialist_tools, prompt=PHASE1_PROMPT)
        result = await supervisor.ainvoke({"messages": [HumanMessage(content=incident)]})
        content = extract_text(result["messages"][-1].content)
        severity = _extract_severity(content)

        tool_calls_log = []
        for msg in result["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls_log.append({"tool": tc["name"], "input": tc["args"], "output": ""})
            if hasattr(msg, "name") and msg.name:
                for entry in reversed(tool_calls_log):
                    if entry["tool"] == msg.name and entry["output"] == "":
                        entry["output"] = extract_text(msg.content)
                        break

        return content, severity, tool_calls_log

    def _make_specialist_tools(self, local_tools, mcp_tools):
        llm = self._llm
        kb_tools = [t for t in local_tools if t.name == "search_knowledge_base"]
        calc_tools = [t for t in local_tools if t.name == "calculate_breach_cost"]
        cve_tools = mcp_tools

        @tool
        def cve_analyst(cve_ids: str) -> str:
            """Specialist sub-agent: looks up and analyses CVE vulnerabilities."""
            logger.info("[HITL Supervisor] → CveAnalystAgent")
            agent = create_agent(llm, cve_tools, prompt=CVE_ANALYST_PROMPT)
            result = run_in_thread(
                agent.ainvoke({"messages": [HumanMessage(content=f"Analyse: {cve_ids}")]})
            )
            return extract_text(result["messages"][-1].content)

        @tool
        def cost_analyst(industry: str, record_count: str, data_type: str) -> str:
            """Specialist sub-agent: researches breach costs for a given industry and incident profile."""
            logger.info("[HITL Supervisor] → CostAnalystAgent")
            agent = create_agent(llm, kb_tools + calc_tools, prompt=COST_ANALYST_PROMPT)
            result = run_in_thread(
                agent.ainvoke({"messages": [HumanMessage(
                    content=f"Industry: {industry}\nRecords: {record_count}\nData: {data_type}"
                )]})
            )
            return extract_text(result["messages"][-1].content)

        @tool
        def compliance_analyst(incident_summary: str) -> str:
            """Specialist sub-agent: assesses regulatory and compliance implications of a breach."""
            logger.info("[HITL Supervisor] → ComplianceAnalystAgent")
            response = llm.invoke([
                SystemMessage(content=COMPLIANCE_ANALYST_PROMPT),
                HumanMessage(content=incident_summary),
            ])
            return extract_text(response.content)

        return [cve_analyst, cost_analyst, compliance_analyst]


