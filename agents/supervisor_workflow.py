"""LLM supervisor that routes to specialist sub-agents with an injected LLM.
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
from prompts.supervisor_workflow import SUPERVISOR_PROMPT, CVE_ANALYST_PROMPT, COST_ANALYST_PROMPT, COMPLIANCE_ANALYST_PROMPT


class SupervisorWorkflow:
    
    def __init__(self, llm: BaseChatModel):
        self._llm = llm

    @_call_logger.log_llm_call("Supervisor")
    def run(self, incident: str, k: int = 6) -> tuple[str, list[dict]]:
        try:
            return asyncio.run(self._run_async(incident, k))
        except Exception as exc:
            logger.error(f"[Supervisor] Failed — {type(exc).__name__}: {exc}")
            return FALLBACK_MESSAGE, []

    async def _run_async(self, incident: str, k: int) -> tuple[str, list[dict]]:
        local_tools = make_tools(k=k)
        mcp_client = MultiServerMCPClient({
            "cve": {"command": "python", "args": [CVE_SERVER_PATH], "transport": "stdio"}
        })
        mcp_tools = await mcp_client.get_tools()
        specialist_tools = self._make_specialist_tools(local_tools, mcp_tools)

        supervisor = create_agent(self._llm, specialist_tools, prompt=SUPERVISOR_PROMPT)
        result = await supervisor.ainvoke({"messages": [HumanMessage(content=incident)]})
        content = extract_text(result["messages"][-1].content)

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

        return content, tool_calls_log

    def _make_specialist_tools(self, local_tools, mcp_tools):
        llm = self._llm
        kb_tools = [t for t in local_tools if t.name == "search_knowledge_base"]
        calc_tools = [t for t in local_tools if t.name == "calculate_breach_cost"]
        cve_tools = mcp_tools

        @tool
        def cve_analyst(cve_ids: str) -> str:
            """Specialist sub-agent: looks up and analyses CVE vulnerabilities.
            Pass a comma-separated list of CVE IDs (e.g. 'CVE-2021-44228, CVE-2023-1234').
            Returns severity scores, descriptions and breach relevance for each CVE."""
            logger.info("[Supervisor] → CveAnalystAgent invoked")
            agent = create_agent(llm, cve_tools, prompt=CVE_ANALYST_PROMPT)
            result = run_in_thread(
                agent.ainvoke({"messages": [HumanMessage(content=f"Analyse these CVEs: {cve_ids}")]})
            )
            return extract_text(result["messages"][-1].content)

        @tool
        def cost_analyst(industry: str, record_count: str, data_type: str) -> str:
            """Specialist sub-agent: researches breach costs for a specific industry and incident profile.
            Pass the industry name, estimated record count, and type of data exposed.
            Returns cost benchmarks and financial impact estimates from the IBM report."""
            logger.info("[Supervisor] → CostAnalystAgent invoked")
            agent = create_agent(llm, kb_tools + calc_tools, prompt=COST_ANALYST_PROMPT)
            query = f"Industry: {industry}\nRecord count: {record_count}\nData type: {data_type}"
            result = run_in_thread(
                agent.ainvoke({"messages": [HumanMessage(content=query)]})
            )
            return extract_text(result["messages"][-1].content)

        @tool
        def compliance_analyst(incident_summary: str) -> str:
            """Specialist sub-agent: assesses regulatory and compliance implications of a breach.
            Pass a brief incident summary including industry, data type, and geography if known.
            Returns applicable regulations, notification deadlines, and potential penalties."""
            logger.info("[Supervisor] → ComplianceAnalystAgent invoked")
            response = llm.invoke([
                SystemMessage(content=COMPLIANCE_ANALYST_PROMPT),
                HumanMessage(content=incident_summary),
            ])
            return extract_text(response.content)

        return [cve_analyst, cost_analyst, compliance_analyst]