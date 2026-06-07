import asyncio
import concurrent.futures
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from tools.tools import make_tools
from app.config import OPENAI_API_KEY, OPENAI_MODEL, CVE_SERVER_PATH
from observability.logging_setup import log_llm_call, logger
from observability.fault_tolerance import FALLBACK_MESSAGE



def _run_in_thread(coro, timeout: int = 45) -> str:
    """Run an async coroutine in a fresh thread with its own event loop.
    Needed because specialist tools are sync @tool functions called from inside
    an already-running async event loop — nested asyncio.run() is not allowed."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(asyncio.run, coro)
        return future.result(timeout=timeout)


def _make_llm(temperature: float = 0.0, max_tokens: int = 1024) -> ChatOpenAI:
    return ChatOpenAI(
        model=OPENAI_MODEL,
        api_key=OPENAI_API_KEY,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            b.get("text", "") for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return str(content)


# ── Specialist sub-agents exposed as tools to the supervisor ───────────────────
#
# Each function is a fully autonomous agent with its own system prompt and
# (where needed) its own tool access. The supervisor calls them like tools,
# but unlike simple functions, each one reasons internally before returning.
#
# This is the key distinction from Step 12 (conditional workflow):
# - Step 12: CODE decides routing (hardcoded severity check)
# - Step 13: LLM SUPERVISOR decides routing based on context at runtime


def _make_specialist_tools(local_tools, mcp_tools):
    """Build the three specialist sub-agent tools, each with access to relevant tools."""

    kb_tools = [t for t in local_tools if t.name == "search_knowledge_base"]
    calc_tools = [t for t in local_tools if t.name == "calculate_breach_cost"]
    cve_tools = mcp_tools

    # ── Sub-agent 1: CveAnalystAgent ──────────────────────────────────────────
    _CVE_ANALYST_PROMPT = """You are a CVE specialist. You will be given one or more CVE IDs.
For each CVE: look it up, report the CVSS score, severity, description, and specific relevance
to the incident. If no CVE IDs are provided, return "No CVEs to analyse."
Be concise and factual."""

    @tool
    def cve_analyst(cve_ids: str) -> str:
        """Specialist sub-agent: looks up and analyses CVE vulnerabilities.
        Pass a comma-separated list of CVE IDs (e.g. 'CVE-2021-44228, CVE-2023-1234').
        Returns severity scores, descriptions and breach relevance for each CVE."""
        logger.info("[Supervisor] → CveAnalystAgent invoked")
        llm = _make_llm(max_tokens=512)
        agent = create_react_agent(llm, cve_tools, prompt=_CVE_ANALYST_PROMPT)
        result = _run_in_thread(
            agent.ainvoke({"messages": [HumanMessage(content=f"Analyse these CVEs: {cve_ids}")]})
        )
        return _extract_text(result["messages"][-1].content)

    # ── Sub-agent 2: CostAnalystAgent ─────────────────────────────────────────
    _COST_ANALYST_PROMPT = """You are a breach cost analyst. Search the knowledge base for
breach cost data relevant to the given industry and record count. Report:
- Industry average total breach cost
- Relevant cost factors (attack vector, data type, detection time)
- Financial impact estimate for the given record count if calculable
Cite specific figures. Do not guess."""

    @tool
    def cost_analyst(industry: str, record_count: str, data_type: str) -> str:
        """Specialist sub-agent: researches breach costs for a specific industry and incident profile.
        Pass the industry name, estimated record count, and type of data exposed.
        Returns cost benchmarks and financial impact estimates from the IBM report."""
        logger.info("[Supervisor] → CostAnalystAgent invoked")
        llm = _make_llm(max_tokens=512)
        agent = create_react_agent(llm, kb_tools + calc_tools, prompt=_COST_ANALYST_PROMPT)
        query = f"Industry: {industry}\nRecord count: {record_count}\nData type: {data_type}"
        result = _run_in_thread(
            agent.ainvoke({"messages": [HumanMessage(content=query)]})
        )
        return _extract_text(result["messages"][-1].content)

    # ── Sub-agent 3: ComplianceAnalystAgent ───────────────────────────────────
    _COMPLIANCE_ANALYST_PROMPT = """You are a regulatory compliance analyst. Given an incident,
identify:
- Applicable regulations (GDPR, HIPAA, PCI-DSS, SOC 2, CCPA, etc.)
- Mandatory breach notification requirements and deadlines
- Potential regulatory fines and penalties
- Required documentation and evidence preservation steps
Be specific about timelines (e.g. "GDPR: notify supervisory authority within 72 hours")."""

    @tool
    def compliance_analyst(incident_summary: str) -> str:
        """Specialist sub-agent: assesses regulatory and compliance implications of a breach.
        Pass a brief incident summary including industry, data type, and geography if known.
        Returns applicable regulations, notification deadlines, and potential penalties."""
        logger.info("[Supervisor] → ComplianceAnalystAgent invoked")
        llm = _make_llm(max_tokens=512)
        response = llm.invoke([
            SystemMessage(content=_COMPLIANCE_ANALYST_PROMPT),
            HumanMessage(content=incident_summary),
        ])
        return _extract_text(response.content)

    return [cve_analyst, cost_analyst, compliance_analyst]


# ── Supervisor system prompt ───────────────────────────────────────────────────
# The supervisor decides at runtime which sub-agents to invoke based on what
# the incident actually requires — no hardcoded routing rules.

SUPERVISOR_PROMPT = """You are the Aegis Threat Supervisor. You coordinate a team of specialist analysts.

Your job is to AUTONOMOUSLY decide which specialists to invoke based on the incident, then compile a final report.

Available specialists:
- cve_analyst: call when the incident mentions specific CVE IDs or known vulnerabilities
- cost_analyst: call when you can identify the industry AND have some estimate of records affected
- compliance_analyst: call when the incident involves personal data, health records, financial data, or regulated industries

Decision rules (apply your own judgement — these are guidelines, not code):
- Do NOT call a specialist if the incident clearly lacks the information they need
- You MAY call multiple specialists if the incident warrants it
- You MAY call none if the incident is too vague for specialist analysis

After gathering specialist input, compile a structured final report:
- **Incident Summary**
- **Specialist Findings** (one section per specialist you invoked)
- **Recommended Actions** (3-5 concrete steps)

State clearly which specialists you chose NOT to invoke and why."""


async def _run_supervisor_async(
    incident: str,
    k: int,
    temperature: float,
    max_tokens: int,
) -> tuple[str, list[dict]]:
    local_tools = make_tools(k=k)
    mcp_client = MultiServerMCPClient({
        "cve": {
            "command": "python",
            "args": [CVE_SERVER_PATH],
            "transport": "stdio",
        }
    })
    mcp_tools = await mcp_client.get_tools()

    specialist_tools = _make_specialist_tools(local_tools, mcp_tools)

    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        api_key=OPENAI_API_KEY,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    supervisor = create_react_agent(llm, specialist_tools, prompt=SUPERVISOR_PROMPT)
    result = await supervisor.ainvoke({"messages": [HumanMessage(content=incident)]})

    content = _extract_text(result["messages"][-1].content)

    # Build tool call log
    tool_calls_log = []
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls_log.append({"tool": tc["name"], "input": tc["args"], "output": ""})
        if hasattr(msg, "name") and msg.name:
            for entry in reversed(tool_calls_log):
                if entry["tool"] == msg.name and entry["output"] == "":
                    entry["output"] = _extract_text(msg.content)
                    break

    return content, tool_calls_log


@log_llm_call("Supervisor")
def run_supervisor(
    incident: str,
    k: int = 6,
    temperature: float = 0.2,
    max_tokens: int = 2048,
) -> tuple[str, list[dict]]:
    try:
        return asyncio.run(_run_supervisor_async(incident, k, temperature, max_tokens))
    except Exception as exc:
        logger.error(f"[Supervisor] Failed — {type(exc).__name__}: {exc}")
        return FALLBACK_MESSAGE, []
