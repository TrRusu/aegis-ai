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
from app.config import OPENAI_API_KEY, OPENAI_MODEL
from observability.logging_setup import log_llm_call, logger
from observability.fault_tolerance import FALLBACK_MESSAGE

_CVE_SERVER = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "mcp_servers", "cve_server.py"))

# Severities that require human approval before the final report is produced.
APPROVAL_SEVERITIES = {"critical", "high"}


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


def _extract_severity(text: str) -> str:
    import re
    # Match "Severity: Critical" anywhere in the text, case-insensitive, ignoring markdown
    match = re.search(r"severity[:\s*_]+([a-z]+)", text, re.IGNORECASE)
    if match:
        level = match.group(1).strip().lower().capitalize()
        if level in ("Critical", "High", "Medium", "Low"):
            return level
    # Fallback: scan for the words themselves
    lower = text.lower()
    for level in ("critical", "high", "medium", "low"):
        if level in lower:
            return level.capitalize()
    return "Unknown"


def _run_in_thread(coro, timeout: int = 45) -> str:
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(asyncio.run, coro)
        return future.result(timeout=timeout)


# ── Specialist sub-agents (same as supervisor_workflow.py) ────────────────────

def _make_specialist_tools(local_tools, mcp_tools):
    kb_tools = [t for t in local_tools if t.name == "search_knowledge_base"]
    calc_tools = [t for t in local_tools if t.name == "calculate_breach_cost"]
    cve_tools = mcp_tools

    _CVE_ANALYST_PROMPT = """You are a CVE specialist. Look up each CVE ID provided, report the CVSS score,
severity, description, and relevance to the incident. Be concise and factual."""

    _COST_ANALYST_PROMPT = """You are a breach cost analyst. Search the knowledge base for breach cost data
relevant to the given industry and record count. Report industry average, relevant cost factors, and financial
impact estimate. Cite specific figures."""

    _COMPLIANCE_ANALYST_PROMPT = """You are a regulatory compliance analyst. Identify applicable regulations,
mandatory notification requirements and deadlines, potential fines, and required documentation steps."""

    @tool
    def cve_analyst(cve_ids: str) -> str:
        """Specialist: looks up and analyses CVE vulnerabilities. Pass comma-separated CVE IDs."""
        logger.info("[HITL Supervisor] → CveAnalystAgent")
        llm = _make_llm(max_tokens=512)
        agent = create_react_agent(llm, cve_tools, prompt=_CVE_ANALYST_PROMPT)
        result = _run_in_thread(
            agent.ainvoke({"messages": [HumanMessage(content=f"Analyse: {cve_ids}")]})
        )
        return _extract_text(result["messages"][-1].content)

    @tool
    def cost_analyst(industry: str, record_count: str, data_type: str) -> str:
        """Specialist: researches breach costs. Pass industry, record count, and data type."""
        logger.info("[HITL Supervisor] → CostAnalystAgent")
        llm = _make_llm(max_tokens=512)
        agent = create_react_agent(llm, kb_tools + calc_tools, prompt=_COST_ANALYST_PROMPT)
        result = _run_in_thread(
            agent.ainvoke({"messages": [HumanMessage(
                content=f"Industry: {industry}\nRecords: {record_count}\nData: {data_type}"
            )]})
        )
        return _extract_text(result["messages"][-1].content)

    @tool
    def compliance_analyst(incident_summary: str) -> str:
        """Specialist: assesses regulatory implications. Pass a brief incident summary."""
        logger.info("[HITL Supervisor] → ComplianceAnalystAgent")
        llm = _make_llm(max_tokens=512)
        response = llm.invoke([
            SystemMessage(content=_COMPLIANCE_ANALYST_PROMPT),
            HumanMessage(content=incident_summary),
        ])
        return _extract_text(response.content)

    return [cve_analyst, cost_analyst, compliance_analyst]


# ── Phase 1 prompt — intelligence gathering only, no final actions ─────────────

_PHASE1_PROMPT = """You are the Aegis Threat Supervisor coordinating specialist analysts.

Invoke relevant specialists to gather intelligence on the incident, then produce a preliminary briefing.

Available specialists:
- cve_analyst: when CVE IDs are mentioned
- cost_analyst: when industry AND record count are known
- compliance_analyst: when personal/health/financial data or regulated industries are involved

Produce a preliminary briefing with these exact sections:
- **Incident Summary**
- **Severity Assessment**: your reasoning, then on its own line write EXACTLY: Severity: Critical  (or High, Medium, Low)
- **Specialist Findings** (one section per specialist invoked)
- **Specialists not invoked** and why

IMPORTANT: The line "Severity: <level>" must appear on its own line with no extra formatting.
Do NOT include recommended actions — those come after human review for high-severity incidents."""


# ── Phase 2 prompt — final report after human decision ────────────────────────

_PHASE2_PROMPT = """You are the Aegis Threat Supervisor compiling the final incident report.

You have a preliminary briefing and a human decision. Compile the final report:
- **Incident Summary**
- **Specialist Findings** (from briefing)
- **Human Decision**: {decision} — Reason: {reason}
- **Recommended Actions** (3-5 concrete steps):
  - If APPROVED: full containment and remediation plan
  - If REJECTED: conservative fallback (monitor, document, defer)"""


async def _phase1_async(incident: str, k: int, temperature: float, max_tokens: int):
    local_tools = make_tools(k=k)
    mcp_client = MultiServerMCPClient({
        "cve": {"command": "python", "args": [_CVE_SERVER], "transport": "stdio"}
    })
    mcp_tools = await mcp_client.get_tools()
    specialist_tools = _make_specialist_tools(local_tools, mcp_tools)

    llm = ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY,
                     temperature=temperature, max_tokens=max_tokens)
    supervisor = create_react_agent(llm, specialist_tools, prompt=_PHASE1_PROMPT)
    result = await supervisor.ainvoke({"messages": [HumanMessage(content=incident)]})

    content = _extract_text(result["messages"][-1].content)
    severity = _extract_severity(content)

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

    return content, severity, tool_calls_log


@log_llm_call("HITL-Phase1")
def run_hitl_phase1(
    incident: str,
    k: int = 6,
    temperature: float = 0.2,
    max_tokens: int = 2048,
) -> tuple[str, str, list[dict]]:
    """Phase 1: gather intelligence. Returns (preliminary_analysis, severity, tool_calls_log).
    If severity is Critical/High, the caller should pause and request human approval."""
    try:
        return asyncio.run(_phase1_async(incident, k, temperature, max_tokens))
    except Exception as exc:
        logger.error(f"[HITL] Phase 1 failed — {type(exc).__name__}: {exc}")
        return FALLBACK_MESSAGE, "Unknown", []


@log_llm_call("HITL-Phase2")
def run_hitl_phase2(
    incident: str,
    preliminary_analysis: str,
    decision: str,
    reason: str,
    temperature: float = 0.2,
    max_tokens: int = 2048,
) -> str:
    """Phase 2: compile final report with human decision. Returns final report string."""
    try:
        llm = _make_llm(temperature=temperature, max_tokens=max_tokens)
        prompt = _PHASE2_PROMPT.format(decision=decision, reason=reason)
        task = f"Incident:\n{incident}\n\nPreliminary briefing:\n{preliminary_analysis}"
        response = llm.invoke([SystemMessage(content=prompt), HumanMessage(content=task)])
        return _extract_text(response.content)
    except Exception as exc:
        logger.error(f"[HITL] Phase 2 failed — {type(exc).__name__}: {exc}")
        return FALLBACK_MESSAGE


def requires_approval(severity: str) -> bool:
    # Low/Medium/Unknown require human review before acting
    # High/Critical proceed automatically — too urgent to wait for approval
    s = severity.strip().lower()
    return s not in ("critical", "high")
