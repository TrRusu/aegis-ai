import asyncio
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import TypedDict, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langchain_mcp_adapters.client import MultiServerMCPClient
from tools.tools import make_tools
from app.config import OPENAI_API_KEY, OPENAI_MODEL, CVE_SERVER_PATH
from observability.logging_setup import log_llm_call, logger
from observability.fault_tolerance import FALLBACK_MESSAGE



# ── Shared state ───────────────────────────────────────────────────────────────

class ComposedWorkflowState(TypedDict):
    incident: str               # original user input
    threat_analysis: str        # output of ThreatFeedbackAgent   (parallel)
    compliance_analysis: str    # output of ComplianceFeedbackAgent (parallel)
    severity: str               # extracted from threat_analysis for routing
    response_plan: str          # output of conditional response agent
    report: str                 # final synthesis
    tool_calls_log: list


# ── Helpers ────────────────────────────────────────────────────────────────────

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


# ── PARALLEL WORKFLOW ──────────────────────────────────────────────────────────
# Equivalent to @ParallelAgent in the tutorial.
# ThreatFeedbackAgent and ComplianceFeedbackAgent run concurrently via
# asyncio.gather — neither depends on the other's output.

_THREAT_PROMPT = """You are a threat intelligence analyst.
Analyze the incident for: attack vector, exploited vulnerabilities, threat actor TTPs,
affected systems, severity level (Critical / High / Medium / Low), and breach scope.
Be concise and factual. End your response with a line: "Severity: <level>"."""

_COMPLIANCE_PROMPT = """You are a regulatory compliance analyst.
Analyze the incident for: applicable regulations (GDPR, HIPAA, PCI-DSS, etc.),
mandatory notification requirements, notification deadlines, potential fines,
and required documentation. Be concise and factual."""


async def _threat_agent(incident: str) -> str:
    llm = _make_llm()
    response = llm.invoke([
        SystemMessage(content=_THREAT_PROMPT),
        HumanMessage(content=incident),
    ])
    return _extract_text(response.content)


async def _compliance_agent(incident: str) -> str:
    llm = _make_llm()
    response = llm.invoke([
        SystemMessage(content=_COMPLIANCE_PROMPT),
        HumanMessage(content=incident),
    ])
    return _extract_text(response.content)


async def parallel_analysis_node(state: ComposedWorkflowState) -> dict:
    logger.info("[Composed] Node 1/4 — Parallel: ThreatAgent + ComplianceAgent")
    threat, compliance = await asyncio.gather(
        _threat_agent(state["incident"]),
        _compliance_agent(state["incident"]),
    )

    # Extract severity from threat analysis for the router
    severity = "Medium"
    for line in threat.splitlines():
        if line.lower().startswith("severity:"):
            severity = line.split(":", 1)[1].strip()
            break

    return {
        "threat_analysis": threat,
        "compliance_analysis": compliance,
        "severity": severity,
    }


# ── CONDITIONAL ROUTING ────────────────────────────────────────────────────────
# Equivalent to @ConditionalAgent in the tutorial.
# Routes to CriticalResponseAgent or StandardResponseAgent based on severity.

def severity_router(state: ComposedWorkflowState) -> Literal["critical", "standard"]:
    s = state["severity"].lower()
    if "critical" in s or "high" in s:
        logger.info("[Composed] Router → critical response path")
        return "critical"
    logger.info("[Composed] Router → standard response path")
    return "standard"


_CRITICAL_RESPONSE_PROMPT = """You are an incident commander handling a CRITICAL or HIGH severity breach.

Given the threat and compliance analysis, produce an IMMEDIATE action plan:
- First 24 hours: containment steps (be specific)
- First 72 hours: notification obligations and who to contact
- First 2 weeks: remediation priorities

Use the knowledge base data provided. Be direct and actionable. No fluff."""

_STANDARD_RESPONSE_PROMPT = """You are a security analyst handling a MEDIUM or LOW severity breach.

Given the threat and compliance analysis, produce a standard remediation plan:
- Remediation steps in priority order
- Compliance notifications required (if any)
- Preventive measures to avoid recurrence

Be concise and practical."""


async def critical_response_node(state: ComposedWorkflowState, tools, tool_map) -> dict:
    logger.info("[Composed] Node 2/4 — CriticalResponseAgent")
    from langgraph.prebuilt import create_react_agent

    llm = _make_llm(temperature=0.1, max_tokens=1024)
    agent = create_react_agent(llm, tools, prompt=_CRITICAL_RESPONSE_PROMPT)

    task = (
        f"Incident:\n{state['incident']}\n\n"
        f"Threat Analysis:\n{state['threat_analysis']}\n\n"
        f"Compliance Analysis:\n{state['compliance_analysis']}"
    )
    result = await agent.ainvoke({"messages": [HumanMessage(content=task)]})
    response = _extract_text(result["messages"][-1].content)

    log = list(state.get("tool_calls_log", []))
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                log.append({"tool": tc["name"], "input": tc["args"], "output": ""})
        if hasattr(msg, "name") and msg.name:
            for entry in reversed(log):
                if entry["tool"] == msg.name and entry["output"] == "":
                    entry["output"] = _extract_text(msg.content)
                    break

    return {"response_plan": response, "tool_calls_log": log}


async def standard_response_node(state: ComposedWorkflowState, tools, tool_map) -> dict:
    logger.info("[Composed] Node 2/4 — StandardResponseAgent")
    from langgraph.prebuilt import create_react_agent

    llm = _make_llm(temperature=0.1, max_tokens=1024)
    agent = create_react_agent(llm, tools, prompt=_STANDARD_RESPONSE_PROMPT)

    task = (
        f"Incident:\n{state['incident']}\n\n"
        f"Threat Analysis:\n{state['threat_analysis']}\n\n"
        f"Compliance Analysis:\n{state['compliance_analysis']}"
    )
    result = await agent.ainvoke({"messages": [HumanMessage(content=task)]})
    response = _extract_text(result["messages"][-1].content)

    log = list(state.get("tool_calls_log", []))
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                log.append({"tool": tc["name"], "input": tc["args"], "output": ""})
        if hasattr(msg, "name") and msg.name:
            for entry in reversed(log):
                if entry["tool"] == msg.name and entry["output"] == "":
                    entry["output"] = _extract_text(msg.content)
                    break

    return {"response_plan": response, "tool_calls_log": log}


# ── SYNTHESIS ─────────────────────────────────────────────────────────────────
# Final node — combines all outputs into one coherent report.

_SYNTHESIS_PROMPT = """You are a senior security officer writing a final breach report.

Synthesise everything into a structured report:
- **Executive Summary**
- **Threat Analysis** (from threat assessment)
- **Compliance & Regulatory Impact** (from compliance assessment)
- **Response Plan** (from the response agent — preserve all action items)
- **Key Metrics** (severity, records affected, estimated cost if known)

Be concise. Preserve all specific actions and deadlines from the response plan."""


def synthesis_node(state: ComposedWorkflowState) -> dict:
    logger.info("[Composed] Node 4/4 — SynthesisAgent")
    llm = _make_llm(temperature=0.2, max_tokens=2048)
    task = (
        f"Incident:\n{state['incident']}\n\n"
        f"Threat Analysis:\n{state['threat_analysis']}\n\n"
        f"Compliance Analysis:\n{state['compliance_analysis']}\n\n"
        f"Response Plan (severity: {state['severity']}):\n{state['response_plan']}"
    )
    response = llm.invoke([
        SystemMessage(content=_SYNTHESIS_PROMPT),
        HumanMessage(content=task),
    ])
    return {"report": _extract_text(response.content)}


# ── Graph assembly ─────────────────────────────────────────────────────────────

async def _run_composed_async(
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
    all_tools = local_tools + mcp_tools
    tool_map = {t.name: t for t in all_tools}

    # Bind tools into closures so nodes can access them
    async def _critical(state): return await critical_response_node(state, all_tools, tool_map)
    async def _standard(state): return await standard_response_node(state, all_tools, tool_map)

    graph = StateGraph(ComposedWorkflowState)
    graph.add_node("parallel_analysis", parallel_analysis_node)
    graph.add_node("critical_response", _critical)
    graph.add_node("standard_response", _standard)
    graph.add_node("synthesize",        synthesis_node)

    graph.add_edge(START, "parallel_analysis")

    # Conditional edges — router function decides which response node fires
    graph.add_conditional_edges(
        "parallel_analysis",
        severity_router,
        {"critical": "critical_response", "standard": "standard_response"},
    )

    graph.add_edge("critical_response", "synthesize")
    graph.add_edge("standard_response", "synthesize")
    graph.add_edge("synthesize", END)

    workflow = graph.compile()

    initial_state: ComposedWorkflowState = {
        "incident": incident,
        "threat_analysis": "",
        "compliance_analysis": "",
        "severity": "",
        "response_plan": "",
        "report": "",
        "tool_calls_log": [],
    }

    final_state = await workflow.ainvoke(initial_state)
    return final_state["report"], final_state["tool_calls_log"]


@log_llm_call("ComposedWorkflow")
def run_composed_workflow(
    incident: str,
    k: int = 6,
    temperature: float = 0.2,
    max_tokens: int = 2048,
) -> tuple[str, list[dict]]:
    try:
        return asyncio.run(_run_composed_async(incident, k, temperature, max_tokens))
    except Exception as exc:
        logger.error(f"[ComposedWorkflow] Failed — {type(exc).__name__}: {exc}")
        return FALLBACK_MESSAGE, []
