import asyncio
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langchain_mcp_adapters.client import MultiServerMCPClient
from tools.tools import make_tools
from app.config import OPENAI_API_KEY, OPENAI_MODEL
from observability.logging_setup import log_llm_call, logger
from observability.fault_tolerance import FALLBACK_MESSAGE

_CVE_SERVER = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "mcp_servers", "cve_server.py"))


# ── Shared state flowing through the workflow ──────────────────────────────────
# Equivalent to AgenticScope in the tutorial.
# Each node reads from this dict and writes its output back into it.

class BreachWorkflowState(TypedDict):
    incident: str           # original user input — set at workflow start
    assessment: str         # output of BreachAssessmentAgent
    research: str           # output of ThreatResearchAgent
    report: str             # output of ReportCompilerAgent
    tool_calls_log: list    # accumulated tool calls across all agents


# ── Agent 1: BreachAssessmentAgent ────────────────────────────────────────────
# Reads:  state["incident"]
# Writes: state["assessment"]
# Role:   Extract structured facts from the raw incident description.
#         No tools — pure LLM reasoning.

_ASSESSMENT_PROMPT = """You are a breach intake analyst. Your only job is to extract structured facts from an incident description.

Extract and return:
- Industry/sector
- Attack vector (e.g. ransomware, phishing, credential theft)
- CVE IDs mentioned (if any)
- Number of records affected (if mentioned)
- Type of data exposed (e.g. PII, health records, financial)
- Estimated severity (Low / Medium / High / Critical)

Be concise and factual. If something is unknown, say "Unknown". Do not add interpretation."""


def _make_llm(temperature: float = 0.0, max_tokens: int = 512) -> ChatOpenAI:
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


def assessment_node(state: BreachWorkflowState) -> dict:
    logger.info("[Workflow] Node 1/3 — BreachAssessmentAgent")
    llm = _make_llm()
    response = llm.invoke([
        SystemMessage(content=_ASSESSMENT_PROMPT),
        HumanMessage(content=state["incident"]),
    ])
    return {"assessment": _extract_text(response.content)}


# ── Agent 2: ThreatResearchAgent ──────────────────────────────────────────────
# Reads:  state["incident"] + state["assessment"]
# Writes: state["research"], state["tool_calls_log"]
# Role:   Use the structured assessment to search the knowledge base and
#         look up any CVEs. Has access to all tools.

_RESEARCH_PROMPT = """You are a threat intelligence researcher. You will be given a structured breach assessment.

Your job is to gather supporting data using your tools:
1. Search the knowledge base for breach costs, statistics and attack vector data relevant to this incident
2. Look up any CVE IDs mentioned using lookup_cve
3. Search for industry-specific breach benchmarks

Search multiple times with different queries to build a complete picture. Do not draw conclusions — just gather data. Report everything you find."""


async def _research_node_async(state: BreachWorkflowState, tools, tool_map) -> dict:
    from langgraph.prebuilt import create_react_agent

    llm = _make_llm(temperature=0.1, max_tokens=2048)
    agent = create_react_agent(llm, tools, prompt=_RESEARCH_PROMPT)

    task = f"Assessment:\n{state['assessment']}\n\nOriginal incident:\n{state['incident']}"
    result = await agent.ainvoke({"messages": [HumanMessage(content=task)]})

    final = _extract_text(result["messages"][-1].content)

    # Collect tool calls from this node
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

    return {"research": final, "tool_calls_log": log}


# ── Agent 3: ReportCompilerAgent ──────────────────────────────────────────────
# Reads:  state["incident"] + state["assessment"] + state["research"]
# Writes: state["report"]
# Role:   Synthesise all previous outputs into a final structured report.
#         No tools — pure synthesis.

_REPORT_PROMPT = """You are a senior cybersecurity analyst writing a final breach triage report.

You will receive:
- The original incident description
- A structured assessment of the incident facts
- Research data gathered from the knowledge base and CVE database

Synthesise everything into a structured report with these sections:
- **Incident Summary**
- **Threat Context** (statistics and benchmarks from the research)
- **CVE Analysis** (if applicable)
- **Financial Impact Estimate**
- **Recommended Actions** (3-5 concrete steps)

Be precise, cite specific figures from the research, and be actionable."""


def report_node(state: BreachWorkflowState) -> dict:
    logger.info("[Workflow] Node 3/3 — ReportCompilerAgent")
    llm = _make_llm(temperature=0.2, max_tokens=2048)
    prompt = f"""Incident:\n{state['incident']}

Assessment:\n{state['assessment']}

Research findings:\n{state['research']}"""

    response = llm.invoke([
        SystemMessage(content=_REPORT_PROMPT),
        HumanMessage(content=prompt),
    ])
    return {"report": _extract_text(response.content)}


# ── Workflow assembly ──────────────────────────────────────────────────────────

async def _run_workflow_async(
    incident: str,
    k: int,
    temperature: float,
    max_tokens: int,
) -> tuple[str, list[dict]]:
    local_tools = make_tools(k=k)
    mcp_client = MultiServerMCPClient({
        "cve": {
            "command": "python",
            "args": [_CVE_SERVER],
            "transport": "stdio",
        }
    })
    mcp_tools = await mcp_client.get_tools()
    all_tools = local_tools + mcp_tools
    tool_map = {t.name: t for t in all_tools}

    # Wrap the async research node so it can access tools via closure
    async def research_node(state: BreachWorkflowState) -> dict:
        logger.info("[Workflow] Node 2/3 — ThreatResearchAgent")
        return await _research_node_async(state, all_tools, tool_map)

    # Build the sequential graph
    graph = StateGraph(BreachWorkflowState)
    graph.add_node("assess",   assessment_node)
    graph.add_node("research", research_node)
    graph.add_node("compile",  report_node)

    graph.add_edge(START,      "assess")
    graph.add_edge("assess",   "research")
    graph.add_edge("research", "compile")
    graph.add_edge("compile",  END)

    workflow = graph.compile()

    initial_state: BreachWorkflowState = {
        "incident": incident,
        "assessment": "",
        "research": "",
        "report": "",
        "tool_calls_log": [],
    }

    final_state = await workflow.ainvoke(initial_state)
    return final_state["report"], final_state["tool_calls_log"]


@log_llm_call("Workflow")
def run_workflow(
    incident: str,
    k: int = 6,
    temperature: float = 0.2,
    max_tokens: int = 2048,
) -> tuple[str, list[dict]]:
    try:
        return asyncio.run(_run_workflow_async(incident, k, temperature, max_tokens))
    except Exception as exc:
        logger.error(f"[Workflow] Failed — {type(exc).__name__}: {exc}")
        return FALLBACK_MESSAGE, []
