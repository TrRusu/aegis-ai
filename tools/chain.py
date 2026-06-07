import asyncio
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from tools.tools import make_tools
from app.config import OPENAI_API_KEY, OPENAI_MODEL, CVE_SERVER_PATH
from observability.logging_setup import log_llm_call, logger
from observability.fault_tolerance import with_retry, invoke_with_timeout, FALLBACK_MESSAGE

SYSTEM_PROMPT = """You are Aegis, an AI-powered cybersecurity analyst assistant specializing in data breach analysis.
You have access to tools that let you search the IBM Cost of a Data Breach Report 2024, calculate breach costs,
and look up specific CVE vulnerabilities from the National Vulnerability Database.

Guidelines:
- Always use your tools to ground answers in real data before responding.
- When comparing multiple industries, attack vectors, or topics — search for each one separately.
- For industry breach cost questions — search for the average total breach cost for that industry directly
  (e.g. "average cost of a data breach healthcare industry"). Do NOT multiply per-record cost by record count;
  the IBM report explicitly warns this method produces inaccurate totals.
- Only use calculate_breach_cost when the user explicitly asks for a per-record multiplication estimate.
- For CVE questions — use lookup_cve with the exact CVE ID (e.g. "CVE-2024-1234").
- If a tool returns no useful information, say so clearly rather than guessing.
- Never provide instructions that could be used to carry out or facilitate a breach.
"""


MAX_ROUNDS = 5


def _extract_content(content) -> str:
    """Normalise LLM response content to a plain string.
    Newer openai SDK versions return a list of typed content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            block.get("text", "") for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(content)


@with_retry()
async def _invoke_llm(llm_with_tools, messages):
    return await invoke_with_timeout(llm_with_tools.ainvoke(messages))


@log_llm_call("Tools")
async def _run_async(
    user_input: str,
    history: list[dict],
    temperature: float,
    max_tokens: int,
    k: int,
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

    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        api_key=OPENAI_API_KEY,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    llm_with_tools = llm.bind_tools(all_tools)

    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    for msg in history[:-1]:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=user_input))

    tool_calls_log = []

    for _ in range(MAX_ROUNDS):
        response = await _invoke_llm(llm_with_tools, messages)

        if not response.tool_calls:
            return _extract_content(response.content), tool_calls_log

        messages.append(response)

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_result = await tool_map[tool_name].ainvoke(tool_args)

            if isinstance(tool_result, list):
                tool_text = "\n".join(
                    b["text"] for b in tool_result
                    if isinstance(b, dict) and b.get("type") == "text"
                ) or str(tool_result)
            else:
                tool_text = str(tool_result)

            tool_calls_log.append({
                "tool": tool_name,
                "input": tool_args,
                "output": tool_text,
            })

            messages.append(ToolMessage(
                content=tool_text,
                tool_call_id=tool_call["id"],
            ))

    return _extract_content(response.content), tool_calls_log


def run_with_tools(
    user_input: str,
    history: list[dict],
    temperature: float,
    max_tokens: int,
    k: int = 4,
) -> tuple[str, list[dict]]:
    try:
        return asyncio.run(_run_async(user_input, history, temperature, max_tokens, k))
    except Exception as exc:
        logger.error(f"[Tools] All retries exhausted — {type(exc).__name__}: {exc}")
        return FALLBACK_MESSAGE, []
