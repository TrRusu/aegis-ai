import asyncio
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from tools.tools import make_tools
from app.config import OPENAI_API_KEY, OPENAI_MODEL, CVE_SERVER_PATH
from observability.logging_setup import log_llm_call, logger
from observability.fault_tolerance import FALLBACK_MESSAGE


AGENT_SYSTEM_PROMPT = """You are Aegis Triage, an autonomous cybersecurity breach analyst.

When given an incident description, you MUST autonomously investigate it without waiting for further instructions. Your job is to:

1. Search the knowledge base for relevant breach statistics (costs, attack vectors, industry benchmarks)
2. Look up any CVEs mentioned or implied by the attack vector
3. Estimate financial impact if enough information is available
4. Produce a structured triage report

Your report must follow this structure:
- **Incident Summary**: what happened
- **Threat Context**: relevant statistics from the IBM report (breach costs, attack vector data)
- **CVE Analysis**: if a specific vulnerability is involved, its severity and details
- **Financial Impact Estimate**: estimated cost range based on industry and record count if known
- **Recommended Actions**: 2-3 concrete next steps

Be autonomous. Do not ask the user for clarification — investigate with the information given and state clearly what is unknown.
"""


class BreachTriageAgent:
    """Autonomous breach triage agent with an injected LLM."""

    def __init__(self, llm: BaseChatModel):
        self._llm = llm

    @log_llm_call("Agent")
    def run(self, task: str, k: int = 4) -> tuple[str, list[dict]]:
        try:
            return asyncio.run(self._run_async(task, k))
        except Exception as exc:
            logger.error(f"[Agent] Failed — {type(exc).__name__}: {exc}")
            return FALLBACK_MESSAGE, []

    async def _run_async(self, task: str, k: int) -> tuple[str, list[dict]]:
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

        agent = create_react_agent(self._llm, all_tools, prompt=AGENT_SYSTEM_PROMPT)
        result = await agent.ainvoke({"messages": [HumanMessage(content=task)]})

        final_message = result["messages"][-1]
        content = final_message.content
        if isinstance(content, list):
            content = "\n".join(
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )

        tool_calls_log = []
        for msg in result["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls_log.append({"tool": tc["name"], "input": tc["args"], "output": ""})
            if hasattr(msg, "name") and msg.name:
                for entry in reversed(tool_calls_log):
                    if entry["tool"] == msg.name and entry["output"] == "":
                        entry["output"] = msg.content
                        break

        return content, tool_calls_log