"""Autonomous breach triage agent with an injected LLM.
"""

import asyncio
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from tools.tools import make_tools
from app.config import CVE_SERVER_PATH
from observability.logging_setup import CallLogger, logger

_call_logger = CallLogger()
from observability.fault_tolerance import FALLBACK_MESSAGE
from prompts.breach_triage_agent import AGENT_SYSTEM_PROMPT


class BreachTriageAgent:

    def __init__(self, llm: BaseChatModel):
        self._llm = llm

    @_call_logger.log_llm_call("Agent")
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

        agent = create_agent(self._llm, all_tools, prompt=AGENT_SYSTEM_PROMPT)
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