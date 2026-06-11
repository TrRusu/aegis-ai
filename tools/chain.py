"""Runs the multi-turn tool-calling conversation loop with an injected LLM.
"""

import asyncio
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from app.config import CVE_SERVER_PATH
from observability.logging_setup import log_llm_call, logger
from observability.fault_tolerance import FaultTolerance, FALLBACK_MESSAGE

_ft = FaultTolerance()
from app.utils import extract_text
from prompts.tool_chain import SYSTEM_PROMPT

MAX_ROUNDS = 5


class ToolChain:
    
    def __init__(self, llm: BaseChatModel):
        self._llm = llm

    @_ft.retry()
    async def _invoke_llm(self, llm_with_tools, messages):
        return await _ft.invoke_with_timeout(llm_with_tools.ainvoke(messages))

    @log_llm_call("Tools")
    async def _run_async(
        self,
        user_input: str,
        history: list[dict],
        tools: list,
    ) -> tuple[str, list[dict]]:
        mcp_client = MultiServerMCPClient({
            "cve": {
                "command": "python",
                "args": [CVE_SERVER_PATH],
                "transport": "stdio",
            }
        })
        mcp_tools = await mcp_client.get_tools()
        all_tools = tools + mcp_tools
        tool_map = {t.name: t for t in all_tools}

        llm_with_tools = self._llm.bind_tools(all_tools)

        messages = [SystemMessage(content=SYSTEM_PROMPT)]
        for msg in history[:-1]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        messages.append(HumanMessage(content=user_input))

        tool_calls_log = []

        for _ in range(MAX_ROUNDS):
            response = await self._invoke_llm(llm_with_tools, messages)

            if not response.tool_calls:
                return extract_text(response.content), tool_calls_log

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

        return extract_text(response.content), tool_calls_log

    def run(
        self,
        user_input: str,
        history: list[dict],
        tools: list,
    ) -> tuple[str, list[dict]]:
        try:
            return asyncio.run(self._run_async(user_input, history, tools))
        except Exception as exc:
            logger.error(f"[Tools] All retries exhausted — {type(exc).__name__}: {exc}")
            return FALLBACK_MESSAGE, []
