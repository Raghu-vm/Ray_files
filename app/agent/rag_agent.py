import json
import logging
from typing import Any, Dict, Iterable, List, Tuple

import anyio

from ..config import get_settings
from ..services.gemini_service import get_chat_model
from ..tools.rag_tool import rag_search
from .prompts import SYSTEM_PROMPT
from .tool_router import TOOLS, ToolRouter

logger = logging.getLogger("ray.agent")


class RagAgent:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.model = get_chat_model(tools=TOOLS, system_instruction=SYSTEM_PROMPT)
        self.tool_router = ToolRouter()

    async def run(
        self, session, chat_input: str, history_messages: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        rag_sources = await rag_search(session, chat_input)
        history = self._build_history(history_messages)
        chat = self.model.start_chat(history=history)
        user_message = self._build_user_message(chat_input, rag_sources)
        response_text, sources, sql_used = await self._tool_loop(
            session, chat, user_message, rag_sources, chat_input
        )

        if self.tool_router.numeric_query_required(chat_input) and not sql_used:
            return {"output": "", "sources": sources}

        return {"output": response_text or "", "sources": sources}

    def _build_history(self, history_messages: Iterable[Dict[str, str]]):
        history = []
        for message in history_messages:
            role = "user" if message.get("message_type") == "user" else "model"
            content = message.get("content") or ""
            history.append({"role": role, "parts": [{"text": content}]})
        return history

    def _build_user_message(self, chat_input: str, rag_sources: List[Dict[str, Any]]) -> str:
        rag_payload = json.dumps(rag_sources, ensure_ascii=True, default=str)
        return f"RAG_RESULTS:\n{rag_payload}\n\nUSER_QUESTION:\n{chat_input}"

    async def _tool_loop(
        self,
        session,
        chat,
        user_message: str,
        rag_sources: List[Dict[str, Any]],
        default_query: str,
    ) -> Tuple[str | None, List[Dict[str, Any]], bool]:
        self.tool_router.reset()
        response = await self._send_message(chat, user_message)
        sources = rag_sources

        for _ in range(self.settings.agent_max_tool_iterations):
            function_call = self.tool_router.extract_function_call(response)
            if not function_call:
                return self._extract_text(response), sources, self.tool_router.sql_used

            tool_name, tool_args = function_call
            tool_result = await self.tool_router.dispatch_tool(
                session, tool_name, tool_args, default_query
            )
            tool_result = self._json_safe(tool_result)

            if tool_name == "rag_search":
                sources = tool_result.get("sources") or sources

            tool_part = self.tool_router.build_tool_part(tool_name, tool_result)
            response = await self._send_message(chat, tool_part)

        return self._extract_text(response), sources, self.tool_router.sql_used

    async def _send_message(self, chat, message):
        return await anyio.to_thread.run_sync(chat.send_message, message)

    def _extract_text(self, response) -> str:
        text = getattr(response, "text", None)
        if text:
            return text

        candidates = getattr(response, "candidates", None) or []
        parts: List[str] = []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            if not content:
                continue
            for part in content.parts or []:
                part_text = getattr(part, "text", None)
                if part_text:
                    parts.append(part_text)
        return "\n".join(parts).strip()

    def _json_safe(self, value: Any) -> Any:
        try:
            return json.loads(json.dumps(value, ensure_ascii=True, default=str))
        except Exception:
            return value
