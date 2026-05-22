import json
import logging
from typing import Any, Dict, Iterable, List, Tuple

import anyio
from google.generativeai import types as genai_types

from ..config import get_settings
from ..services.llm import get_gemini_model
from ..tools.documents import get_file_contents, list_documents
from ..tools.rag import rag_search
from ..tools.sql import query_document_rows

logger = logging.getLogger("ray.agent")

SYSTEM_PROMPT = """You are a strict knowledge assistant that answers questions using a structured document knowledge base.

You have access to tools that allow you to:
- Perform RAG search over the 'documents' table
- Retrieve document metadata from 'document_metadata'
- Extract full document content when needed
- Query structured/tabular data from 'document_rows' using SQL

---

## CORE BEHAVIOR

1. ALWAYS begin by performing a RAG search to find relevant information.

2. If the query requires numerical aggregation or structured analysis (e.g., sums, averages, comparisons), use SQL queries on the 'document_rows' table instead of RAG.

3. If RAG results are insufficient:
   - Identify relevant documents from metadata
   - Retrieve and analyze their contents

---

## STRICT RULES (NON-NEGOTIABLE)

- Answer ONLY using retrieved context from the knowledge base
- DO NOT use prior knowledge or assumptions
- DO NOT hallucinate or fabricate information
- If no relevant information is found, respond exactly:
  "No relevant information found in the knowledge base."

---

## SOURCE & CONFIDENCE REQUIREMENTS

You MUST include:
- The exact document name (file_title) from the retrieved context
- A confidence score based on how directly the answer is supported

Confidence scoring:
- High (80–100%): Answer clearly present in a single document
- Medium (50–79%): Answer derived from multiple pieces of context
- Low (0–49%): Weak or partial relevance

---

## OUTPUT FORMAT (MANDATORY)

Respond in EXACTLY this format:

Answer: <your answer>



***Source: <document name>***

***Confidence: <percentage>***"""

TOOL_DECLARATIONS = [
    {
        "name": "rag_search",
        "description": "Use RAG to look up information in the knowledgebase.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "list_documents",
        "description": "Use this tool to fetch all available documents, including the table schema if the file is a CSV or Excel file.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_file_contents",
        "description": "Given a file ID, fetches the text from the document.",
        "parameters": {
            "type": "object",
            "properties": {"file_id": {"type": "string"}},
            "required": ["file_id"],
        },
    },
    {
        "name": "query_document_rows",
        "description": "Run a SQL query - use this to query from the document_rows table once you know the file ID you are querying. dataset_id is the file_id and you are always using the row_data for filtering, which is a jsonb field that has all the keys from the file schema given in the document_metadata table.",
        "parameters": {
            "type": "object",
            "properties": {"sql_query": {"type": "string"}},
            "required": ["sql_query"],
        },
    },
]

TOOLS = [{"function_declarations": TOOL_DECLARATIONS}]


class RayAgent:
    def __init__(self, pool) -> None:
        self.pool = pool
        self.settings = get_settings()
        self.model = get_gemini_model(tools=TOOLS, system_instruction=SYSTEM_PROMPT)

    async def run(
        self, chat_input: str, history_messages: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        rag_sources = await rag_search(self.pool, chat_input)
        history = self._build_history(history_messages)
        chat = self.model.start_chat(history=history)
        user_message = self._build_user_message(chat_input, rag_sources)
        result_text, sources = await self._tool_loop(
            chat, user_message, rag_sources, chat_input
        )
        return {"output": result_text or "", "sources": sources}

    def _build_history(self, history_messages: Iterable[Dict[str, str]]):
        history = []
        for message in history_messages:
            role = "user" if message.get("message_type") == "user" else "model"
            content = message.get("content") or ""
            history.append({"role": role, "parts": [{"text": content}]})
        return history

    def _build_user_message(self, chat_input: str, rag_sources: List[Dict[str, Any]]) -> str:
        rag_payload = json.dumps(rag_sources, ensure_ascii=False, default=str)
        return f"RAG_RESULTS:\n{rag_payload}\n\nUSER_QUESTION:\n{chat_input}"

    async def _tool_loop(
        self,
        chat,
        user_message: str,
        rag_sources: List[Dict[str, Any]],
        default_query: str,
    ) -> Tuple[str | None, List[Dict[str, Any]]]:
        response = await self._send_message(chat, user_message)
        sources = rag_sources

        for _ in range(self.settings.agent_max_tool_iterations):
            function_call = self._extract_function_call(response)
            if not function_call:
                return self._extract_text(response), sources

            tool_name, tool_args = function_call
            tool_result = await self._dispatch_tool(tool_name, tool_args, default_query)
            tool_result = self._json_safe(tool_result)

            if tool_name == "rag_search":
                sources = tool_result.get("sources") or sources

            tool_part = self._build_tool_part(tool_name, tool_result)
            response = await self._send_message(chat, tool_part)

        return self._extract_text(response), sources

    async def _send_message(self, chat, message):
        return await anyio.to_thread.run_sync(chat.send_message, message)

    def _extract_function_call(self, response) -> Tuple[str, Any] | None:
        function_calls = getattr(response, "function_calls", None)
        if function_calls:
            call = function_calls[0]
            return call.name, call.args

        candidates = getattr(response, "candidates", None) or []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            if not content:
                continue
            for part in content.parts or []:
                function_call = getattr(part, "function_call", None)
                if not function_call:
                    continue
                name = getattr(function_call, "name", None)
                args = getattr(function_call, "args", None)
                if name is not None:
                    return name, args
                if isinstance(function_call, dict):
                    name = function_call.get("name")
                    args = function_call.get("args")
                    if name is not None:
                        return name, args
        return None

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

    def _normalize_args(self, args: Any) -> Dict[str, Any]:
        if args is None:
            return {}
        if isinstance(args, dict):
            return args
        if hasattr(args, "items"):
            return dict(args)
        if isinstance(args, str):
            try:
                return json.loads(args)
            except json.JSONDecodeError:
                return {}
        return {}

    def _json_safe(self, value: Any) -> Any:
        try:
            return json.loads(json.dumps(value, ensure_ascii=False, default=str))
        except Exception:
            return value

    def _build_tool_part(self, tool_name: str, tool_result: Dict[str, Any]):
        try:
            return genai_types.Part(
                function_response=genai_types.FunctionResponse(
                    name=tool_name, response=tool_result
                )
            )
        except Exception:
            return {"function_response": {"name": tool_name, "response": tool_result}}

    async def _dispatch_tool(
        self, tool_name: str, tool_args: Any, default_query: str
    ) -> Dict[str, Any]:
        args = self._normalize_args(tool_args)
        try:
            if tool_name == "list_documents":
                documents = await list_documents(self.pool)
                return {"documents": documents}

            if tool_name == "get_file_contents":
                file_id = args.get("file_id") or args.get("fileId")
                if not file_id:
                    return {"error": "file_id is required"}
                document_text = await get_file_contents(self.pool, file_id)
                return {"document_text": document_text, "file_id": file_id}

            if tool_name == "query_document_rows":
                sql_query = args.get("sql_query") or args.get("query")
                if not sql_query:
                    return {"error": "sql_query is required"}
                rows = await query_document_rows(self.pool, sql_query)
                return {"rows": rows}

            if tool_name == "rag_search":
                query = args.get("query") or default_query
                sources = await rag_search(self.pool, query)
                return {"sources": sources}

            return {"error": f"Unknown tool: {tool_name}"}
        except Exception:
            logger.exception("Tool execution failed: %s", tool_name)
            return {"error": f"Tool execution failed: {tool_name}"}
