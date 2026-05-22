import json
import logging
import re
from typing import Any, Dict, List, Tuple

from google.generativeai import types as genai_types
from sqlalchemy.ext.asyncio import AsyncSession

from ..tools.file_content_tool import get_file_contents
from ..tools.metadata_tool import list_documents
from ..tools.rag_tool import rag_search
from ..tools.sql_query_tool import query_document_rows

logger = logging.getLogger("ray.agent.tools")

NUMERIC_QUERY_RE = re.compile(
    r"\b(count|sum|average|avg|median|mean|max|min|total|percent|percentage|how many)\b",
    re.IGNORECASE,
)

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


class ToolRouter:
    def __init__(self) -> None:
        self.sql_used = False

    @staticmethod
    def numeric_query_required(query: str) -> bool:
        return bool(NUMERIC_QUERY_RE.search(query or ""))

    def reset(self) -> None:
        self.sql_used = False

    def extract_function_call(self, response) -> Tuple[str, Any] | None:
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

    def build_tool_part(self, tool_name: str, tool_result: Dict[str, Any]):
        try:
            return genai_types.Part(
                function_response=genai_types.FunctionResponse(
                    name=tool_name, response=tool_result
                )
            )
        except Exception:
            return {"function_response": {"name": tool_name, "response": tool_result}}

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

    async def dispatch_tool(
        self,
        session: AsyncSession,
        tool_name: str,
        tool_args: Any,
        default_query: str,
    ) -> Dict[str, Any]:
        args = self._normalize_args(tool_args)
        try:
            if tool_name == "list_documents":
                documents = await list_documents(session)
                return {"documents": documents}

            if tool_name == "get_file_contents":
                file_id = args.get("file_id") or args.get("fileId")
                if not file_id:
                    return {"error": "file_id is required"}
                document_text = await get_file_contents(session, file_id)
                return {"document_text": document_text, "file_id": file_id}

            if tool_name == "query_document_rows":
                self.sql_used = True
                sql_query = args.get("sql_query") or args.get("query")
                if not sql_query:
                    return {"error": "sql_query is required"}
                rows = await query_document_rows(session, sql_query)
                return {"rows": rows}

            if tool_name == "rag_search":
                query = args.get("query") or default_query
                sources = await rag_search(session, query)
                return {"sources": sources}

            return {"error": f"Unknown tool: {tool_name}"}
        except Exception:
            logger.exception("Tool execution failed: %s", tool_name)
            return {"error": f"Tool execution failed: {tool_name}"}
