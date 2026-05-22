import time
import json
from typing import Any, Dict, Iterable

from fastapi import Request


def _get_nested(obj: Any, *keys: str) -> Any:
    current = obj
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _first_truthy(values: Iterable[Any]) -> Any:
    for value in values:
        if value:
            return value
    return ""


def _first_not_none(values: Iterable[Any]) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


async def extract_request_payload(request: Request) -> Dict[str, Any]:
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            payload = {}
    except Exception:
        payload = {}

    query = dict(request.query_params)
    headers = dict(request.headers)

    chat_input = _first_truthy(
        [
            payload.get("chatInput"),
            _get_nested(payload, "body", "chatInput"),
            _get_nested(payload, "query", "chatInput"),
            payload.get("text"),
            payload.get("message"),
            query.get("chatInput"),
            query.get("text"),
            query.get("message"),
        ]
    )

    session_id = _first_truthy(
        [
            payload.get("sessionId"),
            _get_nested(payload, "body", "sessionId"),
            _get_nested(payload, "query", "sessionId"),
            _get_nested(payload, "chat", "sessionId"),
            _get_nested(payload, "metadata", "sessionId"),
            _get_nested(payload, "headers", "x-session-id"),
            headers.get("x-session-id"),
        ]
    )

    if not session_id:
        session_id = f"session-{int(time.time() * 1000)}"

    return {"chatInput": chat_input or "", "sessionId": session_id}


def _get_source_from_sources(sources: list[dict] | None) -> str:
    if not sources:
        return ""
    first = sources[0] or {}
    metadata = first.get("metadata")
    if isinstance(metadata, str):
        try:
            parsed = json.loads(metadata)
            if isinstance(parsed, dict):
                metadata = parsed
        except json.JSONDecodeError:
            metadata = None
    return (
        first.get("title")
        or first.get("file_title")
        or first.get("name")
        or first.get("source")
        or (metadata.get("file_title") if isinstance(metadata, dict) else None)
        or (metadata.get("title") if isinstance(metadata, dict) else None)
        or _get_nested(first, "metadata", "file_title")
        or _get_nested(first, "metadata", "title")
        or _get_nested(first, "document", "metadata", "file_title")
        or ""
    )


def _get_confidence_from_sources(sources: list[dict] | None) -> Any:
    if not sources:
        return None
    first = sources[0] or {}
    return _first_not_none(
        [
            first.get("confidence"),
            first.get("score"),
            first.get("relevance_score"),
            _get_nested(first, "metadata", "score"),
        ]
    )


def build_response_payload(agent_result: Dict[str, Any]) -> Dict[str, Any]:
    answer = _first_truthy(
        [
            agent_result.get("output"),
            agent_result.get("answer"),
            agent_result.get("response"),
            agent_result.get("text"),
            agent_result.get("message"),
            "",
        ]
    )

    source = _first_truthy(
        [
            agent_result.get("source"),
            _get_nested(agent_result, "metadata", "file_title"),
            _get_source_from_sources(agent_result.get("sources")),
            "",
        ]
    )

    confidence_value = _first_not_none(
        [
            agent_result.get("confidence"),
            agent_result.get("score"),
            _get_confidence_from_sources(agent_result.get("sources")),
        ]
    )
    confidence = "" if confidence_value is None else str(confidence_value)

    return {
        "answer": answer,
        "source": source,
        "confidence": confidence,
        "sources": agent_result.get("sources") or [],
    }


def fallback_response() -> Dict[str, Any]:
    return {
        "answer": "No relevant information found in the knowledge base.",
        "source": "",
        "confidence": "",
        "sources": [],
    }
