import logging
from typing import Any, List

import anyio
import google.generativeai as genai

from ..config import get_settings

logger = logging.getLogger("ray.embedding")


def _configure() -> None:
    settings = get_settings()
    genai.configure(api_key=settings.gemini_api_key)


def _embed(text: str, task_type: str) -> List[float]:
    settings = get_settings()
    result = genai.embed_content(
        model=settings.gemini_embedding_model,
        content=text,
        task_type=task_type,
    )
    return result["embedding"]


async def embed_query(text: str) -> List[float]:
    _configure()
    try:
        return await anyio.to_thread.run_sync(_embed, text, "retrieval_query")
    except Exception:
        logger.exception("Query embedding failed")
        raise


async def embed_document(text: str) -> List[float]:
    _configure()
    try:
        return await anyio.to_thread.run_sync(_embed, text, "retrieval_document")
    except Exception:
        logger.exception("Document embedding failed")
        raise
