import logging
from typing import List

import anyio
import cohere

from ..config import get_settings

logger = logging.getLogger("ray.rerank")


def _get_client() -> cohere.Client:
    settings = get_settings()
    return cohere.Client(settings.cohere_api_key)


async def rerank(query: str, documents: List[str], top_n: int) -> List[cohere.RerankResult]:
    if not documents:
        return []
    settings = get_settings()
    client = _get_client()

    def _call():
        return client.rerank(
            model=settings.cohere_rerank_model,
            query=query,
            documents=documents,
            top_n=top_n,
        )

    try:
        response = await anyio.to_thread.run_sync(_call)
        return response.results or []
    except Exception:
        logger.exception("Rerank failed")
        raise
