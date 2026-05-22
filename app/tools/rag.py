import logging
import json
from typing import Any, Dict, List

from ..config import get_settings
from ..services.llm import embed_text, get_cohere_client

logger = logging.getLogger("ray.rag")


async def rag_search(pool, query: str) -> List[Dict[str, Any]]:
    settings = get_settings()
    if not query:
        return []

    try:
        embedding = embed_text(query)
    except Exception:
        logger.exception("Embedding failed")
        return []

    try:
        rows = await pool.fetch(
            """
            SELECT id, text, metadata, (embedding <=> $1) AS distance
            FROM documents_pg
            ORDER BY embedding <=> $1
            LIMIT $2;
            """,
            embedding,
            settings.rag_top_k,
        )
    except Exception:
        logger.exception("Vector search failed")
        return []

    docs: List[Dict[str, Any]] = []
    for row in rows:
        distance = row["distance"]
        raw_metadata = row["metadata"]
        metadata: Dict[str, Any]
        if isinstance(raw_metadata, dict):
            metadata = raw_metadata
        elif isinstance(raw_metadata, str):
            try:
                parsed = json.loads(raw_metadata)
                metadata = parsed if isinstance(parsed, dict) else {"raw": raw_metadata}
            except json.JSONDecodeError:
                metadata = {"raw": raw_metadata}
        else:
            metadata = {}

        docs.append(
            {
                "id": str(row["id"]),
                "text": row["text"],
                "metadata": metadata,
                "distance": float(distance) if distance is not None else None,
                "score": float(1 - distance) if distance is not None else None,
            }
        )

    if not docs:
        return []

    try:
        cohere_client = get_cohere_client()
        rerank = cohere_client.rerank(
            model=settings.cohere_rerank_model,
            query=query,
            documents=[doc["text"] for doc in docs],
            top_n=settings.rag_top_n,
        )
        ranked = []
        for item in rerank.results:
            doc = docs[item.index]
            doc["relevance_score"] = item.relevance_score
            ranked.append(doc)
        return ranked
    except Exception:
        logger.exception("Rerank failed")
        return docs[: settings.rag_top_n]
