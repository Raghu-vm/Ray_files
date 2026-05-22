import json
import logging
from typing import Any, Dict, List

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import get_settings
from ..db.models import DocumentPG
from ..services.embedding_service import embed_query
from ..services.rerank_service import rerank

logger = logging.getLogger("ray.rag")


async def rag_search(session: AsyncSession, query: str) -> List[Dict[str, Any]]:
    settings = get_settings()
    if not query:
        return []

    try:
        embedding = await embed_query(query)
    except Exception:
        logger.exception("Embedding failed")
        return []

    try:
        distance_expr = DocumentPG.embedding.cosine_distance(embedding)
        stmt = (
            select(
                DocumentPG.id,
                DocumentPG.content,
                DocumentPG.metadata_.label("metadata"),
                distance_expr.label("distance"),
            )
            .order_by(distance_expr)
            .limit(settings.rag_top_k)
        )
        result = await session.execute(stmt)
        rows = result.all()
    except Exception:
        logger.exception("Vector search failed")
        return []

    docs: List[Dict[str, Any]] = []
    for row in rows:
        metadata: Dict[str, Any]
        raw_metadata = row.metadata
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

        distance = row.distance
        docs.append(
            {
                "id": str(row.id),
                "content": row.content,
                "metadata": metadata,
                "distance": float(distance) if distance is not None else None,
                "score": float(1 - distance) if distance is not None else None,
            }
        )

    if not docs:
        return []

    try:
        rerank_results = await rerank(
            query=query,
            documents=[doc["content"] for doc in docs],
            top_n=settings.rag_top_n,
        )
        ranked: List[Dict[str, Any]] = []
        for item in rerank_results:
            doc = docs[item.index]
            doc["relevance_score"] = item.relevance_score
            ranked.append(doc)
        return ranked
    except Exception:
        logger.exception("Rerank failed")
        return docs[: settings.rag_top_n]
