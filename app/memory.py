from typing import Any, Dict, List

from .db import get_pool


async def insert_chat_message(
    session_id: str,
    message_type: str,
    content: str,
    source: str | None = None,
    confidence: str | None = None,
) -> None:
    pool = await get_pool()
    await pool.execute(
        """
        INSERT INTO public.ray_chat_history (
            session_id,
            message_type,
            content,
            source,
            confidence,
            created_at
        )
        VALUES ($1, $2, $3, $4, $5, NOW());
        """,
        session_id,
        message_type,
        content,
        source,
        confidence,
    )


async def get_session_history(session_id: str) -> List[Dict[str, Any]]:
    pool = await get_pool()
    rows = await pool.fetch(
        """
        SELECT message_type, content
        FROM public.ray_chat_history
        WHERE session_id = $1
        ORDER BY created_at ASC, id ASC;
        """,
        session_id,
    )
    return [
        {"message_type": row["message_type"], "content": row["content"]}
        for row in rows
    ]
