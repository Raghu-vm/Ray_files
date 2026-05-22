from typing import Any, Dict, List


async def list_documents(pool) -> List[Dict[str, Any]]:
    rows = await pool.fetch("SELECT * FROM public.document_metadata;")
    return [dict(row) for row in rows]


async def get_file_contents(pool, file_id: str) -> str:
    row = await pool.fetchrow(
        """
        SELECT string_agg(text, ' ') as document_text
        FROM documents_pg
        WHERE metadata->>'file_id' = $1
        GROUP BY metadata->>'file_id';
        """,
        file_id,
    )
    if not row:
        return ""
    return row["document_text"] or ""
