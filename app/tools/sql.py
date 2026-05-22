from typing import Any, Dict, List


async def query_document_rows(pool, sql_query: str) -> List[Dict[str, Any]]:
    rows = await pool.fetch(sql_query)
    return [dict(row) for row in rows]
