import asyncpg
import asyncio
from pgvector.asyncpg import register_vector

from .config import get_settings

_pool: asyncpg.Pool | None = None


async def _init_connection(conn: asyncpg.Connection) -> None:
    await register_vector(conn)


async def init_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        settings = get_settings()
        last_error: Exception | None = None
        for attempt in range(1, 6):
            try:
                _pool = await asyncpg.create_pool(
                    dsn=settings.postgres_url,
                    min_size=1,
                    max_size=1,
                    init=_init_connection,
                    timeout=30,
                )
                async with _pool.acquire() as conn:
                    await conn.execute("SELECT 1;")
                break
            except Exception as exc:
                last_error = exc
                if _pool is not None:
                    await _pool.close()
                    _pool = None
                if attempt == 5:
                    raise
                await asyncio.sleep(min(2 * attempt, 8))
        if _pool is None and last_error is not None:
            raise last_error
    return _pool


async def get_pool() -> asyncpg.Pool:
    if _pool is None:
        await init_pool()
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


async def ensure_chat_history_table(pool: asyncpg.Pool) -> None:
    await pool.execute(
        """
        CREATE TABLE IF NOT EXISTS public.ray_chat_history (
          id BIGSERIAL PRIMARY KEY,
          session_id TEXT NOT NULL,
          message_type TEXT NOT NULL CHECK (message_type IN ('user', 'assistant')),
          content TEXT NOT NULL,
          source TEXT,
          confidence TEXT,
          created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
    )
    await pool.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_ray_chat_history_session_created_at
        ON public.ray_chat_history (session_id, created_at DESC);
        """
    )
