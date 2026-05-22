from dataclasses import dataclass
from functools import lru_cache
import os

from dotenv import load_dotenv

load_dotenv()


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if value is None or value == "":
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"Invalid integer for {name}: {value}") from exc


@dataclass(frozen=True)
class Settings:
    postgres_url: str
    gemini_api_key: str
    cohere_api_key: str
    gemini_chat_model: str
    gemini_embedding_model: str
    cohere_rerank_model: str
    rag_top_k: int
    rag_top_n: int
    agent_max_tool_iterations: int


@lru_cache
def get_settings() -> Settings:
    return Settings(
        postgres_url=_require_env("POSTGRES_URL"),
        gemini_api_key=_require_env("GEMINI_API_KEY"),
        cohere_api_key=_require_env("COHERE_API_KEY"),
        gemini_chat_model=os.getenv("GEMINI_CHAT_MODEL", "models/gemini-flash-latest"),
        gemini_embedding_model=os.getenv(
            "GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-2-preview"
        ),
        cohere_rerank_model=os.getenv("COHERE_RERANK_MODEL", "rerank-english-v3.0"),
        rag_top_k=_get_int("RAG_TOP_K", 25),
        rag_top_n=_get_int("RAG_TOP_N", 4),
        agent_max_tool_iterations=_get_int("AGENT_MAX_TOOL_ITERATIONS", 6),
    )
