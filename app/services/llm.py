from functools import lru_cache

import cohere
import google.generativeai as genai

from ..config import get_settings


@lru_cache
def _configure_gemini() -> bool:
    settings = get_settings()
    genai.configure(api_key=settings.gemini_api_key)
    return True


def get_gemini_model(tools=None, system_instruction: str | None = None):
    _configure_gemini()
    settings = get_settings()
    return genai.GenerativeModel(
        model_name=settings.gemini_chat_model,
        tools=tools,
        system_instruction=system_instruction,
    )


def embed_text(text: str) -> list[float]:
    _configure_gemini()
    settings = get_settings()
    result = genai.embed_content(
        model=settings.gemini_embedding_model,
        content=text,
        task_type="retrieval_query",
    )
    return result["embedding"]


@lru_cache
def get_cohere_client() -> cohere.Client:
    settings = get_settings()
    return cohere.Client(settings.cohere_api_key)
