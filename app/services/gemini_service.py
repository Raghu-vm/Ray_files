from functools import lru_cache
import logging
from typing import Any

import anyio
import google.generativeai as genai

from ..config import get_settings

logger = logging.getLogger("ray.gemini")


@lru_cache
def _configure() -> bool:
    settings = get_settings()
    genai.configure(api_key=settings.gemini_api_key)
    return True


def get_chat_model(tools: list[dict] | None = None, system_instruction: str | None = None):
    _configure()
    settings = get_settings()
    return genai.GenerativeModel(
        model_name=settings.gemini_chat_model,
        tools=tools,
        system_instruction=system_instruction,
    )


async def generate_text(prompt: str) -> str:
    _configure()
    settings = get_settings()
    model = genai.GenerativeModel(model_name=settings.gemini_chat_model)

    def _call() -> Any:
        return model.generate_content(prompt)

    try:
        response = await anyio.to_thread.run_sync(_call)
    except Exception:
        logger.exception("Gemini generate_content failed")
        raise

    text = getattr(response, "text", None)
    if text:
        return text

    candidates = getattr(response, "candidates", None) or []
    parts: list[str] = []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        if not content:
            continue
        for part in content.parts or []:
            part_text = getattr(part, "text", None)
            if part_text:
                parts.append(part_text)
    return "\n".join(parts).strip()
