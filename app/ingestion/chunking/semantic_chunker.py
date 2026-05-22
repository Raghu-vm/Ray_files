import logging
import re
from typing import Dict, List

from ...services.gemini_service import generate_text

logger = logging.getLogger("ray.chunker")

MAX_CHUNK_SIZE = 1000
MIN_CHUNK_SIZE = 400


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


async def semantic_chunk(text: str) -> List[Dict[str, int | str]]:
    document_content = text or ""
    if not document_content:
        return []

    remaining_text = _clean_text(document_content)
    chunks: List[Dict[str, int | str]] = []
    chunk_number = 1

    if len(remaining_text) <= MAX_CHUNK_SIZE:
        chunks.append(
            {
                "content": remaining_text,
                "chunk": chunk_number,
                "chunk_size": len(remaining_text),
            }
        )
    else:
        while remaining_text:
            text_to_analyze = remaining_text[:MAX_CHUNK_SIZE]
            prompt_text = (
                "You are analyzing a document to find the best transition point to split it into meaningful sections.\n\n"
                "Your goal:\nKeep related content together and split where topics naturally transition.\n\n"
                "Read this text carefully and identify where one topic/section ends and another begins.\n\n"
                f"{text_to_analyze}\n\n"
                f"Find the best transition point BEFORE character position {MAX_CHUNK_SIZE}.\n\n"
                "Look for:\n"
                "- section headings\n"
                "- topic changes\n"
                "- paragraph boundaries\n"
                "- complete conclusions\n"
                "- natural transitions\n\n"
                "Return ONLY the LAST WORD before the split."
            )

            break_point = MAX_CHUNK_SIZE
            try:
                response_text = await generate_text(prompt_text)
                break_word = (response_text or "").strip()

                if break_word:
                    word_index = text_to_analyze.rfind(break_word)
                    if word_index != -1:
                        break_point = word_index + len(break_word)
                        while (
                            break_point < len(text_to_analyze)
                            and text_to_analyze[break_point] in ".!?,;: "
                        ):
                            break_point += 1
                            if text_to_analyze[break_point - 1] == " ":
                                break
                        break_point = min(break_point, MAX_CHUNK_SIZE)
            except Exception as exc:
                logger.warning("LLM failed to determine breakpoint: %s", exc)
                break_point = MAX_CHUNK_SIZE

            chunk_text = remaining_text[:break_point].strip()
            if chunk_text:
                chunks.append(
                    {
                        "content": chunk_text,
                        "chunk": chunk_number,
                        "chunk_size": len(chunk_text),
                    }
                )
                chunk_number += 1

            remaining_text = remaining_text[break_point:].strip()
            if not remaining_text:
                break

    i = 0
    while i < len(chunks):
        if int(chunks[i]["chunk_size"]) < MIN_CHUNK_SIZE:
            if (
                i + 1 < len(chunks)
                and int(chunks[i]["chunk_size"]) + int(chunks[i + 1]["chunk_size"])
                <= MAX_CHUNK_SIZE
            ):
                chunks[i]["content"] = f"{chunks[i]['content']} {chunks[i + 1]['content']}"
                chunks[i]["chunk_size"] = len(str(chunks[i]["content"]))
                chunks.pop(i + 1)
            elif (
                i > 0
                and int(chunks[i - 1]["chunk_size"]) + int(chunks[i]["chunk_size"])
                <= MAX_CHUNK_SIZE
            ):
                chunks[i - 1]["content"] = f"{chunks[i - 1]['content']} {chunks[i]['content']}"
                chunks[i - 1]["chunk_size"] = len(str(chunks[i - 1]["content"]))
                chunks.pop(i)
            else:
                i += 1
        else:
            i += 1

    return chunks
