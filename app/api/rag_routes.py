import logging

from fastapi import APIRouter, Request

from ..agent.rag_agent import RagAgent
from ..db.session import get_session
from ..memory.postgres_memory import get_session_history, insert_chat_message
from ..utils.helpers import build_response_payload, extract_request_payload, fallback_response

logger = logging.getLogger("ray.api.rag")

router = APIRouter()


@router.post("/ray-rag-model")
async def ray_rag_model(request: Request):
    payload = await extract_request_payload(request)
    chat_input = payload["chatInput"]
    session_id = payload["sessionId"]

    async with get_session() as db_session:
        history_messages = []
        try:
            history_messages = await get_session_history(db_session, session_id)
        except Exception:
            logger.exception("Failed to load session history")

        try:
            await insert_chat_message(db_session, session_id, "user", chat_input)
            await db_session.commit()
        except Exception:
            logger.exception("Failed to insert user message")
            await db_session.rollback()

        try:
            agent: RagAgent = request.app.state.rag_agent
            agent_result = await agent.run(db_session, chat_input, history_messages)
            response = build_response_payload(agent_result)
            if not response["answer"]:
                response = fallback_response()
        except Exception:
            logger.exception("Agent execution failed")
            response = fallback_response()

        try:
            await insert_chat_message(
                db_session,
                session_id,
                "assistant",
                response["answer"],
                response["source"],
                response["confidence"],
            )
            await db_session.commit()
        except Exception:
            logger.exception("Failed to insert assistant message")
            await db_session.rollback()

    return response
