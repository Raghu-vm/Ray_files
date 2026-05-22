import logging

from fastapi import FastAPI, Request

from .db import close_pool, ensure_chat_history_table, init_pool
from .memory import get_session_history, insert_chat_message
from .services.agent import RayAgent
from .utils import build_response_payload, extract_request_payload, fallback_response

logger = logging.getLogger("ray")
logging.basicConfig(level=logging.INFO)

app = FastAPI()


@app.on_event("startup")
async def on_startup() -> None:
    pool = await init_pool()
    await ensure_chat_history_table(pool)
    app.state.agent = RayAgent(pool)


@app.on_event("shutdown")
async def on_shutdown() -> None:
    await close_pool()


@app.post("/ray-rag-model")
async def ray_rag_model(request: Request):
    payload = await extract_request_payload(request)
    chat_input = payload["chatInput"]
    session_id = payload["sessionId"]

    history_messages = []
    try:
        history_messages = await get_session_history(session_id)
    except Exception:
        logger.exception("Failed to load session history")

    try:
        await insert_chat_message(session_id, "user", chat_input)
    except Exception:
        logger.exception("Failed to insert user message")

    try:
        agent: RayAgent = app.state.agent
        agent_result = await agent.run(chat_input, history_messages)
        response = build_response_payload(agent_result)
        if not response["answer"]:
            response = fallback_response()
    except Exception:
        logger.exception("Agent execution failed")
        response = fallback_response()

    try:
        await insert_chat_message(
            session_id,
            "assistant",
            response["answer"],
            response["source"],
            response["confidence"],
        )
    except Exception:
        logger.exception("Failed to insert assistant message")

    return response
