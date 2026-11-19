"""FastAPI 应用入口。

本文件在 backend_langgraph/ 与 backend_agents/ 中逐字相同。
框架差异全部收敛在 orchestrator.py 以及各自的图 / Agent 定义中；
main.py 只认识统一的 orchestrator.chat_stream() 契约。
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from . import orchestrator
from .contracts import ChatRequest
from .sse import sse_event, sse_response

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await orchestrator.startup()
    try:
        yield
    finally:
        await orchestrator.shutdown()


app = FastAPI(title="Emotion Support API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本地测试；Docker / 生产阶段收紧
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat")
async def chat(req: ChatRequest, request: Request):
    async def event_stream():
        stream = orchestrator.chat_stream(req)
        try:
            async for chunk in stream:
                if await request.is_disconnected():
                    break
                yield chunk
        except Exception as exc:
            # StreamingResponse 一旦开始发送，HTTP 状态码已经固定，
            # 业务异常只能通过 SSE error 事件告知前端（设计文档 §6.2）。
            logger.exception("chat_stream failed")
            yield sse_event("error", {"message": str(exc)})
        finally:
            try:
                await stream.aclose()
            except Exception:  # 客户端断开时的清理异常不应污染响应
                logger.debug("error while closing chat stream", exc_info=True)

    return sse_response(event_stream())
