"""OpenAI Agents 后端的编排层。

对外只暴露 chat_stream(req) -> AsyncIterator[str]（产出 SSE 文本块）。
流式必须分两阶段：情感识别（非流式）→ 路由 → 回应生成（流式），
因为路由依赖情感识别的最终标签。
"""

from __future__ import annotations

import logging
import os
from contextlib import suppress
from typing import AsyncIterator

from agents import Runner

from .agents import (
    encourage_agent,
    sentiment_agent,
    sentiment_text_agent,
    soothe_agent,
)
from .contracts import ChatRequest, SentimentLabel, SentimentResult, coerce_sentiment
from .conversation_store import ConversationStore
from .sse import sse_event

logger = logging.getLogger(__name__)

# 模块级单例：按 thread_id 隔离的会话历史（进程内）。
_store = ConversationStore()

# 进程级开关：结构化情感识别一旦失败，后续直接走文本兜底。
_structured_sentiment_ok = True


async def startup() -> None:
    if not os.getenv("DEEPSEEK_API_KEY"):
        logger.warning(
            "DEEPSEEK_API_KEY 未设置，/chat 将返回 error 事件；"
            "请在 backend_agents/.env 中配置（或使用 scripts/mock_deepseek.py 联调）。"
        )


async def shutdown() -> None:
    return None


async def _classify(message: str) -> SentimentResult:
    """只用当前轮用户消息做情感识别，不混入历史。"""
    global _structured_sentiment_ok

    if _structured_sentiment_ok:
        try:
            result = await Runner.run(sentiment_agent, message)
            return coerce_sentiment(result.final_output)
        except Exception as exc:
            logger.warning("结构化情感识别失败，后续退化为文本解析：%s", exc)
            _structured_sentiment_ok = False

    result = await Runner.run(sentiment_text_agent, message)
    return coerce_sentiment(result.final_output)


def _delta_from_event(event) -> str | None:
    """从 stream_events() 的事件中提取文本 delta。

    Chat Completions 模型在 SDK 内部会被转换成 Responses 风格事件，
    文本增量是 response.output_text.delta。字段路径随 SDK 版本可能变化，
    因此用 getattr 做宽容读取（设计文档 §12.1）。
    """
    if getattr(event, "type", None) != "raw_response_event":
        return None
    data = getattr(event, "data", None)
    if getattr(data, "type", None) != "response.output_text.delta":
        return None
    delta = getattr(data, "delta", None)
    return delta if isinstance(delta, str) and delta else None


async def chat_stream(req: ChatRequest) -> AsyncIterator[str]:
    # 阶段一：情感识别（非流式，快速完成；路由依赖其最终标签）
    sentiment = await _classify(req.message)
    yield sse_event("sentiment", sentiment.model_dump(mode="json"))

    # 阶段二：流式生成回应（完整历史 + 本轮用户消息）
    history = _store.get(req.thread_id)
    input_messages = [*history, {"role": "user", "content": req.message}]
    target = (
        encourage_agent
        if sentiment.label == SentimentLabel.POSITIVE
        else soothe_agent
    )

    streamed = Runner.run_streamed(target, input=input_messages)
    events = streamed.stream_events()
    chunks: list[str] = []

    try:
        async for event in events:
            delta = _delta_from_event(event)
            if delta:
                chunks.append(delta)
                yield sse_event("token", {"delta": delta})

        # 客户端断开时不会走到这里；只有正常结束才写入会话历史。
        reply = streamed.final_output
        if not isinstance(reply, str) or not reply:
            reply = "".join(chunks)

        _store.append(req.thread_id, "user", req.message)
        _store.append(req.thread_id, "assistant", reply)

        yield sse_event("done", {"framework": "openai-agents"})
    finally:
        # 客户端断开 / 生成器被关闭时，立即取消对 DeepSeek 的付费请求。
        if not streamed.is_complete:
            streamed.cancel()
        with suppress(Exception):
            await events.aclose()
