"""SSE 格式化与响应封装。

本文件在 backend_langgraph/ 与 backend_agents/ 中逐字相同。
前端只依赖这里产出的协议：event: sentiment / token / done / error。
"""

from __future__ import annotations

import json
from typing import AsyncIterator

from fastapi.responses import StreamingResponse


def sse_event(event: str, data: dict) -> str:
    """把一条事件序列化为 SSE 文本块（一个事件以空行结束）。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def sse_response(generator: AsyncIterator[str]) -> StreamingResponse:
    """包装成 StreamingResponse，并关闭沿途所有可能的缓冲。"""
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # 禁止 Nginx 缓冲，否则 token 会被攒批、打字机效果消失。
            "X-Accel-Buffering": "no",
        },
    )
