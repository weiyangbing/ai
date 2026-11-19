"""Agents 侧的手动多轮会话历史。

Agents SDK 没有 LangGraph 那样的 checkpointer，需要应用层按 thread_id
自维护消息列表；语义与 LangGraph 的 InMemorySaver 对齐（进程内、单 worker）。
"""

from __future__ import annotations

from threading import Lock


class ConversationStore:
    def __init__(self) -> None:
        self._sessions: dict[str, list[dict[str, str]]] = {}
        self._lock = Lock()

    def get(self, thread_id: str) -> list[dict[str, str]]:
        """返回历史副本，调用方修改不会影响存储。"""
        with self._lock:
            return [dict(item) for item in self._sessions.get(thread_id, [])]

    def append(self, thread_id: str, role: str, content: str) -> None:
        with self._lock:
            self._sessions.setdefault(thread_id, []).append(
                {"role": role, "content": content}
            )

    def clear(self, thread_id: str) -> None:
        with self._lock:
            self._sessions.pop(thread_id, None)

    def session_count(self) -> int:
        with self._lock:
            return len(self._sessions)
