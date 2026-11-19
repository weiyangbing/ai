"""两个后端共享的数据契约。

本文件在 backend_langgraph/ 与 backend_agents/ 中逐字相同。
只放与框架无关的内容：Pydantic 模型、枚举、以及把模型输出收敛为
SentimentResult 的兜底解析（DeepSeek 结构化输出不稳定时的保险，见设计文档 §12.2）。
"""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SentimentLabel(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class SentimentResult(BaseModel):
    label: SentimentLabel
    confidence: float = Field(default=0.85, ge=0.0, le=1.0)


class ChatRequest(BaseModel):
    thread_id: str
    message: str


# 只保留“分类词”本身，避免把“我不开心”这类文本里的情绪词误当成标签。
_LABEL_KEYWORDS: tuple[tuple[SentimentLabel, tuple[str, ...]], ...] = (
    (SentimentLabel.POSITIVE, ("positive", "积极", "正面")),
    (SentimentLabel.NEGATIVE, ("negative", "消极", "负面")),
    (SentimentLabel.NEUTRAL, ("neutral", "中性")),
)


def _label_from_text(text: str) -> SentimentLabel | None:
    lowered = text.strip().lower()
    if not lowered:
        return None
    for label, keywords in _LABEL_KEYWORDS:
        if any(keyword in lowered for keyword in keywords):
            return label
    return None


def _clamp_confidence(value: Any, default: float = 0.85) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, confidence))


def coerce_sentiment(value: Any) -> SentimentResult:
    """把模型输出（Pydantic / dict / 纯文本）统一收敛为合法的 SentimentResult。

    正常路径下模型会给出结构化结果；这里额外兼容“只输出一个单词或一段 JSON”
    的退化路径，保证编排层拿到的永远是合法对象，而不是在推流中途炸掉。
    """
    if isinstance(value, SentimentResult):
        return value

    candidate: dict[str, Any] | None = None
    if isinstance(value, dict):
        candidate = value
    elif hasattr(value, "model_dump"):
        dumped = value.model_dump()
        if isinstance(dumped, dict):
            candidate = dumped

    if candidate is not None:
        label = _label_from_text(str(candidate.get("label", "")))
        if label is not None:
            return SentimentResult(
                label=label,
                confidence=_clamp_confidence(candidate.get("confidence")),
            )

    text = value if isinstance(value, str) else str(value or "")

    # 退化的模型可能只在文本里包了一段 JSON，先尝试提取。
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            label = _label_from_text(str(parsed.get("label", "")))
            if label is not None:
                return SentimentResult(
                    label=label,
                    confidence=_clamp_confidence(parsed.get("confidence")),
                )

    # 最兜底：全文里找标签关键词，找不到按 neutral 处理。
    label = _label_from_text(text) or SentimentLabel.NEUTRAL
    return SentimentResult(label=label, confidence=0.85)
