"""OpenAI Agents SDK 的 Agent 定义。

DeepSeek 走 OpenAI 兼容端点，因此必须显式构造
OpenAIChatCompletionsModel，并给每一个 Agent 传 model=model，
否则 SDK 会拿默认 GPT 模型名去请求 DeepSeek，直接 400（设计文档 §8.1）。
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from agents import (
    Agent,
    AsyncOpenAI,
    ModelSettings,
    OpenAIChatCompletionsModel,
    set_tracing_disabled,
)

from .contracts import SentimentResult

load_dotenv()

# 本项目只访问 DeepSeek，关闭向 OpenAI 的 tracing 上报。
set_tracing_disabled(True)

deepseek_client = AsyncOpenAI(
    # 占位 key 让服务在未配置密钥时也能启动；真正的错误会以 SSE error 返回。
    api_key=os.getenv("DEEPSEEK_API_KEY") or "not-configured",
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
)

model = OpenAIChatCompletionsModel(
    model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
    openai_client=deepseek_client,
)

sentiment_agent = Agent(
    name="sentiment_analyzer",
    instructions=(
        "你是情感分析器。判断用户最新一条消息的整体情感倾向，"
        "label 只能是 positive、negative 或 neutral，"
        "confidence 是 0 到 1 之间的置信度。"
    ),
    model=model,
    output_type=SentimentResult,
    model_settings=ModelSettings(temperature=0.0),
)

# 兜底 Agent：DeepSeek 若不支持 json_schema 结构化输出（设计文档 §12.2），
# 退化为“只输出一个 JSON，Python 解析”。
sentiment_text_agent = Agent(
    name="sentiment_analyzer_text",
    instructions=(
        "你是情感分析器。判断用户消息的情感倾向，只输出一个 json 对象，"
        '格式为 {"label": "positive|negative|neutral", "confidence": 0.0}，'
        "confidence 取 0 到 1 之间的数。不要输出 json 以外的任何内容。"
    ),
    model=model,
    model_settings=ModelSettings(temperature=0.0),
)

encourage_agent = Agent(
    name="encourager",
    instructions=(
        "你是一位温暖、真诚的倾听者。用户刚刚表达了正面情绪。"
        "请用中文回应：先具体地肯定对方此刻的开心或成就感，再给出真诚的鼓励。"
        "语气自然，2 到 4 句话，不要使用列表、标题或说教。"
    ),
    model=model,
    model_settings=ModelSettings(temperature=0.8),
)

soothe_agent = Agent(
    name="soother",
    instructions=(
        "你是一位温暖、有耐心的倾听者。用户刚刚表达了负面或中性情绪。"
        "请用中文回应：先准确共情、验证对方的感受，再温和地帮助对方梳理和排解。"
        "不要说教，2 到 4 句话，不要使用列表或标题。"
    ),
    model=model,
    model_settings=ModelSettings(temperature=0.8),
)
