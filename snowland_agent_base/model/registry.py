# Copyright 2026 Snowland
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LLM 后端注册表与工厂 (插拔核心, 类型感知)。

后端按「能力类型」(:class:`~snowland_agent_base.model.types.ModelType`) 分类注册,
同一厂商名称可在不同能力类型下注册不同的后端实现 (例如 ``dashscope`` 既有推理
ChatModel, 也有向量 / 图像 / 视频 / 语音合成后端)。
"""
from __future__ import annotations

from typing import Dict, Optional

from snowland_agent_base.model.backends.anthropic import AnthropicLLMBackend
from snowland_agent_base.model.backends.anthropic_openai import (
    AnthropicOpenAILLMBackend,
)
from snowland_agent_base.model.backends.dashscope import (
    DashScopeEmbeddingBackend,
    DashScopeImageBackend,
    DashScopeLLMBackend,
    DashScopeRerankBackend,
    DashScopeTTSBackend,
    DashScopeVideoBackend,
)
from snowland_agent_base.model.backends.deepseek import (
    DeepSeekEmbeddingBackend,
    DeepSeekLLMBackend,
    DeepSeekRerankBackend,
)
from snowland_agent_base.model.backends.orcarouter import (
    OrcaRouterEmbeddingBackend,
    OrcaRouterImageBackend,
    OrcaRouterLLMBackend,
    OrcaRouterRerankBackend,
    OrcaRouterTTSBackend,
    OrcaRouterVideoBackend,
)
from snowland_agent_base.model.backends.openai_compat import (
    OpenAICompatEmbeddingBackend,
    OpenAICompatImageBackend,
    OpenAICompatLLMBackend,
    OpenAICompatRerankBackend,
    OpenAICompatTTSBackend,
    OpenAICompatVideoBackend,
)
from snowland_agent_base.model.backends.zhipu import (
    ZhipuEmbeddingBackend,
    ZhipuImageBackend,
    ZhipuLLMBackend,
    ZhipuRerankBackend,
    ZhipuTTSBackend,
    ZhipuVideoBackend,
)
from snowland_agent_base.model.base import ModelBackend
from snowland_agent_base.model.types import ModelType
from snowland_agent_base.messages import Message, OpenAIMessage


_BACKENDS: Dict[ModelType, Dict[str, type]] = {}


def register_llm_backend(
    name: str, cls: type, model_type: ModelType = ModelType.REASONING
) -> None:
    """注册一个模型后端, 使其可通过名称字符串被工厂解析。

    Args:
        name: 后端名称或别名。
        cls: 后端类, 必须继承 :class:`ModelBackend` 的某一能力子类。
        model_type: 能力类型, 默认 REASONING (向后兼容旧调用)。
    """
    if not (isinstance(cls, type) and issubclass(cls, ModelBackend)):
        raise TypeError(f"模型后端 {cls!r} 必须继承 ModelBackend 的某一能力子类")
    _BACKENDS.setdefault(model_type, {})[name] = cls


def get_llm_backend(name: str, model_type: ModelType = ModelType.REASONING) -> type:
    """按名称取回指定能力类型的已注册后端类。"""
    table = _BACKENDS.get(model_type, {})
    if name not in table:
        raise KeyError(
            f"未知模型后端: {name!r} (类型 {model_type}); 可用: {list_llm_backends(model_type)}"
        )
    return table[name]


def list_llm_backends(model_type: Optional[ModelType] = None) -> list:
    """列出已注册的后端名称。

    Args:
        model_type: 若给定, 仅列出该能力类型下的后端; 否则列出全部。
    """
    if model_type is None:
        seen = set()
        for table in _BACKENDS.values():
            seen.update(table)
        return sorted(seen)
    return list(_BACKENDS.get(model_type, {}))


def get_message_class(backend: Optional[str]) -> type[Message]:
    """按后端名取回其配套的消息风格类 (``OpenAIMessage`` / ``AnthropicMessage``)。

    仅推理类后端有意义; 非推理后端返回默认的 ``OpenAIMessage``。

    Args:
        backend: 后端名称或已注册别名; ``None`` / 未知名称均回退为 ``OpenAIMessage``。

    Returns:
        消息风格类。
    """
    if not backend:
        return OpenAIMessage
    try:
        return get_llm_backend(backend, ModelType.REASONING).message_class or OpenAIMessage
    except KeyError:
        return OpenAIMessage


def create_model(
    model_type: ModelType = ModelType.REASONING,
    name: str = "dashscope",
    **kwargs,
) -> ModelBackend:
    """按能力类型 + 名称构建一个模型后端实例。"""
    return get_llm_backend(name, model_type)(**kwargs)


def create_llm_backend(name: str = "dashscope", **kwargs) -> ModelBackend:
    """按名称构建一个推理 (REASONING) 后端实例 (向后兼容门面)。"""
    return create_model(ModelType.REASONING, name, **kwargs)


# ---------------------------------------------------------------------------
# 注册内置后端
# ---------------------------------------------------------------------------

# 阿里云百炼 (DashScope) / 通义千问 (Qwen)
register_llm_backend("dashscope", DashScopeLLMBackend)
register_llm_backend("bailian", DashScopeLLMBackend)
register_llm_backend("qwen", DashScopeLLMBackend)
register_llm_backend("qianwen", DashScopeLLMBackend)
# 百炼: 向量 / 图像 / 视频 / 语音合成
register_llm_backend("dashscope", DashScopeEmbeddingBackend, ModelType.EMBEDDING)
register_llm_backend("dashscope", DashScopeImageBackend, ModelType.IMAGE)
register_llm_backend("dashscope", DashScopeVideoBackend, ModelType.VIDEO)
register_llm_backend("dashscope", DashScopeTTSBackend, ModelType.TTS)

# OrcaRouter (OpenAI 兼容路由服务)
register_llm_backend("orcarouter", OrcaRouterLLMBackend)
# OrcaRouter 同样暴露 OpenAI 兼容的向量接口
register_llm_backend("orcarouter", OrcaRouterEmbeddingBackend, ModelType.EMBEDDING)

# 智谱 AI (BigModel / GLM), OpenAI 兼容
register_llm_backend("zhipu", ZhipuLLMBackend)
register_llm_backend("zhipu", ZhipuEmbeddingBackend, ModelType.EMBEDDING)

# DeepSeek, OpenAI 兼容
register_llm_backend("deepseek", DeepSeekLLMBackend)
register_llm_backend("deepseek", DeepSeekEmbeddingBackend, ModelType.EMBEDDING)

# Anthropic 原生 Messages API
register_llm_backend("anthropic", AnthropicLLMBackend)
# Anthropic OpenAI 兼容接口
register_llm_backend("anthropic-openai", AnthropicOpenAILLMBackend)
register_llm_backend("anthropic-compatible", AnthropicOpenAILLMBackend)

# OpenAI 兼容基类 (通用兜底, 供自定义厂商继承)
register_llm_backend("openai-compat", OpenAICompatLLMBackend)
register_llm_backend("openai-compat", OpenAICompatEmbeddingBackend, ModelType.EMBEDDING)
register_llm_backend("openai-compat", OpenAICompatImageBackend, ModelType.IMAGE)
register_llm_backend("openai-compat", OpenAICompatVideoBackend, ModelType.VIDEO)
register_llm_backend("openai-compat", OpenAICompatTTSBackend, ModelType.TTS)
register_llm_backend("openai-compat", OpenAICompatRerankBackend, ModelType.RERANK)

# OrcaRouter (OpenAI 兼容路由网关): 推理 + 向量 + 图像 + 视频 + 语音 + 重排
register_llm_backend("orcarouter", OrcaRouterImageBackend, ModelType.IMAGE)
register_llm_backend("orcarouter", OrcaRouterVideoBackend, ModelType.VIDEO)
register_llm_backend("orcarouter", OrcaRouterTTSBackend, ModelType.TTS)
register_llm_backend("orcarouter", OrcaRouterRerankBackend, ModelType.RERANK)

# 智谱 (OpenAI 兼容推理/图像/语音/重排 + 原生 CogVideo 视频)
register_llm_backend("zhipu", ZhipuImageBackend, ModelType.IMAGE)
register_llm_backend("zhipu", ZhipuTTSBackend, ModelType.TTS)
register_llm_backend("zhipu", ZhipuRerankBackend, ModelType.RERANK)
register_llm_backend("zhipu", ZhipuVideoBackend, ModelType.VIDEO)

# DeepSeek: 公开 API 仅有 推理 / 向量 / 重排 (deepseek-rerank), 无图像/视频/语音
register_llm_backend("deepseek", DeepSeekRerankBackend, ModelType.RERANK)

# 百炼: 语义重排
register_llm_backend("dashscope", DashScopeRerankBackend, ModelType.RERANK)
register_llm_backend("bailian", DashScopeRerankBackend, ModelType.RERANK)
register_llm_backend("qwen", DashScopeRerankBackend, ModelType.RERANK)
