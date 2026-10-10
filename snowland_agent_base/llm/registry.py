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

"""LLM 后端注册表与工厂 (插拔核心)。"""
from __future__ import annotations

from typing import Dict

from snowland_agent_base.llm.backends.dashscope import DashScopeLLMBackend
from snowland_agent_base.llm.backends.deepseek import DeepSeekLLMBackend
from snowland_agent_base.llm.backends.orcarouter import OrcaRouterLLMBackend
from snowland_agent_base.llm.backends.zhipu import ZhipuLLMBackend
from snowland_agent_base.llm.backends.anthropic import AnthropicLLMBackend
from snowland_agent_base.llm.backends.anthropic_openai import (
    AnthropicOpenAILLMBackend,
)
from snowland_agent_base.llm.base import LLMBackend
from snowland_agent_base.messages import Message, OpenAIMessage


_LLM_BACKENDS: Dict[str, type] = {}


def register_llm_backend(name: str, cls: type) -> None:
    """注册一个大模型后端, 使其可通过名称字符串被工厂解析。"""
    if not (isinstance(cls, type) and issubclass(cls, LLMBackend)):
        raise TypeError(f"大模型后端 {cls!r} 必须继承 LLMBackend")
    _LLM_BACKENDS[name] = cls


def get_llm_backend(name: str) -> type:
    """按名称取回已注册的大模型后端类。"""
    if name not in _LLM_BACKENDS:
        raise KeyError(f"未知大模型后端: {name!r}; 可用: {list_llm_backends()}")
    return _LLM_BACKENDS[name]


def list_llm_backends() -> list:
    """列出所有已注册的大模型后端名称。"""
    return list(_LLM_BACKENDS)


def get_message_class(backend: Optional[str]) -> type[Message]:
    """按后端名取回其配套的消息风格类 (``OpenAIMessage`` / ``AnthropicMessage``)。

    每个 :class:`LLMBackend` 子类通过 ``message_class`` 声明自己使用的消息格式,
    调用方据此构造与后端协议匹配的消息对象, 无需硬编码 ``backend`` 字符串分支。

    Args:
        backend: 后端名称或已注册别名; ``None`` / 未知名称均回退为 ``OpenAIMessage``。

    Returns:
        消息风格类。
    """
    if not backend:
        return OpenAIMessage
    try:
        return get_llm_backend(backend).message_class
    except KeyError:
        return OpenAIMessage


def create_llm_backend(name: str = "dashscope", **kwargs) -> LLMBackend:
    """按名称构建一个大模型后端实例。"""
    return get_llm_backend(name)(**kwargs)


# 注册内置后端
# 阿里云百炼 (DashScope) / 通义千问 (Qwen)
register_llm_backend("dashscope", DashScopeLLMBackend)
register_llm_backend("bailian", DashScopeLLMBackend)
register_llm_backend("qwen", DashScopeLLMBackend)
register_llm_backend("qianwen", DashScopeLLMBackend)
# OrcaRouter (OpenAI 兼容路由服务)
register_llm_backend("orcarouter", OrcaRouterLLMBackend)
# 智谱 AI (BigModel / GLM), OpenAI 兼容
register_llm_backend("zhipu", ZhipuLLMBackend)
# DeepSeek, OpenAI 兼容
register_llm_backend("deepseek", DeepSeekLLMBackend)
# Anthropic 原生 Messages API
register_llm_backend("anthropic", AnthropicLLMBackend)
# Anthropic OpenAI 兼容接口
register_llm_backend("anthropic-openai", AnthropicOpenAILLMBackend)
register_llm_backend("anthropic-compatible", AnthropicOpenAILLMBackend)
