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

"""可插拔大模型接入层。

把「构建 ChatModel」抽象为统一后端接口 :class:`LLMBackend`, 不同厂商可热插拔:

- :class:`DashScopeLLMBackend`: 基于阿里云百炼 (DashScope) 服务 (默认)。

:func:`build_llm` 是向后兼容的门面, 通过 ``backend`` 选择后端::

    build_llm()                                   # 默认百炼
    build_llm(backend="dashscope", model="qwen-max")
    build_llm(backend=MyLLMBackend(...))          # 直接注入实例

新增后端只需继承 :class:`LLMBackend` 并 ``register_llm_backend`` 注册,
即可被名称字符串解析, 实现真正的插拔式扩展。
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Dict, Optional

from langchain_core.language_models import BaseChatModel


# --------------------------------------------------------------------------- #
# 抽象后端
# --------------------------------------------------------------------------- #
class LLMBackend(ABC):
    """大模型后端统一接口: 产出 LangChain ``BaseChatModel``。"""

    @abstractmethod
    def build(self) -> BaseChatModel:
        """构建并返回一个 ``BaseChatModel`` 实例。"""


# --------------------------------------------------------------------------- #
# 后端: 阿里云百炼 (DashScope)
# --------------------------------------------------------------------------- #
class DashScopeLLMBackend(LLMBackend):
    """基于阿里云百炼 (DashScope) 的 ChatModel 后端。

    官方文档: https://help.aliyun.com/zh/model-studio/
    """

    def __init__(
        self,
        model: str = "qwen-plus",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 0.8,
        enable_thinking: bool = False,
        **kwargs,
    ) -> None:
        """初始化百炼后端。

        Args:
            model: 百炼平台模型名, 如 ``qwen-plus`` / ``qwen-max`` / ``qwen-turbo``。
            api_key: DashScope API Key; 缺省时从环境变量 ``DASHSCOPE_API_KEY`` 读取。
            temperature: 采样温度 (0~1)。
            max_tokens: 单次回复最大 token 数。
            top_p: 核采样概率。
            enable_thinking: 是否开启 qwq 系列模型的思考模式。
            kwargs: 透传给 LangChain DashScope 构造器的其他参数。
        """
        self._params = {
            "model": model,
            "api_key": api_key or os.getenv("DASHSCOPE_API_KEY"),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "enable_thinking": enable_thinking,
            **kwargs,
        }

    def build(self) -> BaseChatModel:
        if not self._params.get("api_key"):
            raise ValueError(
                "DashScope API Key 未提供: 请传入 api_key 或设置环境变量 DASHSCOPE_API_KEY"
            )
        from langchain_dashscope import ChatDashScope

        return ChatDashScope(**self._params)


# --------------------------------------------------------------------------- #
# 注册表 + 工厂 (插拔核心)
# --------------------------------------------------------------------------- #
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


def create_llm_backend(name: str = "dashscope", **kwargs) -> LLMBackend:
    """按名称构建一个大模型后端实例。"""
    return get_llm_backend(name)(**kwargs)


# 注册内置后端 ("bailian" 作为 "dashscope" 的别名)
register_llm_backend("dashscope", DashScopeLLMBackend)
register_llm_backend("bailian", DashScopeLLMBackend)


# --------------------------------------------------------------------------- #
# 门面 (向后兼容)
# --------------------------------------------------------------------------- #
def build_llm(
    backend: str = "dashscope",
    model: str = "qwen-plus",
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    top_p: float = 0.8,
    enable_thinking: bool = False,
    **kwargs,
) -> BaseChatModel:
    """构建一个 ChatModel 实例 (门面)。

    Args:
        backend: 后端名称或已注册别名, 默认 ``"dashscope"`` (亦可用 ``"bailian"``)。
        model: 百炼平台模型名, 如 ``qwen-plus`` / ``qwen-max`` / ``qwen-turbo``。
        api_key: DashScope API Key; 缺省时从环境变量 ``DASHSCOPE_API_KEY`` 读取。
        temperature: 采样温度 (0~1)。
        max_tokens: 单次回复最大 token 数。
        top_p: 核采样概率。
        enable_thinking: 是否开启 qwq 系列模型的思考模式。
        kwargs: 透传给具体后端构造器的其他参数。

    Returns:
        LangChain ``BaseChatModel`` 实例。
    """
    return create_llm_backend(
        backend,
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        enable_thinking=enable_thinking,
        **kwargs,
    ).build()


__all__ = [
    "LLMBackend",
    "DashScopeLLMBackend",
    "build_llm",
    "register_llm_backend",
    "get_llm_backend",
    "list_llm_backends",
    "create_llm_backend",
]
