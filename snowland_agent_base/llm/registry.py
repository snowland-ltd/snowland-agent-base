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
from snowland_agent_base.llm.base import LLMBackend


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
