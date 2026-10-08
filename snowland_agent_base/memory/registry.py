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

"""记忆后端注册表与工厂 (插拔核心)。"""
from __future__ import annotations

from typing import Dict

from snowland_agent_base.memory.backends.chroma import ChromaMemoryBackend
from snowland_agent_base.memory.base import MemoryBackend


_MEMORY_BACKENDS: Dict[str, type] = {}


def register_memory_backend(name: str, cls: type) -> None:
    """注册一个记忆后端, 使其可通过名称字符串被工厂解析。"""
    if not (isinstance(cls, type) and issubclass(cls, MemoryBackend)):
        raise TypeError(f"记忆后端 {cls!r} 必须继承 MemoryBackend")
    _MEMORY_BACKENDS[name] = cls


def get_memory_backend(name: str) -> type:
    """按名称取回已注册的记忆后端类。"""
    if name not in _MEMORY_BACKENDS:
        raise KeyError(f"未知记忆后端: {name!r}; 可用: {list_memory_backends()}")
    return _MEMORY_BACKENDS[name]


def list_memory_backends() -> list:
    """列出所有已注册的记忆后端名称。"""
    return list(_MEMORY_BACKENDS)


def create_memory_backend(name: str = "chroma", **kwargs) -> MemoryBackend:
    """按名称构建一个记忆后端实例。"""
    return get_memory_backend(name)(**kwargs)


# 注册内置后端
register_memory_backend("chroma", ChromaMemoryBackend)
