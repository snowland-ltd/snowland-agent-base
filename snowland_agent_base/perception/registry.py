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

"""感知后端注册表与工厂 (插拔核心)。"""
from __future__ import annotations

from typing import Dict, List

from snowland_agent_base.perception.backends.bailian import BailianPerceptionBackend
from snowland_agent_base.perception.backends.spacy import SpacyPerceptionBackend
from snowland_agent_base.perception.base import PerceptionBackend


_PERCEPTION_BACKENDS: Dict[str, type] = {}


def register_perception_backend(name: str, cls: type) -> None:
    """注册一个感知后端, 使其可通过名称字符串被工厂解析。

    Args:
        name: 后端名称, 如 ``"spacy"`` / ``"bailian"``。
        cls: 继承自 :class:`PerceptionBackend` 的后端类。
    """
    if not (isinstance(cls, type) and issubclass(cls, PerceptionBackend)):
        raise TypeError(f"感知后端 {cls!r} 必须继承 PerceptionBackend")
    _PERCEPTION_BACKENDS[name] = cls


def get_perception_backend(name: str) -> type:
    """按名称取回已注册的感知后端类。"""
    if name not in _PERCEPTION_BACKENDS:
        raise KeyError(
            f"未知感知后端: {name!r}; 可用: {list_perception_backends()}"
        )
    return _PERCEPTION_BACKENDS[name]


def list_perception_backends() -> List[str]:
    """列出所有已注册的感知后端名称。"""
    return list(_PERCEPTION_BACKENDS)


def create_perception_backend(name: str = "spacy", **kwargs) -> PerceptionBackend:
    """按名称构建一个感知后端实例。"""
    return get_perception_backend(name)(**kwargs)


# 注册内置后端
register_perception_backend("spacy", SpacyPerceptionBackend)
register_perception_backend("bailian", BailianPerceptionBackend)
