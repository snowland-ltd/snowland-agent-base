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

"""向后兼容门面: PerceptionLayer。"""
from __future__ import annotations

from typing import Dict, List, Optional

from snowland_agent_base.perception.base import PerceptionBackend
from snowland_agent_base.perception.registry import create_perception_backend


class PerceptionLayer:
    """可插拔感知层门面。

    对外暴露与原 :class:`PerceptionLayer` 完全一致的 ``perceive`` /
    ``extract_entities`` / ``classify_intent`` 接口, 内部委派给具体后端。

    ``backend`` 支持两种形式:
        - 字符串名称 (经注册表解析, 可插拔): ``"spacy"`` / ``"bailian"``。
        - 后端实例 (直接注入, 便于测试 / 自定义): ``SpacyPerceptionBackend()``。
    缺省 (不传) 使用 spaCy 后端, 与原行为兼容。
    """

    def __init__(self, backend: Optional[str] = "spacy", **kwargs) -> None:
        if isinstance(backend, PerceptionBackend):
            self._backend = backend
            self._backend_name = type(backend).__name__
        elif isinstance(backend, str):
            self._backend = create_perception_backend(backend, **kwargs)
            self._backend_name = backend
        elif backend is None:
            self._backend = None  # type: ignore[assignment]
            self._backend_name = "none"
        else:
            raise TypeError("backend 需为 PerceptionBackend 实例或已注册的名称字符串")

    @property
    def backend(self) -> Optional[PerceptionBackend]:
        """当前实际使用的感知后端实例。"""
        return self._backend

    @property
    def backend_name(self) -> str:
        """当前后端的名称 (字符串名或类名)。"""
        return self._backend_name

    def perceive(self, text: str) -> Dict[str, object]:
        if self._backend is None:
            return {"text": text, "entities": [], "intent": "unknown"}
        return self._backend.perceive(text)

    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        if self._backend is None:
            return []
        return self._backend.extract_entities(text)

    def classify_intent(self, text: str, default: str = "unknown") -> str:
        if self._backend is None:
            return default
        return self._backend.classify_intent(text, default)
