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

"""感知后端抽象接口。

把「感知」抽象为统一后端接口 :class:`PerceptionBackend`, 不同实现可热插拔。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List


class PerceptionBackend(ABC):
    """感知后端统一接口。

    所有具体后端 (spaCy / 百炼 / 自定义) 都必须实现这三个方法,
    ``perceive`` 的返回结构约定为 ``{"text", "entities", "intent"}``。
    """

    @abstractmethod
    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        """实体识别, 返回 ``[{"text": ..., "label": ...}, ...]``。"""

    @abstractmethod
    def classify_intent(self, text: str, default: str = "unknown") -> str:
        """意图分类, 返回意图类别字符串。"""

    @abstractmethod
    def perceive(self, text: str) -> Dict[str, object]:
        """统一入口: 一次性产出实体与意图。"""
