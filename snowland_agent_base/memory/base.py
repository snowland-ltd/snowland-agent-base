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

"""记忆后端抽象接口。

把「长期记忆向量存储」抽象为统一后端接口 :class:`MemoryBackend`, 不同向量库可热插拔。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional


class MemoryBackend(ABC):
    """记忆后端统一接口: 长期记忆的写入与语义检索。"""

    @abstractmethod
    def add(self, texts: List[str], metadatas: Optional[List[dict]] = None) -> None:
        """写入记忆片段。"""

    @abstractmethod
    def search(self, query: str, top_k: int = 4) -> List[str]:
        """语义检索最相关的历史记忆文本。"""

    @abstractmethod
    def as_retriever(self, top_k: int = 4):
        """返回 LangChain Retriever, 供检索链使用。"""

    def clear(self) -> None:  # pragma: no cover - 默认无操作
        """清空记忆 (可选实现)。"""
