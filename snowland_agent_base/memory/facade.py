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

"""向后兼容门面: VectorMemory。"""
from __future__ import annotations

from typing import List, Optional

from langchain_core.embeddings import Embeddings

from snowland_agent_base.memory.base import MemoryBackend
from snowland_agent_base.memory.registry import create_memory_backend


class VectorMemory:
    """可插拔记忆层门面。

    对外暴露与原 :class:`VectorMemory` 完全一致的 ``add`` / ``search`` /
    ``as_retriever`` / ``clear`` 接口, 内部委派给具体后端。

    ``backend`` 支持两种形式:
        - 字符串名称 (经注册表解析, 可插拔): ``"chroma"``。
        - 后端实例 (直接注入, 便于测试 / 自定义): ``ChromaMemoryBackend(...)``。
    缺省 (不传) 使用 ChromaDB 后端, 与原行为兼容。
    """

    def __init__(
        self,
        backend: Optional[str] = "chroma",
        embedding: Optional[Embeddings] = None,
        **kwargs,
    ) -> None:
        if isinstance(backend, MemoryBackend):
            self._backend = backend
            self._backend_name = type(backend).__name__
        elif isinstance(backend, str):
            self._backend = create_memory_backend(
                backend, embedding=embedding, **kwargs
            )
            self._backend_name = backend
        elif backend is None:
            self._backend = None  # type: ignore[assignment]
            self._backend_name = "none"
        else:
            raise TypeError("backend 需为 MemoryBackend 实例或已注册的名称字符串")

    @property
    def backend(self) -> Optional[MemoryBackend]:
        """当前实际使用的记忆后端实例。"""
        return self._backend

    @property
    def backend_name(self) -> str:
        """当前后端的名称 (字符串名或类名)。"""
        return self._backend_name

    def add(
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
    ) -> None:
        if self._backend is None:
            raise RuntimeError("memory backend 未配置")
        self._backend.add(texts, metadatas)

    def search(self, query: str, top_k: int = 4) -> List[str]:
        if self._backend is None:
            return []
        return self._backend.search(query, top_k)

    def as_retriever(self, top_k: int = 4):
        if self._backend is None:
            raise RuntimeError("memory backend 未配置")
        return self._backend.as_retriever(top_k)

    def clear(self) -> None:
        if self._backend is None:
            return
        self._backend.clear()
