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

"""可插拔记忆层。

把「长期记忆向量存储」抽象为统一后端接口 :class:`MemoryBackend`, 不同向量库可热插拔:

- :class:`ChromaMemoryBackend`: 基于 ChromaDB 的本地向量知识库 (默认)。

:class:`VectorMemory` 是向后兼容的门面, 通过 ``backend`` 选择具体后端::

    VectorMemory(embedding=emb)                       # 默认 ChromaDB
    VectorMemory(backend="chroma", embedding=emb)
    VectorMemory(backend=ChromaMemoryBackend(embedding=emb))

新增后端只需继承 :class:`MemoryBackend` 并 ``register_memory_backend`` 注册,
即可被名称字符串解析, 实现真正的插拔式扩展。
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


# --------------------------------------------------------------------------- #
# 抽象后端
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# 后端: ChromaDB
# --------------------------------------------------------------------------- #
class ChromaMemoryBackend(MemoryBackend):
    """基于 ChromaDB 的长期记忆向量存储。

    无需复杂运维, 本地持久化即可使用。短期记忆 (对话上下文) 与长期记忆
    (语义检索) 分离管理。
    """

    def __init__(
        self,
        embedding: Embeddings,
        collection_name: str = "agent_long_term_memory",
        persist_directory: Optional[str] = None,
    ) -> None:
        """初始化向量记忆。

        Args:
            embedding: LangChain Embeddings 实例 (如 DashScope / OpenAI embedding)。
            collection_name: Chroma 集合名。
            persist_directory: 持久化目录; 缺省使用 ``./chroma_store``。
        """
        self._embedding = embedding
        self._collection_name = collection_name
        self._persist_directory = persist_directory or os.path.join(
            os.getcwd(), "chroma_store"
        )
        self._vectorstore = None

    def _ensure_store(self):
        if self._vectorstore is not None:
            return self._vectorstore
        from langchain_chroma import Chroma

        self._vectorstore = Chroma(
            collection_name=self._collection_name,
            embedding_function=self._embedding,
            persist_directory=self._persist_directory,
        )
        return self._vectorstore

    def add(self, texts: List[str], metadatas: Optional[List[dict]] = None) -> None:
        """写入记忆片段。"""
        store = self._ensure_store()
        docs = []
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
            docs.append(Document(page_content=text, metadata=metadata))
        store.add_documents(docs)

    def search(self, query: str, top_k: int = 4) -> List[str]:
        """语义检索最相关的历史记忆文本。"""
        store = self._ensure_store()
        results = store.similarity_search(query, k=top_k)
        return [doc.page_content for doc in results]

    def as_retriever(self, top_k: int = 4):
        """返回 LangChain Retriever, 供检索链使用。"""
        store = self._ensure_store()
        return store.as_retriever(search_kwargs={"k": top_k})

    def clear(self) -> None:
        """清空当前集合 (谨慎使用)。"""
        store = self._ensure_store()
        store.delete_collection()
        self._vectorstore = None


# --------------------------------------------------------------------------- #
# 注册表 + 工厂 (插拔核心)
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# 门面 (向后兼容)
# --------------------------------------------------------------------------- #
class VectorMemory:
    """可插拔记忆层门面。

    对外暴露与原 :class:`VectorMemory` 完全一致的 ``add`` / ``search`` /
    ``as_retriever`` / ``clear`` 接口, 内部委派给具体后端。

    ``backend`` 支持两种形式:
        - 字符串名称 (经注册表解析, 可插拔): ``"chroma"``。
        - 后端实例 (直接注入, 便于测试 / 自定义): ``ChromaMemoryBackend(...)``。
    缺省 (不传) 使用 ChromaDB 后端, 与原行为兼容。
    """

    def __init__(self, backend: Optional[str] = "chroma", embedding: Optional[Embeddings] = None, **kwargs) -> None:
        if isinstance(backend, MemoryBackend):
            self._backend = backend
            self._backend_name = type(backend).__name__
        elif isinstance(backend, str):
            self._backend = create_memory_backend(backend, embedding=embedding, **kwargs)
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

    def add(self, texts: List[str], metadatas: Optional[List[dict]] = None) -> None:
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


__all__ = [
    "MemoryBackend",
    "ChromaMemoryBackend",
    "VectorMemory",
    "register_memory_backend",
    "get_memory_backend",
    "list_memory_backends",
    "create_memory_backend",
]
