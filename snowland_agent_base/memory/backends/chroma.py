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

"""ChromaDB 记忆后端实现。

基于 ChromaDB 的本地向量知识库 (默认后端)。
"""
from __future__ import annotations

import os
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from snowland_agent_base.memory.base import MemoryBackend


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
