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

from snowland_agent_base.memory.backends import ChromaMemoryBackend
from snowland_agent_base.memory.base import MemoryBackend
from snowland_agent_base.memory.facade import VectorMemory
from snowland_agent_base.memory.registry import (
    create_memory_backend,
    get_memory_backend,
    list_memory_backends,
    register_memory_backend,
)


__all__ = [
    "MemoryBackend",
    "ChromaMemoryBackend",
    "VectorMemory",
    "register_memory_backend",
    "get_memory_backend",
    "list_memory_backends",
    "create_memory_backend",
]
