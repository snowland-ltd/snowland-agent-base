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

"""可插拔记忆层单元测试: 注册表 / 后端 / 门面 / Agent 注入。"""
import unittest

from langchain_core.embeddings import Embeddings

from snowland_agent_base.agent import BaseAgent
from snowland_agent_base.memory import (
    ChromaMemoryBackend,
    MemoryBackend,
    VectorMemory,
    create_memory_backend,
    list_memory_backends,
    register_memory_backend,
)


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0, 0.0]


class DictMemoryBackend(MemoryBackend):
    """不依赖向量库的纯内存后端, 用于测试插拔机制。"""

    def __init__(self, **kwargs):
        self._docs = []

    def add(self, texts, metadatas=None):
        self._docs.extend(texts)

    def search(self, query, top_k=4):
        return self._docs[:top_k]

    def as_retriever(self, top_k=4):
        raise NotImplementedError("DictMemoryBackend 不提供 retriever")


class TestRegistry(unittest.TestCase):
    def test_builtin_backends_registered(self):
        self.assertIn("chroma", list_memory_backends())

    def test_unknown_backend_raises(self):
        with self.assertRaises(KeyError):
            create_memory_backend("nope", embedding=FakeEmbeddings())

    def test_custom_backend_registration(self):
        register_memory_backend("dict", DictMemoryBackend)
        self.assertIn("dict", list_memory_backends())
        backend = create_memory_backend("dict")
        self.assertIsInstance(backend, DictMemoryBackend)


class TestFacade(unittest.TestCase):
    def test_default_is_chroma(self):
        mem = VectorMemory(embedding=FakeEmbeddings())
        self.assertEqual(mem.backend_name, "chroma")
        self.assertIsInstance(mem.backend, ChromaMemoryBackend)

    def test_accepts_backend_instance(self):
        mem = VectorMemory(backend=DictMemoryBackend())
        self.assertEqual(mem.backend_name, "DictMemoryBackend")
        mem.add(["a", "b"])
        self.assertEqual(mem.search("x", top_k=4), ["a", "b"])

    def test_accepts_backend_name(self):
        register_memory_backend("dict", DictMemoryBackend)
        mem = VectorMemory(backend="dict")
        mem.add(["hello"])
        self.assertEqual(mem.search("q"), ["hello"])

    def test_clear_noop_when_none(self):
        VectorMemory(backend=None).clear()  # 不应抛异常


class TestAgentAcceptsBackend(unittest.TestCase):
    def test_agent_wraps_memory_backend_instance(self):
        agent = BaseAgent(
            llm=_FakeLLM(),
            embedding=FakeEmbeddings(),
            memory=DictMemoryBackend(),
        )
        self.assertIsInstance(agent.memory, VectorMemory)
        self.assertIsInstance(agent.memory.backend, DictMemoryBackend)

    def test_agent_keeps_vector_memory(self):
        mem = VectorMemory(backend=DictMemoryBackend())
        agent = BaseAgent(llm=_FakeLLM(), embedding=FakeEmbeddings(), memory=mem)
        self.assertIs(agent.memory, mem)


class _FakeLLM:
    def invoke(self, *args, **kwargs):
        return "ok"


if __name__ == "__main__":
    unittest.main()
