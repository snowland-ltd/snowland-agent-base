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

"""可插拔 LLM 接入层单元测试: 注册表 / 后端 / 门面。"""
import os
import unittest

from langchain_core.language_models import BaseChatModel

from snowland_agent_base.llm import (
    DashScopeLLMBackend,
    LLMBackend,
    build_llm,
    create_llm_backend,
    list_llm_backends,
    register_llm_backend,
)


class FakeChatModel(BaseChatModel):
    """不联网的假 ChatModel。"""

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(self, *args, **kwargs):
        raise NotImplementedError

    def invoke(self, *args, **kwargs):
        return "ok"


class FakeLLMBackend(LLMBackend):
    def __init__(self, model="fake-model", **kwargs):
        self._model = model

    def build(self) -> BaseChatModel:
        return FakeChatModel()


class TestRegistry(unittest.TestCase):
    def test_builtin_backends_registered(self):
        names = list_llm_backends()
        self.assertIn("dashscope", names)
        self.assertIn("bailian", names)  # 别名

    def test_create_dashscope(self):
        backend = create_llm_backend("dashscope", model="qwen-max")
        self.assertIsInstance(backend, DashScopeLLMBackend)
        self.assertEqual(backend._params["model"], "qwen-max")

    def test_unknown_backend_raises(self):
        with self.assertRaises(KeyError):
            create_llm_backend("nope")

    def test_custom_backend_registration(self):
        register_llm_backend("fake", FakeLLMBackend)
        self.assertIn("fake", list_llm_backends())
        backend = create_llm_backend("fake", model="xyz")
        self.assertIsInstance(backend, FakeLLMBackend)
        self.assertIsInstance(backend.build(), BaseChatModel)


class TestDashScopeBackend(unittest.TestCase):
    def setUp(self):
        # 保证测试不依赖真实 key 环境变量
        self._old = os.environ.pop("DASHSCOPE_API_KEY", None)

    def tearDown(self):
        if self._old is not None:
            os.environ["DASHSCOPE_API_KEY"] = self._old

    def test_missing_api_key_raises(self):
        backend = DashScopeLLMBackend(model="qwen-plus", api_key=None)
        with self.assertRaises(ValueError):
            backend.build()

    def test_build_injects_api_key_from_param(self):
        backend = DashScopeLLMBackend(api_key="test-key", model="qwen-plus")
        self.assertEqual(backend._params["api_key"], "test-key")
        self.assertEqual(backend._params["model"], "qwen-plus")


class TestBuildLLMFacade(unittest.TestCase):
    def test_facade_returns_custom_backend_instance(self):
        register_llm_backend("fake2", FakeLLMBackend)
        model = build_llm(backend="fake2")
        self.assertIsInstance(model, BaseChatModel)

    def test_default_backend_name(self):
        # 不传 backend 时, 门面默认使用 dashscope 后端
        backend = create_llm_backend("dashscope", api_key="x")
        self.assertIsInstance(backend, DashScopeLLMBackend)


if __name__ == "__main__":
    unittest.main()
