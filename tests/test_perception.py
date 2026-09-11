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

"""可插拔感知层单元测试: 注册表 / 后端 / 门面 / Agent 注入。"""
import unittest

from langchain_core.messages import AIMessage

from snowland_agent_base.agent import BaseAgent
from snowland_agent_base.perception import (
    BailianPerceptionBackend,
    PerceptionBackend,
    PerceptionLayer,
    SpacyPerceptionBackend,
    create_perception_backend,
    list_perception_backends,
    register_perception_backend,
)


class FakeChatModel:
    """不联网的假 ChatModel, 按 prompt 内容返回结构化结果。"""

    def invoke(self, messages, *args, **kwargs):
        text = messages[-1].content if messages else ""
        if "entities" in text:  # 命中 perceive 模板
            return AIMessage(
                content='{"entities":[{"text":"北京","label":"GPE"}],"intent":"search"}'
            )
        return AIMessage(content="search")


class TestRegistry(unittest.TestCase):
    def test_builtin_backends_registered(self):
        names = list_perception_backends()
        self.assertIn("spacy", names)
        self.assertIn("bailian", names)

    def test_create_spacy_with_keywords(self):
        backend = create_perception_backend("spacy", intent_keywords={"order": ["下单"]})
        self.assertIsInstance(backend, SpacyPerceptionBackend)
        self.assertEqual(backend.classify_intent("我要下单"), "order")

    def test_unknown_backend_raises(self):
        with self.assertRaises(KeyError):
            create_perception_backend("nope")

    def test_custom_backend_registration(self):
        class MyBackend(PerceptionBackend):
            def extract_entities(self, text):
                return []

            def classify_intent(self, text, default="unknown"):
                return "my-intent"

            def perceive(self, text):
                return {"text": text, "entities": [], "intent": "my-intent"}

        register_perception_backend("my", MyBackend)
        self.assertIn("my", list_perception_backends())
        layer = PerceptionLayer(backend="my")
        self.assertEqual(layer.classify_intent("x"), "my-intent")
        self.assertEqual(layer.perceive("x")["intent"], "my-intent")


class TestSpacyDefault(unittest.TestCase):
    def setUp(self):
        self.layer = PerceptionLayer()  # 默认 spaCy 后端

    def test_default_backend_is_spacy(self):
        self.assertIsInstance(self.layer.backend, SpacyPerceptionBackend)

    def test_intent_no_spacy_model_needed(self):
        # 意图分类是规则实现, 不触发 spaCy 模型加载
        self.assertEqual(self.layer.classify_intent("你好呀"), "greeting")
        self.assertEqual(self.layer.classify_intent("这是为什么呢"), "question")
        self.assertEqual(self.layer.classify_intent("啊这", default="other"), "other")

    def test_backend_instance_directly(self):
        layer = PerceptionLayer(backend=SpacyPerceptionBackend(intent_keywords={"x": ["y"]}))
        self.assertEqual(layer.backend_name, "SpacyPerceptionBackend")


class TestBailianBackend(unittest.TestCase):
    def setUp(self):
        self.backend = BailianPerceptionBackend(llm=FakeChatModel())

    def test_perceive(self):
        res = self.backend.perceive("查询北京的天气")
        self.assertEqual(res["intent"], "search")
        self.assertEqual(res["entities"], [{"text": "北京", "label": "GPE"}])

    def test_classify_intent(self):
        self.assertEqual(self.backend.classify_intent("搜索订单"), "search")

    def test_extract_entities(self):
        self.assertEqual(
            self.backend.extract_entities("北京"), [{"text": "北京", "label": "GPE"}]
        )

    def test_facade_by_name(self):
        layer = PerceptionLayer(backend="bailian", llm=FakeChatModel())
        self.assertEqual(layer.backend_name, "bailian")
        self.assertEqual(layer.perceive("查询北京")["intent"], "search")


class TestAgentAcceptsBackend(unittest.TestCase):
    def test_agent_wraps_backend_instance(self):
        fake = FakeChatModel()
        agent = BaseAgent(llm=fake, perception=BailianPerceptionBackend(llm=fake))
        self.assertIsInstance(agent.perception, PerceptionLayer)
        self.assertIsInstance(agent.perception.backend, BailianPerceptionBackend)

    def test_agent_keeps_perception_layer(self):
        layer = PerceptionLayer(backend="spacy")
        agent = BaseAgent(llm=FakeChatModel(), perception=layer)
        self.assertIs(agent.perception, layer)


if __name__ == "__main__":
    unittest.main()
