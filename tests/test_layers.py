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

"""感知层 / 决策层 / 执行层 单元测试。"""
import json
import time
import unittest

from snowland_agent_base.decision import (
    DEFAULT_SYSTEM_PROMPT,
    AgentAction,
    DecisionLayer,
    Rule,
)
from snowland_agent_base.execution import ToolExecutor
from snowland_agent_base.perception import PerceptionLayer

from tests.test_agent import FakeChatModel


class TestPerceptionLayer(unittest.TestCase):
    """意图分类不依赖 spaCy 实体模型, 可独立验证。"""

    def setUp(self) -> None:
        self.perception = PerceptionLayer()

    def test_classify_greeting(self):
        self.assertEqual(self.perception.classify_intent("你好呀"), "greeting")

    def test_classify_question(self):
        self.assertEqual(self.perception.classify_intent("这是为什么呢"), "question")

    def test_classify_unknown_default(self):
        self.assertEqual(self.perception.classify_intent("啊这"), "unknown")
        self.assertEqual(self.perception.classify_intent("啊这", default="other"), "other")

    def test_custom_keywords_override(self):
        custom = PerceptionLayer(intent_keywords={"order": ["下单", "购买"]})
        self.assertEqual(custom.classify_intent("我要下单"), "order")

    def test_scoring_picks_highest(self):
        """命中关键词更多的意图应胜出 ("搜索"+"查询" -> search 得 2 分)。"""
        self.assertEqual(self.perception.classify_intent("搜索查询订单"), "search")


class TestDecisionLayer(unittest.TestCase):
    def setUp(self) -> None:
        self.llm = FakeChatModel()

    def test_parse_plain_json(self):
        action = DecisionLayer._parse_action('{"thought": "t", "final_answer": "a"}')
        self.assertEqual(action.final_answer, "a")
        self.assertTrue(action.is_final)

    def test_parse_fenced_json(self):
        raw = '```json\n{"thought": "t", "tool": "search", "tool_input": {"q": "x"}}\n```'
        action = DecisionLayer._parse_action(raw)
        self.assertEqual(action.tool, "search")
        self.assertEqual(action.tool_input, {"q": "x"})
        self.assertFalse(action.is_final)

    def test_prompt_keeps_json_example(self):
        """回归: 模板含 JSON 示例, 早期 str.format 会因花括号抛 KeyError。"""
        layer = DecisionLayer(llm=self.llm)
        layer.decide("任意输入")
        content = self.llm.last_messages[0].content
        self.assertIn('"final_answer"', content)
        self.assertIn('"tool_input"', content)

    def test_default_persona_present(self):
        layer = DecisionLayer(llm=self.llm)
        layer.decide("x")
        self.assertIn(DEFAULT_SYSTEM_PROMPT, self.llm.last_messages[0].content)

    def test_custom_system_prompt(self):
        layer = DecisionLayer(llm=self.llm, system_prompt="你是电商客服助手")
        layer.decide("x")
        content = self.llm.last_messages[0].content
        self.assertIn("你是电商客服助手", content)
        self.assertNotIn(DEFAULT_SYSTEM_PROMPT, content)

    def test_per_request_system_prompt_overrides(self):
        layer = DecisionLayer(llm=self.llm, system_prompt="实例级人设")
        layer.decide("x", system_prompt="请求级人设")
        content = self.llm.last_messages[0].content
        self.assertIn("请求级人设", content)
        self.assertNotIn("实例级人设", content)

    def test_build_prompt_sections(self):
        """系统提示词含人设、CoT 指令与工具列表。"""
        prompt = DecisionLayer(llm=self.llm).build_prompt(tools=["search", "calc"])
        self.assertIn("- search", prompt)
        self.assertIn("- calc", prompt)
        self.assertIn(DEFAULT_SYSTEM_PROMPT, prompt)

    def test_build_prompt_no_tools(self):
        prompt = DecisionLayer(llm=self.llm).build_prompt(tools=[])
        self.assertIn("(无可用工具)", prompt)

    def test_rule_match_returns_action(self):
        rule = Rule(name="r", predicate=lambda t, c: "ping" in t, action=AgentAction(final_answer="pong"))
        layer = DecisionLayer(llm=self.llm, rules=[rule])
        action = layer.decide("ping")
        self.assertEqual(action.final_answer, "pong")

    def test_rule_exception_is_skipped(self):
        def bad(t, c):
            raise RuntimeError("规则炸了")

        good = Rule(name="g", predicate=lambda t, c: True, action=AgentAction(final_answer="ok"))
        layer = DecisionLayer(llm=self.llm, rules=[Rule(name="b", predicate=bad, action=AgentAction()), good])
        self.assertEqual(layer.decide("任意").final_answer, "ok")

    def test_add_rule_appends(self):
        layer = DecisionLayer(llm=self.llm)
        layer.add_rule(Rule(name="a", predicate=lambda t, c: False, action=AgentAction()))
        self.assertEqual(len(layer._rules), 1)

    def test_action_to_dict_keys(self):
        data = AgentAction(thought="t", tool="x", tool_input={"a": 1}).to_dict()
        self.assertEqual(set(data), {"thought", "tool", "tool_input", "final_answer"})
        self.assertIsNone(data["final_answer"])


class TestToolExecutor(unittest.TestCase):
    def setUp(self) -> None:
        self.executor = ToolExecutor(max_retries=2, retry_interval=0.01, timeout=5)

    def test_register_and_execute(self):
        self.executor.register("upper", lambda s: s.upper())
        result = self.executor.execute("upper", s="ab")
        self.assertTrue(result["success"])
        self.assertEqual(result["result"], "AB")

    def test_unregister(self):
        self.executor.register("x", lambda: 1)
        self.executor.unregister("x")
        self.assertNotIn("x", self.executor.list_tools())

    def test_unknown_tool(self):
        result = self.executor.execute("nope")
        self.assertFalse(result["success"])
        self.assertIn("未注册", result["error"])

    def test_retry_then_fail(self):
        state = {"count": 0}

        def flaky():
            state["count"] += 1
            raise RuntimeError("总是失败")

        self.executor.register("flaky", flaky)
        result = self.executor.execute("flaky")
        self.assertFalse(result["success"])
        self.assertEqual(state["count"], 2)

    def test_retry_then_succeed(self):
        state = {"count": 0}

        def flaky():
            state["count"] += 1
            if state["count"] < 2:
                raise RuntimeError("第一次失败")
            return "好了"

        self.executor.register("flaky", flaky)
        result = self.executor.execute("flaky")
        self.assertTrue(result["success"])
        self.assertEqual(result["result"], "好了")

    def test_timeout(self):
        """工具耗时超过 timeout 应判定超时。"""
        self.executor.timeout = 0.5  # sleep(2) 远超该值
        self.executor.register("slow", lambda: time.sleep(2))
        result = self.executor.execute("slow")
        self.assertFalse(result["success"])
        self.assertIn("超时", result["error"])

    def test_register_non_callable_rejected(self):
        with self.assertRaises(TypeError):
            self.executor.register("bad", 123)


class TestJsonContract(unittest.TestCase):
    """决策层 prompt 约定的 JSON 契约自检。"""

    def test_expected_keys_present(self):
        payload = json.loads('{"thought": "t", "tool": null, "tool_input": {}, "final_answer": "a"}')
        for key in ("thought", "tool", "tool_input", "final_answer"):
            self.assertIn(key, payload)


if __name__ == "__main__":
    unittest.main()
