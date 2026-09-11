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

"""BaseAgent 主循环测试: 用 FakeLLM / FakeEmbedding 替代外部依赖, 不联网。"""
import unittest
from typing import Any, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult, ChatGeneration

from snowland_agent_base.agent import BaseAgent
from snowland_agent_base.decision import DEFAULT_SYSTEM_PROMPT, AgentAction, Rule


class FakeChatModel(BaseChatModel):
    """按序返回预设回复的假 ChatModel, 并记录最近一次收到的 messages。"""

    responses: List[str] = []
    last_messages: List[BaseMessage] = []

    @property
    def _llm_type(self) -> str:
        return "fake-chat-model"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.last_messages = messages
        content = self.responses[0] if self.responses else '{"thought": "t", "final_answer": "ok"}'
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])


class FakeEmbeddings(Embeddings):
    """确定性假 Embedding, 便于断言检索行为。"""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[float(len(t)), 1.0, 0.0] for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return [float(len(text)), 1.0, 0.0]


class TestBaseAgent(unittest.TestCase):
    def setUp(self) -> None:
        self.llm = FakeChatModel()

    def test_final_answer_short_circuit(self):
        """LLM 直接给出 final_answer 时, 不应调用任何工具。"""
        self.llm.responses = ['{"thought": "直接回答", "final_answer": "你好呀"}']
        agent = BaseAgent(llm=self.llm, use_perception=False, use_memory=False)
        result = agent.run("你好")

        self.assertEqual(result["answer"], "你好呀")
        self.assertEqual(len(result["steps"]), 1)
        self.assertIsNone(result["steps"][0]["observation"])

    def test_tool_call_flow(self):
        """先调工具再收尾: 工具应被执行且观测结果进入 steps。"""
        self.llm.responses = [
            '{"thought": "需要计算", "tool": "add", "tool_input": {"a": 1, "b": 2}, "final_answer": null}'
        ]
        agent = BaseAgent(llm=self.llm, use_perception=False, use_memory=False)
        agent.register_tool("add", lambda a, b: a + b)

        result = agent.run("1 加 2 等于几")
        self.assertIn("add", agent.executor.list_tools())
        self.assertEqual(result["steps"][0]["observation"]["result"], 3)

    def test_rule_engine_takes_priority(self):
        """规则命中时优先于 LLM, 且 LLM 不应被调用。"""
        called = {"llm": False}

        original = self.llm._generate

        def spy(*args, **kwargs):
            called["llm"] = True
            return original(*args, **kwargs)

        self.llm._generate = spy  # type: ignore[assignment]
        rule = Rule(
            name="greet",
            predicate=lambda text, ctx: "你好" in text,
            action=AgentAction(thought="规则命中", final_answer="规则回复"),
        )
        agent = BaseAgent(llm=self.llm, use_perception=False, use_memory=False)
        agent.add_rule(rule)

        result = agent.run("你好")
        self.assertEqual(result["answer"], "规则回复")
        self.assertFalse(called["llm"])

    def test_fallback_on_invalid_json(self):
        """LLM 返回非法 JSON 时走兜底答复, 不抛异常。"""
        self.llm.responses = ["这不是 JSON"]
        agent = BaseAgent(
            llm=self.llm,
            use_perception=False,
            use_memory=False,
        )
        agent.decision._fallback_answer = "兜底"
        result = agent.run("随便问点什么")

        self.assertEqual(result["answer"], "兜底")
        self.assertTrue(result["success"])

    def test_max_steps_guard(self):
        """达到 max_steps 时应终止循环并给出提示。"""
        self.llm.responses = [
            '{"thought": "loop", "tool": "noop", "tool_input": {}, "final_answer": null}'
        ]
        agent = BaseAgent(llm=self.llm, use_perception=False, use_memory=False, max_steps=3)
        agent.register_tool("noop", lambda: "done")

        result = agent.run("一直循环")
        self.assertEqual(len(result["steps"]), 3)
        self.assertIn("未得出最终结论", result["answer"])

    def test_executor_error_is_contained(self):
        """工具抛异常不应中断主流程, 应被收敛进 observation.error。"""
        self.llm.responses = [
            '{"thought": "会失败", "tool": "boom", "tool_input": {}, "final_answer": null}'
        ]
        agent = BaseAgent(llm=self.llm, use_perception=False, use_memory=False, max_steps=1)
        agent.register_tool("boom", lambda: (_ for _ in ()).throw(ValueError("炸了")))

        result = agent.run("触发异常")
        self.assertFalse(result["steps"][0]["observation"]["success"])
        self.assertIn("炸了", result["steps"][0]["observation"]["error"])

    def test_output_messages_is_list(self):
        """输出应提供 messages 列表, 而非仅 answer 字符串。"""
        self.llm.responses = ['{"thought": "t", "final_answer": "答复内容"}']
        agent = BaseAgent(llm=self.llm, use_perception=False, use_memory=False)
        result = agent.run("你好")

        self.assertIsInstance(result["messages"], list)
        self.assertEqual(result["messages"][-1]["role"], "assistant")
        self.assertEqual(result["messages"][-1]["content"], "答复内容")
        # user 消息应在助手之前
        roles = [m["role"] for m in result["messages"]]
        self.assertEqual(roles, ["user", "assistant"])

    def test_output_messages_with_tool_observation(self):
        """消息顺序: 用户输入 -> 工具观测 -> 助手答复。"""
        self.llm.responses = [
            '{"thought": "计算", "tool": "add", "tool_input": {"a": 1, "b": 2}, "final_answer": null}'
        ]
        agent = BaseAgent(llm=self.llm, use_perception=False, use_memory=False, max_steps=1)
        agent.register_tool("add", lambda a, b: a + b)
        result = agent.run("1+2")

        roles = [m["role"] for m in result["messages"]]
        self.assertEqual(roles, ["user", "tool", "assistant"])
        self.assertIn("3", result["messages"][1]["content"])
        self.assertEqual(result["messages"][1]["name"], "add")

    def test_history_passed_as_message_list(self):
        """决策层收到的 history 应是消息列表。"""
        self.llm.responses = ['{"thought": "t", "final_answer": "ok"}']
        agent = BaseAgent(llm=self.llm, use_perception=False, use_memory=False)
        agent.run("你好")

        # 最后一条是 HumanMessage(用户输入), 其前不应混入历史字符串
        messages = self.llm.last_messages
        self.assertEqual(type(messages[-1]).__name__, "HumanMessage")
        self.assertEqual(type(messages[0]).__name__, "SystemMessage")

    def test_system_prompt_passthrough(self):
        """BaseAgent 的 system_prompt 应进入 LLM 的 system message。"""
        self.llm.responses = ['{"thought": "t", "final_answer": "ok"}']
        agent = BaseAgent(
            llm=self.llm, use_perception=False, use_memory=False, system_prompt="你是客服助手"
        )
        agent.run("你好")
        content = self.llm.last_messages[0].content
        self.assertIn("你是客服助手", content)

    def test_default_persona_without_system_prompt(self):
        self.llm.responses = ['{"thought": "t", "final_answer": "ok"}']
        agent = BaseAgent(llm=self.llm, use_perception=False, use_memory=False)
        agent.run("你好")
        self.assertIn(DEFAULT_SYSTEM_PROMPT, self.llm.last_messages[0].content)

    def test_build_system_prompt_hook(self):
        """覆盖 _build_system_prompt 可动态生成人设。"""

        class DynamicAgent(BaseAgent):
            def _build_system_prompt(self, user_input, perception_result):
                return f"动态人设:{user_input}"

        self.llm.responses = ['{"thought": "t", "final_answer": "ok"}']
        agent = DynamicAgent(llm=self.llm, use_perception=False, use_memory=False)
        agent.run("ABC")
        self.assertIn("动态人设:ABC", self.llm.last_messages[0].content)

    def test_memory_disabled_without_embedding(self):
        """未提供 embedding 时记忆层应为 None 且不影响运行。"""
        self.llm.responses = ['{"thought": "t", "final_answer": "ok"}']
        agent = BaseAgent(llm=self.llm, use_perception=False)
        self.assertIsNone(agent.memory)
        self.assertEqual(agent.run("测试")["answer"], "ok")


if __name__ == "__main__":
    unittest.main()
