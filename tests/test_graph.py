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

"""GraphAgent 测试: 覆盖持久化、人机介入、流式、子图编排。

未安装 langgraph 时自动跳过, 不影响 BaseAgent 的测试。
"""
import unittest

from snowland_agent_base.agent import BaseAgent
from snowland_agent_base.decision import AgentAction, Rule
from snowland_agent_base.messages import assistant_message, user_message
from snowland_agent_base.skill import SkillRegistry, skill

try:
    from langgraph.checkpoint.memory import InMemorySaver

    from snowland_agent_base.graph import GraphAgent

    LANGGRAPH_AVAILABLE = True
    SKIP_REASON = ""
except ImportError as exc:  # pragma: no cover
    LANGGRAPH_AVAILABLE = False
    SKIP_REASON = f"langgraph 未安装: {exc}"

from tests.test_agent import FakeChatModel


@unittest.skipUnless(LANGGRAPH_AVAILABLE, SKIP_REASON)
class TestGraphInheritance(unittest.TestCase):
    """锁定继承契约: GraphAgent 必须真正复用 BaseAgent, 而非复制粘贴实现。"""

    def setUp(self) -> None:
        self.llm = FakeChatModel()
        self.agent = GraphAgent(
            llm=self.llm, use_perception=False, use_memory=False, checkpointer=InMemorySaver()
        )

    def test_is_subclass_of_base_agent(self):
        self.assertTrue(issubclass(GraphAgent, BaseAgent))
        self.assertIsInstance(self.agent, BaseAgent)

    def test_inherited_helpers_not_overridden(self):
        """这些能力应直接复用基类实现, 不在此类重复定义。"""
        for name in (
            "_perceive",
            "_recall",
            "_memorize",
            "_build_history",
            "_render_observation",
            "_build_prompt_context",
            "_build_system_prompt",
            "add_rule",
            "register_tool",
        ):
            # 若被重复定义会提升维护成本, 故断言其解析到 BaseAgent 的实现
            self.assertIs(
                getattr(GraphAgent, name),
                getattr(BaseAgent, name),
                f"{name} 不应在 GraphAgent 中重复实现",
            )

    def test_run_is_polymorphic(self):
        """run() 应可用, 与 BaseAgent 保持多态。"""
        self.llm.responses = ['{"thought": "t", "final_answer": "多态答复"}']
        result = self.agent.run("你好", thread_id="poly-1")
        self.assertEqual(result["answer"], "多态答复")
        self.assertEqual(result["thread_id"], "poly-1")

    def test_layers_assembled_by_base(self):
        """四层应由 BaseAgent 装配。"""
        self.assertIsNotNone(self.agent.decision)
        self.assertIsNotNone(self.agent.executor)
        self.assertIsNone(self.agent.perception)  # use_perception=False
        self.assertIsNone(self.agent.memory)  # 无 embedding

    def test_add_rule_and_tool_from_base(self):
        """继承来的注册方法应正常工作。"""
        self.llm.responses = ['{"thought": "t", "final_answer": "ok"}']
        self.agent.add_rule(
            Rule(name="r", predicate=lambda t, c: True, action=AgentAction(final_answer="规则"))
        )
        self.agent.register_tool("noop", lambda: 1)

        self.assertEqual(self.agent.run("任意", thread_id="inher-1")["answer"], "规则")
        self.assertIn("noop", self.agent.executor.list_tools())


@unittest.skipUnless(LANGGRAPH_AVAILABLE, SKIP_REASON)
class TestHistoryInput(unittest.TestCase):
    """invoke / stream / run 应支持显式传入历史会话列表。"""

    def setUp(self) -> None:
        self.llm = FakeChatModel()
        self.agent = GraphAgent(
            llm=self.llm, use_perception=False, use_memory=False, checkpointer=InMemorySaver()
        )

    def test_invoke_with_history(self):
        """历史应作为独立消息传给模型, 而非塞进 system prompt。"""
        self.llm.responses = ['{"thought": "t", "final_answer": "ok"}']
        history = [
            {"role": "user", "content": "我叫小明"},
            {"role": "assistant", "content": "你好小明"},
        ]
        self.agent.invoke("我叫什么?", thread_id="h-1", history=history)

        messages = self.llm.last_messages
        contents = [getattr(m, "content", "") for m in messages]
        self.assertIn("我叫小明", contents)
        self.assertIn("你好小明", contents)
        # 历史不应被并入系统提示词
        self.assertNotIn("我叫小明", messages[0].content)

    def test_history_order(self):
        """顺序: system -> 历史(user, assistant) -> 当前输入。"""
        self.llm.responses = ['{"thought": "t", "final_answer": "ok"}']
        history = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
        ]
        self.agent.invoke("Q2", thread_id="h-2", history=history)

        types = [type(m).__name__ for m in self.llm.last_messages]
        self.assertEqual(types, ["SystemMessage", "HumanMessage", "AIMessage", "HumanMessage"])

    def test_output_messages_include_history(self):
        """输出的 messages 应含前置历史 + 本轮对话。"""
        self.llm.responses = ['{"thought": "t", "final_answer": "答复"}']
        history = [{"role": "user", "content": "上一轮"}]
        result = self.agent.invoke("本轮", thread_id="h-3", history=history)

        contents = [m["content"] for m in result["messages"]]
        self.assertEqual(contents, ["上一轮", "本轮", "答复"])
        self.assertEqual([m["role"] for m in result["messages"]], ["user", "user", "assistant"])

    def test_history_string_compat(self):
        """兼容字符串形式的历史。"""
        self.llm.responses = ['{"thought": "t", "final_answer": "ok"}']
        result = self.agent.invoke("x", thread_id="h-4", history="之前的对话")
        self.assertTrue(result["success"])
        self.assertIn("之前的对话", result["messages"][0]["content"])

    def test_no_history_still_works(self):
        """不传 history 时行为不变。"""
        self.llm.responses = ['{"thought": "t", "final_answer": "ok"}']
        result = self.agent.invoke("你好", thread_id="h-5")
        self.assertEqual([m["content"] for m in result["messages"]], ["你好", "ok"])

    def test_run_accepts_history(self):
        self.llm.responses = ['{"thought": "t", "final_answer": "ok"}']
        result = self.agent.run("Q", thread_id="h-6", history=[{"role": "user", "content": "旧"}])
        self.assertIn("旧", [m["content"] for m in result["messages"]])

    def test_stream_accepts_history(self):
        self.llm.responses = ['{"thought": "t", "final_answer": "ok"}']
        chunks = list(
            self.agent.stream(
                "Q", thread_id="h-7", stream_mode="values",
                history=[{"role": "user", "content": "旧"}],
            )
        )
        self.assertIn("旧", [m["content"] for m in chunks[-1]["messages"]])

    def test_history_helper_constructors(self):
        """支持用辅助函数构造历史。"""
        self.llm.responses = ['{"thought": "t", "final_answer": "ok"}']
        history = [user_message("你好"), assistant_message("嗨")]
        result = self.agent.invoke("再聊", thread_id="h-8", history=history)
        contents = [m["content"] for m in result["messages"]]
        self.assertEqual(contents, ["你好", "嗨", "再聊", "ok"])


@unittest.skipUnless(LANGGRAPH_AVAILABLE, SKIP_REASON)
class TestGraphAgent(unittest.TestCase):
    def setUp(self) -> None:
        self.llm = FakeChatModel()
        self.agent = GraphAgent(
            llm=self.llm, use_perception=False, use_memory=False, checkpointer=InMemorySaver()
        )

    def test_final_answer(self):
        self.llm.responses = ['{"thought": "t", "final_answer": "你好呀"}']
        result = self.agent.invoke("你好", thread_id="t-final")
        self.assertEqual(result["answer"], "你好呀")
        self.assertFalse(result["interrupted"])

    def test_tool_call(self):
        self.llm.responses = [
            '{"thought": "计算", "tool": "add", "tool_input": {"a": 1, "b": 2}, "final_answer": null}'
        ]
        # max_steps=1: FakeChatModel 恒定返回同一 tool call, 限制步数以隔离单步行为
        agent = GraphAgent(
            llm=self.llm, use_perception=False, use_memory=False, max_steps=1,
            checkpointer=InMemorySaver(),
        )
        agent.register_tool("add", lambda a, b: a + b)
        result = agent.invoke("1+2", thread_id="t-tool")

        self.assertEqual(len(result["steps"]), 1)
        self.assertEqual(result["steps"][0]["observation"]["result"], 3)
        self.assertIsNotNone(result["answer"])

    def test_rule_priority(self):
        """规则命中时直接 finalize, 不经过工具。"""
        self.agent.add_rule(
            Rule(name="r", predicate=lambda t, c: "ping" in t, action=AgentAction(final_answer="pong"))
        )
        result = self.agent.invoke("ping", thread_id="t-rule")
        self.assertEqual(result["answer"], "pong")
        self.assertEqual(result["steps"], [])

    def test_max_steps_guard(self):
        self.llm.responses = [
            '{"thought": "loop", "tool": "noop", "tool_input": {}, "final_answer": null}'
        ]
        agent = GraphAgent(
            llm=self.llm, use_perception=False, use_memory=False, max_steps=2,
            checkpointer=InMemorySaver(),
        )
        agent.register_tool("noop", lambda: "done")
        result = agent.invoke("循环", thread_id="t-steps")

        self.assertEqual(len(result["steps"]), 2)
        self.assertIn("未得出最终结论", result["answer"])

    def test_checkpoint_reuse_thread(self):
        """同一 thread_id 可断点续跑: 状态被 checkpointer 保存。"""
        self.llm.responses = ['{"thought": "t", "final_answer": "记住我"}']
        self.agent.invoke("第一轮", thread_id="t-ckpt")
        state = self.agent.get_state("t-ckpt")
        self.assertEqual(state.values["input"], "第一轮")
        self.assertEqual(state.values["answer"], "记住我")


@unittest.skipUnless(LANGGRAPH_AVAILABLE, SKIP_REASON)
class TestGraphInterrupt(unittest.TestCase):
    """人机介入: interrupt 需与 checkpointer + thread_id 配合。"""

    def setUp(self) -> None:
        self.llm = FakeChatModel()
        self.llm.responses = [
            '{"thought": "删订单", "tool": "delete_order", "tool_input": {"id": 123}, "final_answer": null}'
        ]
        self.executed = []
        self.agent = GraphAgent(
            llm=self.llm,
            use_perception=False,
            use_memory=False,
            checkpointer=InMemorySaver(),
            approval_tools=["delete_order"],
            # max_steps=1: 执行后即收敛, 避免 FakeChatModel 重复触发二次审批
            max_steps=1,
        )
        self.agent.register_tool(
            "delete_order", lambda id: self.executed.append(id) or f"已删除 {id}"
        )

    def test_interrupt_then_approve(self):
        st = self.agent.stream_events("删除订单 123", thread_id="hitl-1")
        _ = st.output

        self.assertTrue(st.interrupted)
        self.assertEqual(self.executed, [], "审批前不应执行工具")

        resumed = self.agent.resume(True, thread_id="hitl-1")
        final = resumed.output

        self.assertEqual(self.executed, [123], "审批后应执行工具")
        self.assertFalse(resumed.interrupted)

    def test_interrupt_then_reject(self):
        st = self.agent.stream_events("删除订单 123", thread_id="hitl-2")
        _ = st.output
        self.assertTrue(st.interrupted)

        resumed = self.agent.resume(False, thread_id="hitl-2")
        _ = resumed.output

        self.assertEqual(self.executed, [], "拒绝后不应执行工具")

    def test_approval_can_edit_input(self):
        """审批时可改写入参。"""
        st = self.agent.stream_events("删除订单 123", thread_id="hitl-3")
        _ = st.output

        resumed = self.agent.resume(
            {"approved": True, "tool_input": {"id": 456}}, thread_id="hitl-3"
        )
        _ = resumed.output

        self.assertEqual(self.executed, [456])

    def test_require_approval_callback(self):
        """自定义审批判定优先于 approval_tools。"""
        self.llm.responses = [
            '{"thought": "t", "tool": "safe", "tool_input": {}, "final_answer": null}'
        ]
        agent = GraphAgent(
            llm=self.llm,
            use_perception=False,
            use_memory=False,
            checkpointer=InMemorySaver(),
            require_approval=lambda tool, ti: tool == "safe",
        )
        agent.register_tool("safe", lambda: "ok")

        st = agent.stream_events("执行", thread_id="hitl-4")
        _ = st.output
        self.assertTrue(st.interrupted)

    def test_no_approval_when_not_required(self):
        """未列入审批名单的工具不应中断。"""
        self.llm.responses = [
            '{"thought": "t", "tool": "safe_read", "tool_input": {}, "final_answer": null}'
        ]
        agent = GraphAgent(
            llm=self.llm,
            use_perception=False,
            use_memory=False,
            checkpointer=InMemorySaver(),
            approval_tools=["delete_order"],
        )
        agent.register_tool("safe_read", lambda: "data")

        st = agent.stream_events("读取", thread_id="hitl-5")
        _ = st.output
        self.assertFalse(st.interrupted)


@unittest.skipUnless(LANGGRAPH_AVAILABLE, SKIP_REASON)
class TestGraphStreaming(unittest.TestCase):
    """流式输出。"""

    def setUp(self) -> None:
        self.llm = FakeChatModel()
        self.llm.responses = ['{"thought": "t", "final_answer": "流式答复"}']

    def test_stream_values(self):
        agent = GraphAgent(
            llm=self.llm, use_perception=False, use_memory=False, checkpointer=InMemorySaver()
        )
        chunks = list(agent.stream("你好", thread_id="s-1", stream_mode="values"))
        self.assertTrue(len(chunks) > 0)
        self.assertEqual(chunks[-1].get("answer"), "流式答复")

    def test_stream_updates(self):
        agent = GraphAgent(
            llm=self.llm, use_perception=False, use_memory=False, checkpointer=InMemorySaver()
        )
        chunks = list(agent.stream("你好", thread_id="s-2", stream_mode="updates"))
        node_names = [name for chunk in chunks for name in chunk]
        self.assertIn("finalize", node_names)


@unittest.skipUnless(LANGGRAPH_AVAILABLE, SKIP_REASON)
class TestGraphSkills(unittest.TestCase):
    """GraphAgent 的 skill 支持: 激活状态需贯穿整轮 (节点重跑幂等)。"""

    def setUp(self) -> None:
        self.llm = FakeChatModel()

    def test_skill_catalog_present(self):
        self.llm.responses = ['{"thought": "t", "final_answer": "ok"}']
        registry = SkillRegistry()
        registry.register(skill("pdf-processing", "处理 PDF 文档"))
        agent = GraphAgent(
            llm=self.llm,
            use_perception=False,
            use_memory=False,
            checkpointer=InMemorySaver(),
            skills=registry,
        )
        agent.invoke("你好", thread_id="sk-1")
        self.assertIn("- pdf-processing", self.llm.last_messages[0].content)

    def test_skill_activated_and_in_output(self):
        self.llm.responses = ['{"thought": "t", "final_answer": "ok"}']
        registry = SkillRegistry()
        registry.register(skill("pdf-processing", "处理 PDF 文档", "## 步骤\n1. 提取"))
        agent = GraphAgent(
            llm=self.llm,
            use_perception=False,
            use_memory=False,
            checkpointer=InMemorySaver(),
            skills=registry,
        )
        result = agent.invoke("帮我处理 PDF", thread_id="sk-2")

        self.assertEqual(result["skills"], ["pdf-processing"])
        self.assertIn("1. 提取", self.llm.last_messages[0].content)

    def test_skill_tools_available(self):
        """skill 携带的工具应可在图中被调用。"""
        self.llm.responses = [
            '{"thought": "计算", "tool": "add", "tool_input": {"a": 1, "b": 2}, "final_answer": null}'
        ]
        registry = SkillRegistry()
        registry.register(skill("calc", "做计算", "步骤", tools={"add": lambda a, b: a + b}))
        agent = GraphAgent(
            llm=self.llm,
            use_perception=False,
            use_memory=False,
            checkpointer=InMemorySaver(),
            skills=registry,
            max_steps=1,
        )
        result = agent.invoke("1+2", thread_id="sk-3")
        self.assertEqual(result["steps"][0]["observation"]["result"], 3)


@unittest.skipUnless(LANGGRAPH_AVAILABLE, SKIP_REASON)
class TestSubgraph(unittest.TestCase):
    """子图编排: 把另一个已编译 Agent 注册为工具。"""

    def setUp(self) -> None:
        self.llm = FakeChatModel()

    def test_subgraph_routing(self):
        # 子 Agent: 专门做翻译
        sub_llm = FakeChatModel()
        sub_llm.responses = ['{"thought": "t", "final_answer": "子图处理结果"}']
        sub = GraphAgent(
            llm=sub_llm, use_perception=False, use_memory=False, checkpointer=InMemorySaver()
        ).build()

        # 父 Agent: 决策层选择调用子图
        self.llm.responses = [
            '{"thought": "交给子图", "tool": "translator", "tool_input": {}, "final_answer": null}'
        ]
        parent = GraphAgent(
            llm=self.llm, use_perception=False, use_memory=False, checkpointer=InMemorySaver()
        )
        parent.add_subgraph("translator", sub)

        result = parent.invoke("帮我翻译", thread_id="sg-1")
        self.assertEqual(result["answer"], "子图处理结果")
        self.assertEqual(len(result["steps"]), 1)
        self.assertEqual(result["steps"][0]["action"]["tool"], "translator")

    def test_subgraph_listed_as_tool(self):
        sub_llm = FakeChatModel()
        sub = GraphAgent(
            llm=sub_llm, use_perception=False, use_memory=False, checkpointer=InMemorySaver()
        ).build()

        parent = GraphAgent(
            llm=self.llm, use_perception=False, use_memory=False, checkpointer=InMemorySaver()
        )
        parent.add_subgraph("translator", sub)
        self.assertIn("translator", parent._tool_names())


if __name__ == "__main__":
    unittest.main()
