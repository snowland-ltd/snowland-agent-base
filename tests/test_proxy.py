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

"""ChatModel 代理测试: LazyChatModel (懒加载) / RetryOnEmptyChatModel (空响应自愈)
以及与 BaseAgent 的集成。全程不依赖网络 —— 用 FakeChatModel 模拟底层 LLM。
"""
import asyncio
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from snowland_agent_base.agent import BaseAgent
from snowland_agent_base.model import LazyChatModel, RetryOnEmptyChatModel


class FakeChatModel(BaseChatModel):
    """假 ChatModel: 每次 ``_generate`` 消费一个预设响应。

    response 元素支持:
      - ``str``: 直接作为 ``content``;
      - ``(str, dict)``: ``content`` + 注入 ``additional_kwargs`` (用于模拟 reasoning_content)。
    """

    responses: List[Any] = []
    call_count: int = 0

    def __init__(self, responses=None, **kwargs):
        super().__init__(**kwargs)
        if responses is not None:
            self.responses = list(responses)

    @property
    def _llm_type(self) -> str:
        return "fake-chat-model"

    def bind_tools(self, tools, **kwargs):
        # 测试用: 复用自身, 保留 _generate 行为
        return self

    def _make(self, raw):
        if isinstance(raw, tuple):
            content, extra = raw
        else:
            content, extra = raw, {}
        msg = AIMessage(content=content)
        if extra:
            msg.additional_kwargs.update(extra)
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.call_count += 1
        raw = self.responses.pop(0) if self.responses else ""
        return self._make(raw)


class TestLazyChatModel(unittest.TestCase):
    def test_not_built_until_called(self):
        """构造期不应触发 builder, 首次调用才构建。"""
        built = {"n": 0}

        def builder():
            built["n"] += 1
            return FakeChatModel(responses=["hi"])

        lazy = LazyChatModel(builder=builder)
        self.assertEqual(built["n"], 0)
        self.assertEqual(lazy.invoke("x").content, "hi")
        self.assertEqual(built["n"], 1)

    def test_build_once_cached(self):
        """多次调用只构建一次底层 LLM。"""
        built = {"n": 0}

        def builder():
            built["n"] += 1
            return FakeChatModel(responses=["hi", "hi", "hi"])

        lazy = LazyChatModel(builder=builder)
        for _ in range(3):
            self.assertEqual(lazy.invoke("x").content, "hi")
        self.assertEqual(built["n"], 1)

    def test_thread_safe_single_build(self):
        """多线程并发调用, builder 仅执行一次。"""
        built = {"n": 0}
        lock = threading.Lock()

        def builder():
            with lock:
                built["n"] += 1
            return FakeChatModel(responses=["ok"])

        lazy = LazyChatModel(builder=builder)
        with ThreadPoolExecutor(max_workers=8) as ex:
            list(ex.map(lambda _: lazy.invoke("x"), range(50)))
        self.assertEqual(built["n"], 1)

    def test_ainvoke_delegates(self):
        async def run():
            lazy = LazyChatModel(builder=lambda: FakeChatModel(responses=["async-ok"]))
            return await lazy.ainvoke("x")

        self.assertEqual(asyncio.run(run()).content, "async-ok")

    def test_llm_type(self):
        lazy = LazyChatModel(builder=lambda: FakeChatModel())
        self.assertEqual(lazy._llm_type, "snowland-lazy")

    def test_missing_builder_raises(self):
        lazy = LazyChatModel()
        with self.assertRaises(RuntimeError):
            lazy.invoke("x")


class TestRetryOnEmptyChatModel(unittest.TestCase):
    def test_normal_no_retry(self):
        """正常非空响应只调用一次底层, 不触发重试。"""
        inner = FakeChatModel(responses=["answer"])
        proxy = RetryOnEmptyChatModel(inner=inner)
        self.assertEqual(proxy.invoke("q").content, "answer")
        self.assertEqual(inner.call_count, 1)

    def test_empty_then_ok_retries(self):
        """首次空 content, 重试后拿到有效内容。"""
        inner = FakeChatModel(responses=["", "ok"])
        proxy = RetryOnEmptyChatModel(inner=inner, max_retries=2)
        self.assertEqual(proxy.invoke("q").content, "ok")
        self.assertEqual(inner.call_count, 2)

    def test_all_empty_returns_last(self):
        """全部为空时返回最后一次 (空) 响应, 调用次数 = max_retries + 1。"""
        inner = FakeChatModel(responses=["", "", ""])
        proxy = RetryOnEmptyChatModel(inner=inner, max_retries=2)
        self.assertEqual(proxy.invoke("q").content, "")
        self.assertEqual(inner.call_count, 3)

    def test_reasoning_not_empty(self):
        """content 为空但带 reasoning_content 时不算空, 不重试。"""
        inner = FakeChatModel(responses=[("", {"reasoning_content": "thinking..."})])
        proxy = RetryOnEmptyChatModel(inner=inner, max_retries=2)
        self.assertEqual(proxy.invoke("q").content, "")
        self.assertEqual(inner.call_count, 1)

    def test_is_empty_static(self):
        self.assertTrue(RetryOnEmptyChatModel._is_empty(AIMessage(content="")))
        self.assertFalse(RetryOnEmptyChatModel._is_empty(AIMessage(content="hi")))
        reasoned = AIMessage(content="")
        reasoned.additional_kwargs["reasoning_content"] = "x"
        self.assertFalse(RetryOnEmptyChatModel._is_empty(reasoned))

    def test_combine_lazy_and_retry(self):
        """RetryOnEmptyChatModel(inner=LazyChatModel(...)) 组合可用。"""
        inner = FakeChatModel(responses=["", "ok"])
        proxy = RetryOnEmptyChatModel(inner=LazyChatModel(builder=lambda: inner))
        self.assertEqual(proxy.invoke("q").content, "ok")
        self.assertEqual(inner.call_count, 2)

    def test_ainvoke_retry(self):
        async def run():
            inner = FakeChatModel(responses=["", "ok"])
            proxy = RetryOnEmptyChatModel(inner=inner, max_retries=3)
            return proxy, await proxy.ainvoke("q")

        proxy, result = asyncio.run(run())
        self.assertEqual(result.content, "ok")
        self.assertEqual(proxy.inner.call_count, 2)

    def test_bind_tools_wraps_again(self):
        """bind_tools 的绑定结果再包一层重试代理。"""
        inner = FakeChatModel(responses=["", "ok"])
        proxy = RetryOnEmptyChatModel(inner=inner, max_retries=2)
        bound = proxy.bind_tools([{"type": "function", "function": {"name": "f"}}])
        self.assertIsInstance(bound, RetryOnEmptyChatModel)
        self.assertEqual(bound.invoke("q").content, "ok")
        self.assertEqual(inner.call_count, 2)


class TestAgentRetryIntegration(unittest.TestCase):
    def test_enable_retry_wraps(self):
        """默认 enable_retry=True 会把 LLM 包成 RetryOnEmptyChatModel。"""
        agent = BaseAgent(
            llm=FakeChatModel(responses=["ok"]),
            use_perception=False,
            use_memory=False,
            enable_retry=True,
        )
        self.assertIsInstance(agent.llm, RetryOnEmptyChatModel)

    def test_disable_retry_no_wrap(self):
        agent = BaseAgent(
            llm=FakeChatModel(responses=["ok"]),
            use_perception=False,
            use_memory=False,
            enable_retry=False,
        )
        self.assertNotIsInstance(agent.llm, RetryOnEmptyChatModel)

    def test_lazy_inner_wrapped_with_retry(self):
        """传入 LazyChatModel 时, 外层再叠加重试, 同时具备懒加载 + 重试。"""
        agent = BaseAgent(
            llm=LazyChatModel(builder=lambda: FakeChatModel(responses=["ok"])),
            use_perception=False,
            use_memory=False,
            enable_retry=True,
        )
        self.assertIsInstance(agent.llm, RetryOnEmptyChatModel)
        self.assertIsInstance(agent.llm.inner, LazyChatModel)


if __name__ == "__main__":
    unittest.main()
