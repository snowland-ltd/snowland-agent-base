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

"""并发执行工具测试: map_parallel / run_parallel。

覆盖顺序保持、退化为串行、空输入、异常收敛 (catch_errors)、关闭 catch 抛错、
真实线程池并发, 以及共享 LazyChatModel 在并行下只构建一次。
"""
import unittest
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from snowland_agent_base.model import LazyChatModel
from snowland_agent_base.parallel import map_parallel, run_parallel


class TestMapParallel(unittest.TestCase):
    def test_basic_order_preserved(self):
        out = map_parallel(lambda x: x * 2, [1, 2, 3])
        self.assertEqual(out, [2, 4, 6])

    def test_single_item_serial(self):
        out = map_parallel(lambda x: x + 1, [5], max_workers=1)
        self.assertEqual(out, [6])

    def test_max_workers_le_one_serial(self):
        out = map_parallel(lambda x: x, [1, 2, 3], max_workers=0)
        self.assertEqual(out, [1, 2, 3])

    def test_empty(self):
        self.assertEqual(map_parallel(lambda x: x, []), [])

    def test_catch_errors_default(self):
        def fn(x):
            if x == 2:
                raise ValueError("boom")
            return x

        out = map_parallel(fn, [1, 2, 3])
        self.assertEqual(out[0], 1)
        self.assertEqual(out[2], 3)
        self.assertFalse(out[1]["success"])
        self.assertEqual(out[1]["error"], "boom")
        self.assertEqual(out[1]["item"], 2)

    def test_catch_errors_false_raises(self):
        def fn(x):
            if x == 2:
                raise ValueError("boom")
            return x

        with self.assertRaises(ValueError):
            map_parallel(fn, [1, 2, 3], catch_errors=False)

    def test_actually_parallel(self):
        out = map_parallel(lambda x: x + 1, range(20), max_workers=4)
        self.assertEqual(out, list(range(1, 21)))


class TestRunParallel(unittest.TestCase):
    def test_basic_named(self):
        out = run_parallel({"a": lambda: 1, "b": lambda: 2})
        self.assertEqual(out, {"a": 1, "b": 2})

    def test_single_serial(self):
        out = run_parallel({"a": lambda: 7}, max_workers=0)
        self.assertEqual(out, {"a": 7})

    def test_empty(self):
        self.assertEqual(run_parallel({}), {})

    def test_catch_errors_default(self):
        def fail():
            raise RuntimeError("x")

        out = run_parallel({"ok": lambda: 1, "bad": fail})
        self.assertEqual(out["ok"], 1)
        self.assertFalse(out["bad"]["success"])
        self.assertEqual(out["bad"]["name"], "bad")
        self.assertEqual(out["bad"]["error"], "x")

    def test_catch_errors_false_raises(self):
        def fail():
            raise RuntimeError("x")

        with self.assertRaises(RuntimeError):
            run_parallel({"ok": lambda: 1, "bad": fail}, catch_errors=False)

    def test_actually_parallel(self):
        out = run_parallel(
            {f"t{i}": (lambda i=i: i * 2) for i in range(10)}, max_workers=4
        )
        self.assertEqual(out, {f"t{i}": i * 2 for i in range(10)})


class _StubLLM(BaseChatModel):
    """最小可用 ChatModel, 用于共享懒加载复用测试; 每次固定返回 ``"ok"``。"""

    call_count: int = 0

    def __init__(self, responses=None, **kwargs):
        super().__init__(**kwargs)

    @property
    def _llm_type(self) -> str:
        return "stub"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.call_count += 1
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])


class TestSharedLazyModel(unittest.TestCase):
    def test_shared_lazy_builds_once(self):
        """多个并行任务共享同一 LazyChatModel, 底层仅构建一次。"""
        built = {"n": 0}

        def builder():
            built["n"] += 1
            return _StubLLM(responses=["ok"])

        lazy = LazyChatModel(builder=builder)
        out = run_parallel(
            {
                "a": lambda: lazy.invoke("x").content,
                "b": lambda: lazy.invoke("y").content,
            },
            max_workers=4,
        )
        self.assertEqual(out, {"a": "ok", "b": "ok"})
        self.assertEqual(built["n"], 1)

    def test_shared_lazy_via_threadpool(self):
        built = {"n": 0}

        def builder():
            built["n"] += 1
            return _StubLLM(responses=["ok"])

        lazy = LazyChatModel(builder=builder)
        with ThreadPoolExecutor(max_workers=6) as ex:
            results = list(ex.map(lambda _: lazy.invoke("x").content, range(30)))
        self.assertEqual(results, ["ok"] * 30)
        self.assertEqual(built["n"], 1)


if __name__ == "__main__":
    unittest.main()
