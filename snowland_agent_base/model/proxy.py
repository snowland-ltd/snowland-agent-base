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

"""可复用的 ChatModel 代理: 把「懒加载」与「空响应自愈重试」抽象为通用能力。

两个代理都继承自 :class:`langchain_core.language_models.BaseChatModel`, 对调用方**完全
透明**——可以像普通 ChatModel 一样 ``invoke`` / ``ainvoke`` / ``bind_tools`` /
``stream`` …… 同时**补齐同步与异步两条调用链**:

- :class:`LazyChatModel`: 首次真正调用时才通过 ``builder`` 构造底层 LLM, 避免每个 Agent
  在构造期就拉起 ChatModel (适合按阶段懒构建);
- :class:`RetryOnEmptyChatModel`: 底层 LLM 返回空 ``content`` 时自动重试, 缓解推理模型
  把额度花在 ``reasoning`` 上导致 ``content`` 一个 token 都不剩的问题, 正常模型零额外开销。

二者可组合::

    RetryOnEmptyChatModel(inner=LazyChatModel(builder=build_llm))
"""
from __future__ import annotations

import logging
import threading
from typing import Any

from langchain_core.language_models import BaseChatModel
from pydantic import PrivateAttr

logger = logging.getLogger(__name__)


class LazyChatModel(BaseChatModel):
    """懒加载 ChatModel 代理: 首次真正调用时才构造底层 LLM。

    通过 ``builder`` 注入底层 LLM 的构造闭包; 同步与异步调用都会触发一次构建并缓存。
    """

    builder: Any = None
    _impl: Any = PrivateAttr(default=None)
    _build_lock: Any = PrivateAttr(default_factory=threading.Lock)

    @property
    def _llm_type(self) -> str:
        """返回模型类型标识。"""
        return "snowland-lazy"

    def _ensure(self) -> BaseChatModel:
        """按需构建并缓存底层 LLM (仅构建一次; 并行调用下线程安全)。"""
        if self._impl is None:
            with self._build_lock:
                if self._impl is None:  # 双重检查, 避免多线程各建一次
                    if self.builder is None:
                        raise RuntimeError(
                            "LazyChatModel.builder 未注入, 无法构建底层 LLM"
                        )
                    self._impl = self.builder()
        return self._impl

    # ------------------------------------------------------------------ #
    # 同步
    # ------------------------------------------------------------------ #
    def invoke(self, input, config=None, *, stop=None, **kwargs):
        """委托到底层 LLM 的同步调用 (触发懒加载)。"""
        return self._ensure().invoke(input, config=config, stop=stop, **kwargs)

    def bind_tools(self, tools, **kwargs):
        """委托到底层 LLM 的工具绑定 (触发懒加载)。"""
        return self._ensure().bind_tools(tools, **kwargs)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        """委托到底层 LLM 的低阶同步生成。"""
        return self._ensure()._generate(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        """委托到底层 LLM 的低阶同步流式生成。"""
        return self._ensure()._stream(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )

    def stream(self, input, config=None, *, stop=None, **kwargs):
        """委托到底层 LLM 的同步流式调用。"""
        return self._ensure().stream(input, config=config, stop=stop, **kwargs)

    # ------------------------------------------------------------------ #
    # 异步
    # ------------------------------------------------------------------ #
    async def ainvoke(self, input, config=None, *, stop=None, **kwargs):
        """委托到底层 LLM 的异步调用 (触发懒加载)。"""
        return await self._ensure().ainvoke(input, config=config, stop=stop, **kwargs)

    async def abind_tools(self, tools, **kwargs):
        """委托到底层 LLM 的异步工具绑定 (触发懒加载)。"""
        return await self._ensure().abind_tools(tools, **kwargs)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        """委托到底层 LLM 的低阶异步生成。"""
        return await self._ensure()._agenerate(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        """委托到底层 LLM 的低阶异步流式生成。"""
        return await self._ensure()._astream(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )

    async def astream(self, input, config=None, *, stop=None, **kwargs):
        """委托到底层 LLM 的异步流式调用。"""
        return await self._ensure().astream(input, config=config, stop=stop, **kwargs)


class RetryOnEmptyChatModel(BaseChatModel):
    """空 content 自动重试的 ChatModel 代理 (模型侧自愈, 与具体厂商无关)。

    推理模型在复杂提示下会把 completion 额度大量花在 ``reasoning`` 上, 撞满后
    ``finish_reason=length`` 且 ``content`` 一个 token 都不剩——上层决策层解析不出动作,
    只能走兜底话术。本代理在 ``invoke`` / ``ainvoke`` 层面重试, 提高拿到有效内容的概率;
    正常模型永不触发重试, 零额外开销。

    注意: 重试治标, 真正的根治是换不无界推理的模型。
    """

    inner: Any = None
    #: 重试次数 (不含首次); 总尝试次数 = max_retries + 1
    max_retries: int = 2

    @property
    def _llm_type(self) -> str:
        """返回模型类型标识。"""
        return "snowland-retry-empty"

    @staticmethod
    def _is_empty(resp: Any) -> bool:
        """判断一次 LLM 响应是否「空」: content 与 reasoning 皆为空才视为空。

        推理模型常把答案放在 ``reasoning_content`` (可能在 ``additional_kwargs`` 中, 也可能是
        直接属性), content 反而为空——此类响应不算「空」, 不重试。
        """
        content = getattr(resp, "content", "")
        if isinstance(content, str) and content.strip():
            return False
        reasoning = ""
        extra = getattr(resp, "additional_kwargs", None) or {}
        if isinstance(extra, dict):
            reasoning = extra.get("reasoning_content", "") or ""
        if not reasoning and hasattr(resp, "reasoning_content"):
            reasoning = getattr(resp, "reasoning_content", "") or ""
        return not (isinstance(reasoning, str) and reasoning.strip())

    def _retry_sync(self, call):
        """同步执行 ``call``; 返回空则重试, 耗尽后返回最后一次响应。"""
        last = None
        for attempt in range(1, self.max_retries + 2):
            resp = call()
            if not self._is_empty(resp):
                return resp
            last = resp
            logger.warning(
                "LLM 第 %s/%s 次返回空 content (疑似推理吃满额度), 重试",
                attempt,
                self.max_retries + 1,
            )
        return last

    async def _retry_async(self, acall):
        """异步执行 ``acall``; 返回空则重试, 耗尽后返回最后一次响应。"""
        last = None
        for attempt in range(1, self.max_retries + 2):
            resp = await acall()
            if not self._is_empty(resp):
                return resp
            last = resp
            logger.warning(
                "LLM(异步) 第 %s/%s 次返回空 content (疑似推理吃满额度), 重试",
                attempt,
                self.max_retries + 1,
            )
        return last

    @staticmethod
    def _wrap_bound(model: Any, max_retries: int) -> "RetryOnEmptyChatModel":
        """把 ``bind_tools`` 产出的绑定模型再用重试代理包一层, 使其调用也具备自愈能力。"""
        return RetryOnEmptyChatModel(inner=model, max_retries=max_retries)

    # ------------------------------------------------------------------ #
    # 同步
    # ------------------------------------------------------------------ #
    def invoke(self, input, config=None, *, stop=None, **kwargs):
        """调用底层 LLM; 若返回空 content 则重试, 耗尽后返回最后一次响应。"""
        return self._retry_sync(
            lambda: self.inner.invoke(input, config=config, stop=stop, **kwargs)
        )

    def bind_tools(self, tools, **kwargs):
        """绑定工具, 并把绑定结果再包一层重试代理。"""
        return self._wrap_bound(self.inner.bind_tools(tools, **kwargs), self.max_retries)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        """委托底层 LLM 的低阶同步生成 (带空响应重试)。"""
        return self._retry_sync(
            lambda: self.inner._generate(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
        )

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        """委托底层 LLM 的低阶同步流式生成。"""
        return self.inner._stream(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )

    def stream(self, input, config=None, *, stop=None, **kwargs):
        """委托底层 LLM 的同步流式调用。"""
        return self.inner.stream(input, config=config, stop=stop, **kwargs)

    # ------------------------------------------------------------------ #
    # 异步
    # ------------------------------------------------------------------ #
    async def ainvoke(self, input, config=None, *, stop=None, **kwargs):
        """异步调用底层 LLM; 若返回空 content 则重试, 耗尽后返回最后一次响应。"""
        return await self._retry_async(
            lambda: self.inner.ainvoke(input, config=config, stop=stop, **kwargs)
        )

    async def abind_tools(self, tools, **kwargs):
        """异步绑定工具, 并把绑定结果再包一层重试代理。"""
        return self._wrap_bound(
            await self.inner.abind_tools(tools, **kwargs), self.max_retries
        )

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        """委托底层 LLM 的低阶异步生成 (带空响应重试)。"""
        return await self._retry_async(
            lambda: self.inner._agenerate(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
        )

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        """委托底层 LLM 的低阶异步流式生成。"""
        return await self.inner._astream(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )

    async def astream(self, input, config=None, *, stop=None, **kwargs):
        """委托底层 LLM 的异步流式调用。"""
        return await self.inner.astream(input, config=config, stop=stop, **kwargs)
