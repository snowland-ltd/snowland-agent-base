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

"""百炼 (DashScope) LLM 后端实现。

基于阿里云百炼 (DashScope) 服务的 ChatModel 后端。
官方文档: https://help.aliyun.com/zh/model-studio/
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

from langchain_core.language_models import BaseChatModel

from snowland_agent_base._compat import ensure_pydantic_v1_shim
from snowland_agent_base.model.base import (
    AudioResult,
    EmbeddingBackend,
    ImageBackend,
    ImageModel,
    ImageResult,
    LLMBackend,
    RerankBackend,
    RerankItem,
    RerankModel,
    RerankResult,
    TTSBackend,
    TTSModel,
    VideoBackend,
    VideoModel,
    VideoResult,
)
from snowland_agent_base.model.types import ModelType


class DashScopeLLMBackend(LLMBackend):
    """基于阿里云百炼 (DashScope) 的 ChatModel 后端。

    官方文档: https://help.aliyun.com/zh/model-studio/
    """

    def __init__(
        self,
        model: str = "qwen-plus",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 0.8,
        enable_thinking: bool = False,
        **kwargs,
    ) -> None:
        """初始化百炼后端。

        Args:
            model: 百炼平台模型名, 如 ``qwen-plus`` / ``qwen-max`` / ``qwen-turbo``。
            api_key: DashScope API Key; 缺省时从环境变量 ``DASHSCOPE_API_KEY`` 读取。
            temperature: 采样温度 (0~1)。
            max_tokens: 单次回复最大 token 数。
            top_p: 核采样概率。
            enable_thinking: 是否开启 qwq 系列模型的思考模式。
            kwargs: 透传给 LangChain DashScope 构造器的其他参数。
        """
        self._params: Dict[str, object] = {
            "model": model,
            "api_key": api_key or os.getenv("DASHSCOPE_API_KEY"),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "enable_thinking": enable_thinking,
            **kwargs,
        }

    def build(self) -> BaseChatModel:
        if not self._params.get("api_key"):
            raise ValueError(
                "DashScope API Key 未提供: 请传入 api_key "
                "或设置环境变量 DASHSCOPE_API_KEY"
            )
        # langchain-dashscope 0.1.8 仍 import 已被移除的 langchain_core.pydantic_v1;
        # 用 pydantic.v1 注入 shim, 避免共享环境改动。
        ensure_pydantic_v1_shim()
        from langchain_dashscope import ChatDashScope

        return ChatDashScope(**self._params)


class DashScopeEmbeddingBackend(EmbeddingBackend):
    """百炼向量 / 嵌入后端 (``text-embedding-v2`` 等)。"""

    default_model = "text-embedding-v2"
    api_key_env = "DASHSCOPE_API_KEY"

    def __init__(
        self,
        model: str = "text-embedding-v2",
        api_key: Optional[str] = None,
        **kwargs,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.getenv(self.api_key_env)
        self._kwargs = kwargs

    def build(self):
        if not self._api_key:
            raise ValueError(
                "DashScope API Key 未提供: 请传入 api_key "
                "或设置环境变量 DASHSCOPE_API_KEY"
            )
        ensure_pydantic_v1_shim()
        from langchain_dashscope import DashScopeEmbeddings

        return DashScopeEmbeddings(
            model=self._model,
            dashscope_api_key=self._api_key,
            **self._kwargs,
        )


class DashScopeImageModel(ImageModel):
    """百炼图像生成模型 (wanx 系列), 包装 ``dashscope.ImageSynthesis``。"""

    def __init__(self, model: str, api_key: Optional[str]) -> None:
        self._model = model
        self._api_key = api_key

    def generate(self, prompt: str, n: int = 1, size: Optional[str] = None, **kwargs):
        from dashscope import ImageSynthesis

        params = {
            "model": self._model,
            "prompt": prompt,
            "n": n,
            "api_key": self._api_key,
            **kwargs,
        }
        if size:
            params["size"] = size
        rsp = ImageSynthesis.call(**params)
        results = getattr(getattr(rsp, "output", None), "results", []) or []
        urls = [getattr(r, "url", None) for r in results]
        return ImageResult(urls=[u for u in urls if u], raw=rsp)


class DashScopeImageBackend(ImageBackend):
    """百炼图像生成后端 (``wanx-v1`` 等)。"""

    default_model = "wanx-v1"
    api_key_env = "DASHSCOPE_API_KEY"

    def __init__(
        self,
        model: str = "wanx-v1",
        api_key: Optional[str] = None,
        **kwargs,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.getenv(self.api_key_env)

    def build(self):
        if not self._api_key:
            raise ValueError(
                "DashScope API Key 未提供: 请传入 api_key "
                "或设置环境变量 DASHSCOPE_API_KEY"
            )
        return DashScopeImageModel(self._model, self._api_key)


class DashScopeVideoModel(VideoModel):
    """百炼视频生成模型 (wanx 视频系列), 包装 ``dashscope.VideoSynthesis``。

    视频生成为异步任务: ``generate`` 先发起任务并轮询至完成或超时, 返回最终结果。
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str],
        timeout: int = 600,
        poll_interval: int = 5,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._timeout = timeout
        self._poll_interval = poll_interval

    def generate(self, prompt: str, **kwargs):
        import time

        from dashscope import VideoSynthesis

        rsp = VideoSynthesis.call(
            model=self._model,
            prompt=prompt,
            api_key=self._api_key,
            **kwargs,
        )
        task_id = getattr(getattr(rsp, "output", None), "task_id", None)
        if not task_id:
            # 同步返回或异常, 直接透传原始响应
            return VideoResult(raw=rsp)
        return self._wait(task_id)

    def _wait(self, task_id: str) -> VideoResult:
        import time

        from dashscope import VideoSynthesis

        deadline = time.time() + self._timeout
        while time.time() < deadline:
            rsp = VideoSynthesis.fetch(task_id, api_key=self._api_key)
            status = getattr(getattr(rsp, "output", None), "task_status", None)
            if status == "SUCCEEDED":
                url = getattr(getattr(rsp, "output", None), "video_url", None)
                return VideoResult(
                    urls=[url] if url else [], task_id=task_id, raw=rsp
                )
            if status == "FAILED":
                return VideoResult(task_id=task_id, raw=rsp)
            time.sleep(self._poll_interval)
        return VideoResult(task_id=task_id, raw=None)


class DashScopeVideoBackend(VideoBackend):
    """百炼视频生成后端 (``wanx-v1`` 等)。"""

    default_model = "wanx-v1"
    api_key_env = "DASHSCOPE_API_KEY"

    def __init__(
        self,
        model: str = "wanx-v1",
        api_key: Optional[str] = None,
        timeout: int = 600,
        poll_interval: int = 5,
        **kwargs,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.getenv(self.api_key_env)
        self._timeout = timeout
        self._poll_interval = poll_interval

    def build(self):
        if not self._api_key:
            raise ValueError(
                "DashScope API Key 未提供: 请传入 api_key "
                "或设置环境变量 DASHSCOPE_API_KEY"
            )
        return DashScopeVideoModel(
            self._model, self._api_key, self._timeout, self._poll_interval
        )


class DashScopeTTSModel(TTSModel):
    """百炼语音合成模型 (cosyvoice 系列), 包装 ``dashscope.SpeechSynthesizer``。"""

    def __init__(self, model: str, api_key: Optional[str], voice: str) -> None:
        self._model = model
        self._api_key = api_key
        self._voice = voice

    def synthesize(self, text: str, voice: Optional[str] = None, **kwargs):
        from dashscope import SpeechSynthesizer

        rsp = SpeechSynthesizer.call(
            model=self._model,
            text=text,
            voice=voice or self._voice,
            api_key=self._api_key,
            **kwargs,
        )
        audio = getattr(getattr(rsp, "output", None), "audio", None)
        return AudioResult(audio=audio, raw=rsp)


class DashScopeTTSBackend(TTSBackend):
    """百炼语音合成后端 (``cosyvoice-v1`` 等)。"""

    default_model = "cosyvoice-v1"
    default_voice = "longxiaochun"
    api_key_env = "DASHSCOPE_API_KEY"

    def __init__(
        self,
        model: str = "cosyvoice-v1",
        api_key: Optional[str] = None,
        voice: str = "longxiaochun",
        **kwargs,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.getenv(self.api_key_env)
        self._voice = voice

    def build(self):
        if not self._api_key:
            raise ValueError(
                "DashScope API Key 未提供: 请传入 api_key "
                "或设置环境变量 DASHSCOPE_API_KEY"
            )
        return DashScopeTTSModel(self._model, self._api_key, self._voice)


class DashScopeRerankModel(RerankModel):
    """百炼重排模型, 包装 ``dashscope.Rank`` (需 dashscope>=1.14)。"""

    def __init__(self, model: str, api_key: str, top_n: int) -> None:
        self._model = model
        self._api_key = api_key
        self._top_n = top_n

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
        **kwargs,
    ) -> RerankResult:
        from dashscope import Rank

        resp = Rank.call(
            model=self._model,
            query=query,
            documents=documents,
            top_n=top_n or self._top_n,
            api_key=self._api_key,
            **kwargs,
        )
        out = getattr(resp, "output", None)
        results = getattr(out, "results", []) or []
        items = [
            RerankItem(
                index=getattr(r, "index", i),
                document=getattr(r, "document", None),
                relevance_score=getattr(r, "relevance_score", 0.0),
            )
            for i, r in enumerate(results)
        ]
        return RerankResult(results=items, raw=resp)


class DashScopeRerankBackend(RerankBackend):
    """百炼语义重排后端 (``gte-rerank`` / ``text-rerank`` 等)。"""

    default_model = "gte-rerank"
    api_key_env = "DASHSCOPE_API_KEY"

    def __init__(
        self,
        model: str = "gte-rerank",
        api_key: Optional[str] = None,
        top_n: int = 5,
        **kwargs,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.getenv(self.api_key_env)
        self._top_n = top_n
        self._kwargs = kwargs

    def build(self) -> RerankModel:
        if not self._api_key:
            raise ValueError(
                "DashScope API Key 未提供: 请传入 api_key "
                "或设置环境变量 DASHSCOPE_API_KEY"
            )
        return DashScopeRerankModel(self._model, self._api_key, self._top_n)
