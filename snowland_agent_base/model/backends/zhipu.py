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

"""智谱 AI (BigModel / GLM) 后端实现。

- 推理: 基于智谱 OpenAI 兼容接口 (``/v1/chat/completions``) 封装;
- 视频: 基于智谱原生 CogVideo ``/video_generation`` 接口 (非 OpenAI 兼容)。
"""
from __future__ import annotations

import os
from typing import Optional

from snowland_agent_base.model.backends.openai_compat import (
    OpenAICompatEmbeddingBackend,
    OpenAICompatImageBackend,
    OpenAICompatLLMBackend,
    OpenAICompatRerankBackend,
    OpenAICompatTTSBackend,
)
from snowland_agent_base.model.base import (
    VideoBackend,
    VideoModel,
    VideoResult,
)
from snowland_agent_base.model.types import ModelType


class ZhipuLLMBackend(OpenAICompatLLMBackend):
    """基于智谱 AI (BigModel / GLM) OpenAI 兼容接口的 ChatModel 后端。"""

    default_base_url = "https://open.bigmodel.cn/api/paas/v4/"
    api_key_env = "ZHIPU_API_KEY"
    base_url_env = "ZHIPU_BASE_URL"
    default_model = "glm-4"


class ZhipuEmbeddingBackend(OpenAICompatEmbeddingBackend):
    """智谱向量后端 (OpenAI 兼容 ``/embeddings``)。"""

    default_base_url = "https://open.bigmodel.cn/api/paas/v4/"
    api_key_env = "ZHIPU_API_KEY"
    base_url_env = "ZHIPU_BASE_URL"
    default_model = "embedding-3"


class ZhipuImageBackend(OpenAICompatImageBackend):
    """智谱图像生成后端 (OpenAI 兼容 ``/images/generations``, CogView)。"""

    default_base_url = "https://open.bigmodel.cn/api/paas/v4/"
    api_key_env = "ZHIPU_API_KEY"
    base_url_env = "ZHIPU_BASE_URL"
    default_model = "cogview-3-plus"


class ZhipuTTSBackend(OpenAICompatTTSBackend):
    """智谱语音合成后端 (OpenAI 兼容 ``/audio/speech``)。"""

    default_base_url = "https://open.bigmodel.cn/api/paas/v4/"
    api_key_env = "ZHIPU_API_KEY"
    base_url_env = "ZHIPU_BASE_URL"
    default_model = "cosyvoice"


class ZhipuRerankBackend(OpenAICompatRerankBackend):
    """智谱语义重排后端 (OpenAI 兼容 ``/rerank``)。"""

    default_base_url = "https://open.bigmodel.cn/api/paas/v4/"
    api_key_env = "ZHIPU_API_KEY"
    base_url_env = "ZHIPU_BASE_URL"
    default_model = "rerank"


class ZhipuVideoModel(VideoModel):
    """智谱 CogVideo 模型: 异步提交任务并轮询至完成。"""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        timeout: int,
        poll_interval: int,
        **kwargs,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = timeout
        self._poll_interval = poll_interval
        self._kwargs = kwargs

    def generate(self, prompt: str, **kwargs) -> VideoResult:
        import time

        from openai import OpenAI

        client = OpenAI(api_key=self._api_key, base_url=self._base_url)
        body = {
            "model": self._model,
            "prompt": prompt,
            **self._kwargs,
            **kwargs,
        }
        raw = client.post("/video_generation", body=body)
        data = raw.model_dump() if hasattr(raw, "model_dump") else raw
        task_id = data.get("id") or data.get("task_id") or data.get("taskId")
        if not task_id:
            urls = []
            output = data.get("output") if isinstance(data, dict) else None
            if isinstance(output, dict) and output.get("video_url"):
                urls = [output["video_url"]]
            return VideoResult(urls=urls, raw=raw)
        return self._wait(task_id, client)

    def _wait(self, task_id: str, client) -> VideoResult:
        import time

        deadline = time.time() + self._timeout
        while time.time() < deadline:
            raw = client.get(f"/video_generation/{task_id}")
            data = raw.model_dump() if hasattr(raw, "model_dump") else raw
            output = data.get("output") if isinstance(data, dict) else None
            status = output.get("task_status") if isinstance(output, dict) else None
            if status in ("SUCCESS", "SUCCEEDED"):
                url = output.get("video_url") if isinstance(output, dict) else None
                return VideoResult(
                    urls=[url] if url else [], task_id=task_id, raw=raw
                )
            if status in ("FAIL", "FAILED"):
                return VideoResult(task_id=task_id, raw=raw)
            time.sleep(self._poll_interval)
        return VideoResult(task_id=task_id, raw=None)


class ZhipuVideoBackend(VideoBackend):
    """智谱视频生成后端 (CogVideo), 走智谱原生 ``/video_generation`` 接口。"""

    default_base_url = "https://open.bigmodel.cn/api/paas/v4/"
    api_key_env = "ZHIPU_API_KEY"
    base_url_env = "ZHIPU_BASE_URL"
    default_model = "cogvideo"

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 600,
        poll_interval: int = 10,
        **kwargs,
    ) -> None:
        self._model = model or self.default_model
        self._api_key = api_key or os.getenv(self.api_key_env)
        self._base_url = (
            base_url or os.getenv(self.base_url_env) or self.default_base_url
        )
        self._timeout = timeout
        self._poll_interval = poll_interval
        self._kwargs = kwargs

    def build(self) -> VideoModel:
        if not self._api_key:
            raise ValueError(
                "智谱 API Key 未提供: 请传入 api_key "
                "或设置环境变量 ZHIPU_API_KEY"
            )
        return ZhipuVideoModel(
            self._model,
            self._api_key,
            self._base_url,
            self._timeout,
            self._poll_interval,
            **self._kwargs,
        )
