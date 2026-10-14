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

"""模型后端抽象接口 (按能力类型分类)。

模型后端有两个正交维度:

- **厂商 backend** (dashscope / orcarouter / zhipu ...): 决定用哪家服务;
- **能力类型** (:class:`~snowland_agent_base.model.types.ModelType`): 决定产出什么
  (推理 ChatModel / 向量 Embeddings / 图像 / 视频 / 语音合成 / 语义重排)。

:class:`ModelBackend` 是所有能力类型后端统一抽象根; 它派生出六种同形态接口:

- :class:`LLMBackend`         (REASONING) -> ``BaseChatModel``
- :class:`EmbeddingBackend`   (EMBEDDING) -> ``Embeddings``
- :class:`ImageBackend`       (IMAGE)     -> ``ImageModel``
- :class:`VideoBackend`       (VIDEO)     -> ``VideoModel``
- :class:`TTSBackend`         (TTS)       -> ``TTSModel``
- :class:`RerankBackend`      (RERANK)    -> ``RerankModel``

调用方只依赖抽象与 ``ModelType``, 不感知具体厂商, 从而通用。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from snowland_agent_base.model.types import ModelType
from snowland_agent_base.messages import OpenAIMessage


# ---------------------------------------------------------------------------
# 非推理能力的产出对象 (LangChain 未对图像/视频/语音合成提供统一接口, 这里自定)
# ---------------------------------------------------------------------------


@dataclass
class ImageResult:
    """一次图像生成的产出。"""

    #: 生成图像的 URL 列表。
    urls: List[str] = field(default_factory=list)
    #: 厂商原始响应, 便于取额外字段。
    raw: Any = None


@dataclass
class VideoResult:
    """一次视频生成的产出。"""

    #: 生成视频的 URL 列表。
    urls: List[str] = field(default_factory=list)
    #: 异步任务 id (视频生成通常为异步, 可用其查询结果)。
    task_id: Optional[str] = None
    #: 厂商原始响应。
    raw: Any = None


@dataclass
class AudioResult:
    """一次语音合成的产出。"""

    #: 音频内容: bytes 或 URL, 取决于厂商返回形式。
    audio: Any = None
    #: 音频格式 (如 ``"wav"`` / ``"mp3"``), 可能为空。
    format: Optional[str] = None
    #: 厂商原始响应。
    raw: Any = None


class ImageModel(ABC):
    """图像生成模型接口: 由 :class:`ImageBackend` 的 ``build()`` 产出。"""

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> ImageResult:
        """按 prompt 生成图像。"""


class VideoModel(ABC):
    """视频生成模型接口: 由 :class:`VideoBackend` 的 ``build()`` 产出。"""

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> VideoResult:
        """按 prompt 生成视频 (通常为异步, 内部轮询至完成)。"""


class TTSModel(ABC):
    """语音合成模型接口: 由 :class:`TTSBackend` 的 ``build()`` 产出。"""

    @abstractmethod
    def synthesize(self, text: str, **kwargs: Any) -> AudioResult:
        """将文本合成为语音。"""


# ---------------------------------------------------------------------------
# 统一抽象根 + 五种能力类型后端
# ---------------------------------------------------------------------------


class ModelBackend(ABC):
    """所有模型后端 (无论能力类型) 的统一抽象根。

    子类通过 ``model_type`` 声明自身能力类型, 并实现 ``build()`` 产出对应能力的
    模型对象。仅推理类后端需要 ``message_class`` (其余能力类型默认 ``None``)。
    """

    #: 能力类型; 子类按自身能力覆写。
    model_type: ModelType = ModelType.REASONING
    #: 该后端配套的消息风格类; 仅推理类后端有意义, 默认 ``None``。
    message_class = None

    @abstractmethod
    def build(self) -> Any:
        """构建并返回对应能力的模型实例。"""


class LLMBackend(ModelBackend):
    """推理 / 对话后端: 产出 LangChain ``BaseChatModel``。

    该基类是「backend 派生出两种接口」的抽象根: 不同厂商后端 (OpenAI 兼容 /
    Anthropic 原生) 继承自它, 并实现**相同的方法** (``build`` 与 ``message_class``),
    调用方只依赖抽象, 不感知具体厂商, 从而通用。

    - ``build``: 构建 ChatModel 实例 (各后端实现不同);
    - ``message_class``: 该后端配套的消息风格类 (``OpenAIMessage`` /
      ``AnthropicMessage``), 子类覆写以绑定自身协议; 默认 ``OpenAIMessage``。
    """

    model_type = ModelType.REASONING
    message_class = OpenAIMessage

    @abstractmethod
    def build(self) -> BaseChatModel:
        """构建并返回一个 ``BaseChatModel`` 实例。"""


class EmbeddingBackend(ModelBackend):
    """向量 / 嵌入后端: 产出 LangChain ``Embeddings``。"""

    model_type = ModelType.EMBEDDING

    @abstractmethod
    def build(self) -> Embeddings:
        """构建并返回一个 ``Embeddings`` 实例。"""


class ImageBackend(ModelBackend):
    """图像生成后端: 产出 :class:`ImageModel`。"""

    model_type = ModelType.IMAGE

    @abstractmethod
    def build(self) -> ImageModel:
        """构建并返回一个 :class:`ImageModel` 实例。"""


class VideoBackend(ModelBackend):
    """视频生成后端: 产出 :class:`VideoModel`。"""

    model_type = ModelType.VIDEO

    @abstractmethod
    def build(self) -> VideoModel:
        """构建并返回一个 :class:`VideoModel` 实例。"""


class TTSBackend(ModelBackend):
    """语音合成后端: 产出 :class:`TTSModel`。"""

    model_type = ModelType.TTS

    @abstractmethod
    def build(self) -> TTSModel:
        """构建并返回一个 :class:`TTSModel` 实例。"""


@dataclass
class RerankItem:
    """重排结果中的单条命中。"""

    #: 该文档在原始 ``documents`` 列表中的下标。
    index: int = 0
    #: 文档原文 (若接口返回)。
    document: Any = None
    #: 与 query 的相关性得分。
    relevance_score: float = 0.0


@dataclass
class RerankResult:
    """一次语义重排 (Rerank) 的产出。"""

    #: 按相关性降序排列的命中列表。
    results: List[RerankItem] = field(default_factory=list)
    #: 厂商原始响应, 便于取额外字段。
    raw: Any = None


class RerankModel(ABC):
    """语义重排模型接口: 由 :class:`RerankBackend` 的 ``build()`` 产出。"""

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
        **kwargs: Any,
    ) -> RerankResult:
        """按 query 对 documents 重排, 返回相关性降序的结果。"""


class RerankBackend(ModelBackend):
    """语义重排后端: 产出 :class:`RerankModel`。"""

    model_type = ModelType.RERANK

    @abstractmethod
    def build(self) -> RerankModel:
        """构建并返回一个 :class:`RerankModel` 实例。"""
