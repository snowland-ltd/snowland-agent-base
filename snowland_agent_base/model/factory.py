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

"""向后兼容门面: 按能力类型构建模型实例。"""
from __future__ import annotations

from typing import Optional, Union

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from snowland_agent_base.model.base import (
    ImageModel,
    RerankModel,
    TTSModel,
    VideoModel,
)
from snowland_agent_base.model.registry import create_model
from snowland_agent_base.model.types import ModelType


def build_llm(
    backend: str = "dashscope",
    model: str = "qwen-plus",
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    top_p: float = 0.8,
    enable_thinking: bool = False,
    **kwargs,
) -> BaseChatModel:
    """构建一个 ChatModel 实例 (推理 / 对话, 门面)。

    Args:
        backend: 后端名称或已注册别名, 默认 ``"dashscope"`` (亦可用 ``"bailian"``)。
        model: 平台模型名, 如 ``qwen-plus`` / ``qwen-max`` / ``qwen-turbo``。
        api_key: API Key; 缺省时从环境变量读取。
        temperature: 采样温度 (0~1)。
        max_tokens: 单次回复最大 token 数。
        top_p: 核采样概率。
        enable_thinking: 是否开启 qwq 系列模型的思考模式。
        kwargs: 透传给具体后端构造器的其他参数。

    Returns:
        LangChain ``BaseChatModel`` 实例。
    """
    return create_model(
        ModelType.REASONING,
        backend,
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        enable_thinking=enable_thinking,
        **kwargs,
    ).build()


def build_embedding(
    backend: str = "dashscope",
    model: str = "text-embedding-v2",
    **kwargs,
) -> Embeddings:
    """构建一个向量 / 嵌入模型实例 (``Embeddings``)。

    Args:
        backend: 后端名称; 默认 ``"dashscope"``。OpenAI 兼容厂商可用
            ``"orcarouter"`` / ``"zhipu"`` / ``"deepseek"`` / ``"openai-compat"``。
        model: 嵌入模型名; 缺省取各后端默认值。
        kwargs: 透传给具体后端构造器的其他参数。

    Returns:
        LangChain ``Embeddings`` 实例。
    """
    return create_model(
        ModelType.EMBEDDING, backend, model=model, **kwargs
    ).build()


def build_image(
    backend: str = "dashscope",
    model: str = "wanx-v1",
    **kwargs,
) -> ImageModel:
    """构建一个图像生成模型实例 (``ImageModel``)。

    Args:
        backend: 后端名称; 默认 ``"dashscope"``。
        model: 图像模型名; 缺省取各后端默认值。
        kwargs: 透传给具体后端构造器的其他参数。

    Returns:
        :class:`ImageModel` 实例, 调用 ``.generate(prompt)`` 生成图像。
    """
    return create_model(
        ModelType.IMAGE, backend, model=model, **kwargs
    ).build()


def build_video(
    backend: str = "dashscope",
    model: str = "wanx-v1",
    **kwargs,
) -> VideoModel:
    """构建一个视频生成模型实例 (``VideoModel``)。

    Args:
        backend: 后端名称; 默认 ``"dashscope"``。
        model: 视频模型名; 缺省取各后端默认值。
        kwargs: 透传给具体后端构造器的其他参数。

    Returns:
        :class:`VideoModel` 实例, 调用 ``.generate(prompt)`` 生成视频。
    """
    return create_model(
        ModelType.VIDEO, backend, model=model, **kwargs
    ).build()


def build_tts(
    backend: str = "dashscope",
    model: str = "cosyvoice-v1",
    **kwargs,
) -> TTSModel:
    """构建一个语音合成模型实例 (``TTSModel``)。

    Args:
        backend: 后端名称; 默认 ``"dashscope"``。
        model: 语音合成模型名; 缺省取各后端默认值。
        kwargs: 透传给具体后端构造器的其他参数 (如 ``voice``)。

    Returns:
        :class:`TTSModel` 实例, 调用 ``.synthesize(text)`` 合成语音。
    """
    return create_model(
        ModelType.TTS, backend, model=model, **kwargs
    ).build()


def build_model(
    model_type: Union[ModelType, str],
    backend: str = "dashscope",
    model: Optional[str] = None,
    **kwargs,
):
    """通用构建入口: 按能力类型 + 后端构建模型实例。

    Args:
        model_type: 能力类型 (:class:`ModelType` 或字符串, 如 ``"embedding"``)。
        backend: 后端名称。
        model: 模型名; 缺省取后端默认值。
        kwargs: 透传给具体后端构造器的其他参数。

    Returns:
        对应能力类型的模型对象。
    """
    if isinstance(model_type, str):
        model_type = ModelType(model_type)
    return create_model(model_type, backend, model=model, **kwargs).build()


def build_rerank(
    backend: str = "dashscope",
    model: str = "gte-rerank",
    top_n: int = 5,
    **kwargs,
) -> RerankModel:
    """构建一个语义重排模型实例 (``RerankModel``)。

    Args:
        backend: 后端名称; 默认 ``"dashscope"`` (百炼 gte-rerank)。OpenAI 兼容厂商可用
            ``"orcarouter"`` / ``"zhipu"`` / ``"deepseek"`` / ``"openai-compat"``。
        model: 重排模型名; 缺省取各后端默认值。
        top_n: 返回前 N 条; 缺省 5。
        kwargs: 透传给具体后端构造器的其他参数。

    Returns:
        :class:`RerankModel` 实例, 调用 ``.rerank(query, documents)`` 重排。
    """
    return create_model(
        ModelType.RERANK, backend, model=model, top_n=top_n, **kwargs
    ).build()
