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

"""统一自动后端: 一个对象可调全部能力类型。

:class:`AutoBackend` 持有一个默认厂商名称 (缺省 ``dashscope``, 唯一覆盖推理 / 向量 /
图像 / 视频 / 语音 / 重排六类的厂商), 将各能力方法委托给对应的 ``build_*`` 门面并
自动带上该厂商, 调用方无需为每个能力重复指定 backend::

    backend = AutoBackend(vendor="dashscope")
    backend.reasoning(model="qwen-plus")          # BaseChatModel
    backend.embedding()                            # Embeddings
    backend.image(model="wanx-v1").generate("猫")  # ImageModel
    backend.video(model="wanx-v1").generate("海浪")
    backend.tts(model="cosyvoice-v1").synthesize("你好")
    backend.rerank(query="...", documents=[...])  # RerankResult
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from snowland_agent_base.model.types import ModelType


class AutoBackend:
    """统一自动后端: 以单一厂商覆盖全部能力类型。

    每个能力方法都委托给 :mod:`snowland_agent_base.model.factory` 中的对应门面,
    自动使用构造时指定的 ``vendor``。调用方只需持有一个对象, 即可调用推理 / 向量 /
    图像 / 视频 / 语音 / 重排等全部能力。

    Args:
        vendor: 默认厂商名; 缺省 ``"dashscope"`` (唯一覆盖六类能力的厂商)。
        **common: 透传给所有 ``build_*`` 的通用参数 (如 ``api_key`` / ``base_url``),
            各方法的关键字参数会覆盖这里的同名项。
    """

    def __init__(self, vendor: str = "dashscope", **common: Any) -> None:
        self.vendor = vendor
        self._common = common

    def _merge(self, **overrides: Any) -> Dict[str, Any]:
        """合并通用参数与单次调用参数 (单次调用优先)。"""
        merged: Dict[str, Any] = dict(self._common)
        merged.update(overrides)
        return merged

    def reasoning(self, model: Optional[str] = None, **kwargs: Any):
        """构建推理 (ChatModel) 实例。"""
        from snowland_agent_base.model.factory import build_llm

        return build_llm(self.vendor, model=model, **self._merge(**kwargs))

    def embedding(self, model: Optional[str] = None, **kwargs: Any):
        """构建向量 (Embeddings) 实例。"""
        from snowland_agent_base.model.factory import build_embedding

        return build_embedding(self.vendor, model=model, **self._merge(**kwargs))

    def image(self, model: Optional[str] = None, **kwargs: Any):
        """构建图像生成 (ImageModel) 实例。"""
        from snowland_agent_base.model.factory import build_image

        return build_image(self.vendor, model=model, **self._merge(**kwargs))

    def video(self, model: Optional[str] = None, **kwargs: Any):
        """构建视频生成 (VideoModel) 实例。"""
        from snowland_agent_base.model.factory import build_video

        return build_video(self.vendor, model=model, **self._merge(**kwargs))

    def tts(self, model: Optional[str] = None, **kwargs: Any):
        """构建语音合成 (TTSModel) 实例。"""
        from snowland_agent_base.model.factory import build_tts

        return build_tts(self.vendor, model=model, **self._merge(**kwargs))

    def rerank(
        self,
        query: Optional[str] = None,
        documents: Optional[List[str]] = None,
        model: Optional[str] = None,
        top_n: int = 5,
        **kwargs: Any,
    ):
        """构建语义重排实例; 若同时传入 ``query`` 与 ``documents`` 则直接返回重排结果。

        Args:
            query: 查询文本; 与 ``documents`` 同时给出时, 直接执行重排。
            documents: 待重排文档列表; 与 ``query`` 同时给出时, 直接执行重排。
            model: 重排模型名。
            top_n: 返回前 N 条。
            kwargs: 透传给 ``build_rerank`` 的其他参数。

        Returns:
            :class:`RerankModel` (未给 query/documents) 或 :class:`RerankResult`
            (已给 query/documents 时直接执行)。
        """
        from snowland_agent_base.model.factory import build_rerank

        model_obj = build_rerank(
            self.vendor, model=model, top_n=top_n, **self._merge(**kwargs)
        )
        if query is not None and documents is not None:
            return model_obj.rerank(query, documents, top_n=top_n)
        return model_obj

    def build(
        self,
        model_type: Any,
        model: Optional[str] = None,
        **kwargs: Any,
    ):
        """通用构建入口: 按能力类型构建模型实例 (委托 ``build_model``)。"""
        from snowland_agent_base.model.factory import build_model

        if isinstance(model_type, str):
            model_type = ModelType(model_type)
        return build_model(
            model_type, self.vendor, model=model, **self._merge(**kwargs)
        )


def build_auto_backend(
    vendor: str = "dashscope", **common: Any
) -> AutoBackend:
    """构造一个统一自动后端实例 (门面便捷函数)。

    Args:
        vendor: 默认厂商名; 缺省 ``"dashscope"``。
        **common: 透传给各能力方法的通用参数。

    Returns:
        :class:`AutoBackend` 实例。
    """
    return AutoBackend(vendor=vendor, **common)
