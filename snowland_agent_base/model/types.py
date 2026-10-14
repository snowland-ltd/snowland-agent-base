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

"""模型能力类型枚举 (LLM 分类的第二维度)。

第一维度是「厂商 backend」(dashscope / orcarouter / zhipu ...), 第二维度是
「能力类型」(推理 / 向量 / 图像 / 视频 / 语音合成)。两者正交: 同一厂商后端
可提供多种能力, 例如百炼既产出推理 ChatModel, 也产出 Embeddings / 图像 / 视频 /
语音合成。
"""
from __future__ import annotations

from enum import Enum


class ModelType(str, Enum):
    """模型能力类型。"""

    #: 推理 / 对话 (产出 BaseChatModel)
    REASONING = "reasoning"
    #: 向量 / 嵌入 (产出 Embeddings)
    EMBEDDING = "embedding"
    #: 图像生成 (产出 ImageModel)
    IMAGE = "image"
    #: 视频生成 (产出 VideoModel)
    VIDEO = "video"
    #: 语音合成 (产出 TTSModel)
    TTS = "tts"
    #: 语义重排 (产出 RerankModel)
    RERANK = "rerank"

    def __str__(self) -> str:
        return self.value
