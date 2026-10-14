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

"""可插拔模型接入层 (按能力类型分类)。

模型后端有两个正交维度: **厂商 backend** (dashscope / orcarouter / zhipu ...) 与
**能力类型** (:class:`ModelType`: 推理 / 向量 / 图像 / 视频 / 语音合成 / 语义重排)。
所有后端继承自统一抽象根 :class:`ModelBackend`, 并按能力类型派生出同形态接口
:class:`LLMBackend` / :class:`EmbeddingBackend` / :class:`ImageBackend` /
:class:`VideoBackend` / :class:`TTSBackend` / :class:`RerankBackend`。

内置后端 (以默认厂商百炼为例, 同一名称可注册多种能力)::

    build_llm(backend="dashscope", model="qwen-plus")        # 推理 / 对话
    build_embedding(backend="dashscope")                      # 向量 / 嵌入
    build_image(backend="dashscope", model="wanx-v1")         # 图像生成
    build_video(backend="dashscope", model="wanx-v1")         # 视频生成
    build_tts(backend="dashscope", model="cosyvoice-v1")      # 语音合成
    build_rerank(backend="dashscope", model="gte-rerank")     # 语义重排

OpenAI 兼容厂商 (orcarouter / zhipu / deepseek) 同样提供向量 / 图像 / 语音 / 重排能力::

    build_embedding(backend="orcarouter")
    build_image(backend="orcarouter")
    build_rerank(backend="orcarouter")

新增后端: 继承对应能力基类, 再 ``register_llm_backend(name, cls, model_type=...)``
注册, 即可被名称 + 能力类型解析, 实现真正的插拔式扩展。
"""
from __future__ import annotations

from snowland_agent_base.model.base import (
    AudioResult,
    EmbeddingBackend,
    ImageBackend,
    ImageModel,
    ImageResult,
    LLMBackend,
    ModelBackend,
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
from snowland_agent_base.model.backends import (
    DashScopeEmbeddingBackend,
    DashScopeImageBackend,
    DashScopeImageModel,
    DashScopeLLMBackend,
    DashScopeRerankBackend,
    DashScopeRerankModel,
    DashScopeTTSBackend,
    DashScopeTTSModel,
    DashScopeVideoBackend,
    DashScopeVideoModel,
    DeepSeekLLMBackend,
    OrcaRouterLLMBackend,
    OpenAICompatEmbeddingBackend,
    OpenAICompatImageBackend,
    OpenAICompatImageModel,
    OpenAICompatLLMBackend,
    OpenAICompatRerankBackend,
    OpenAICompatRerankModel,
    OpenAICompatTTSBackend,
    OpenAICompatTTSModel,
    OpenAICompatVideoBackend,
    OpenAICompatVideoModel,
    ORCAROUTER_REF,
    ZhipuLLMBackend,
    ZhipuVideoBackend,
    ZhipuVideoModel,
    AnthropicLLMBackend,
    AnthropicOpenAILLMBackend,
    build_orcarouter_auth_url,
    exchange_orcarouter_code,
)
from snowland_agent_base.model.factory import (
    build_embedding,
    build_image,
    build_llm,
    build_model,
    build_rerank,
    build_tts,
    build_video,
)
from snowland_agent_base.model.auto_backend import (
    AutoBackend,
    build_auto_backend,
)
from snowland_agent_base.model.registry import (
    create_llm_backend,
    create_model,
    get_llm_backend,
    get_message_class,
    list_llm_backends,
    register_llm_backend,
)
from snowland_agent_base.model.types import ModelType


__all__ = [
    "ModelType",
    "ModelBackend",
    "LLMBackend",
    "EmbeddingBackend",
    "ImageBackend",
    "VideoBackend",
    "TTSBackend",
    "RerankBackend",
    "ImageModel",
    "VideoModel",
    "TTSModel",
    "RerankModel",
    "ImageResult",
    "VideoResult",
    "AudioResult",
    "RerankResult",
    "RerankItem",
    "DashScopeLLMBackend",
    "DashScopeEmbeddingBackend",
    "DashScopeImageBackend",
    "DashScopeImageModel",
    "DashScopeVideoBackend",
    "DashScopeVideoModel",
    "DashScopeTTSBackend",
    "DashScopeTTSModel",
    "DashScopeRerankBackend",
    "DashScopeRerankModel",
    "OrcaRouterLLMBackend",
    "ZhipuLLMBackend",
    "ZhipuVideoBackend",
    "ZhipuVideoModel",
    "DeepSeekLLMBackend",
    "AnthropicLLMBackend",
    "AnthropicOpenAILLMBackend",
    "OpenAICompatLLMBackend",
    "OpenAICompatEmbeddingBackend",
    "OpenAICompatImageBackend",
    "OpenAICompatImageModel",
    "OpenAICompatVideoBackend",
    "OpenAICompatVideoModel",
    "OpenAICompatTTSBackend",
    "OpenAICompatTTSModel",
    "OpenAICompatRerankBackend",
    "OpenAICompatRerankModel",
    "ORCAROUTER_REF",
    "build_orcarouter_auth_url",
    "exchange_orcarouter_code",
    "build_llm",
    "build_embedding",
    "build_image",
    "build_video",
    "build_tts",
    "build_rerank",
    "build_model",
    "AutoBackend",
    "build_auto_backend",
    "register_llm_backend",
    "get_llm_backend",
    "list_llm_backends",
    "create_llm_backend",
    "create_model",
    "get_message_class",
]
