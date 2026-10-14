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

"""DeepSeek LLM 后端实现。

基于 DeepSeek 的 OpenAI 兼容接口 (``/v1/chat/completions``) 封装。
官方文档: https://api-docs.deepseek.com/zh-cn/
"""
from __future__ import annotations

from snowland_agent_base.model.backends.openai_compat import (
    OpenAICompatEmbeddingBackend,
    OpenAICompatLLMBackend,
    OpenAICompatRerankBackend,
)


class DeepSeekLLMBackend(OpenAICompatLLMBackend):
    """基于 DeepSeek OpenAI 兼容接口的 ChatModel 后端。"""

    default_base_url = "https://api.deepseek.com/v1"
    api_key_env = "DEEPSEEK_API_KEY"
    base_url_env = "DEEPSEEK_BASE_URL"
    default_model = "deepseek-chat"


class DeepSeekEmbeddingBackend(OpenAICompatEmbeddingBackend):
    """DeepSeek 向量后端 (OpenAI 兼容 ``/embeddings``)。"""

    default_base_url = "https://api.deepseek.com/v1"
    api_key_env = "DEEPSEEK_API_KEY"
    base_url_env = "DEEPSEEK_BASE_URL"
    default_model = "deepseek-embedding"


class DeepSeekRerankBackend(OpenAICompatRerankBackend):
    """DeepSeek 语义重排后端 (OpenAI 兼容 ``/rerank``, 模型 ``deepseek-rerank``)。"""

    default_base_url = "https://api.deepseek.com/v1"
    api_key_env = "DEEPSEEK_API_KEY"
    base_url_env = "DEEPSEEK_BASE_URL"
    default_model = "deepseek-rerank"
