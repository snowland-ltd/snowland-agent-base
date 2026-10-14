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

"""Anthropic LLM 后端实现 (OpenAI 兼容接口)。

Anthropic 官方提供 OpenAI 兼容的 ``/v1/chat/completions`` 接口, 可直接用 OpenAI
SDK 访问: ``base_url="https://api.anthropic.com/v1"``, ``api_key`` 即 Anthropic API Key。

本实现复用 :class:`OpenAICompatLLMBackend` 基类, 通过 ``langchain_openai.ChatOpenAI``
指向 Anthropic 的兼容端点。与 :class:`AnthropicLLMBackend` (原生 Messages API) 的区别
仅在于底层协议封装, 对外都统一返回 ``BaseChatModel``, 调用方无需关心差异。

官方文档: https://docs.anthropic.com/en/api/openai-sdk
"""
from __future__ import annotations

from snowland_agent_base.model.backends.openai_compat import (
    OpenAICompatLLMBackend,
)


class AnthropicOpenAILLMBackend(OpenAICompatLLMBackend):
    """基于 Anthropic OpenAI 兼容接口的 ChatModel 后端。"""

    default_base_url = "https://api.anthropic.com/v1"
    api_key_env = "ANTHROPIC_API_KEY"
    base_url_env = "ANTHROPIC_OPENAI_BASE_URL"
    default_model = "claude-sonnet-4-5"
