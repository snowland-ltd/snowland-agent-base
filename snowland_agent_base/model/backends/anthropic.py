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

"""Anthropic LLM 后端实现 (原生 Messages API)。

基于 Anthropic 官方 Messages API (``/v1/messages``), 使用 ``langchain_anthropic.ChatAnthropic``。

与 OpenAI 不同, Anthropic 的消息协议是独立的:
- ``system`` 提示词是顶层字段, 而非 ``role="system"`` 的 message;
- ``content`` 是内容块数组 (text / tool_use / tool_result 等), 不是纯文本字符串;
- 鉴权走请求头 ``x-api-key`` 与 ``anthropic-version``。

``ChatAnthropic`` 会在内部把 LangChain 消息 (``HumanMessage`` / ``SystemMessage`` /
``AIMessage``) 转换为 Anthropic 原生格式, 因此对外仍统一返回 ``BaseChatModel``,
呼叫方无需关心底层协议差异。

官方文档: https://docs.anthropic.com/en/api/messages
"""
from __future__ import annotations

import os
from typing import Dict, Optional

from langchain_core.language_models import BaseChatModel

from snowland_agent_base._compat import ensure_pydantic_v1_shim
from snowland_agent_base.model.base import LLMBackend
from snowland_agent_base.messages import AnthropicMessage


class AnthropicLLMBackend(LLMBackend):
    """基于 Anthropic 原生 Messages API 的 ChatModel 后端。"""

    #: 该后端使用的消息风格 (Anthropic 原生 Messages API -> AnthropicMessage)。
    message_class = AnthropicMessage

    #: Anthropic 官方 API 地址 (可经 base_url / ANTHROPIC_BASE_URL 覆盖)。
    default_base_url: str = "https://api.anthropic.com"

    def __init__(
        self,
        model: str = "claude-sonnet-4-5",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 0.8,
        **kwargs,
    ) -> None:
        """初始化 Anthropic 后端。

        Args:
            model: Anthropic 模型名, 如 ``claude-sonnet-4-5`` / ``claude-opus-4-5``。
            api_key: Anthropic API Key; 缺省时从环境变量 ``ANTHROPIC_API_KEY`` 读取。
            base_url: 自建/代理网关地址; 缺省时从环境变量 ``ANTHROPIC_BASE_URL`` 读取,
                再回退到官方地址 ``default_base_url``。
            temperature: 采样温度 (0~1)。
            max_tokens: 单次回复最大 token 数 (Anthropic 必填)。
            top_p: 核采样概率。
            kwargs: 透传给 ``ChatAnthropic`` 的其他参数 (如 ``timeout`` / ``metadata``)。
        """
        resolved_base_url = (
            base_url or os.getenv("ANTHROPIC_BASE_URL") or self.default_base_url
        )
        self._params: Dict[str, object] = {
            "model": model,
            "api_key": api_key or os.getenv("ANTHROPIC_API_KEY"),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            **kwargs,
        }
        self._params["anthropic_api_url"] = resolved_base_url

    def build(self) -> BaseChatModel:
        if not self._params.get("api_key"):
            raise ValueError(
                "Anthropic API Key 未提供: 请传入 api_key "
                "或设置环境变量 ANTHROPIC_API_KEY"
            )
        # 共享环境 langchain-core 1.x 已移除 pydantic_v1;
        # 用 pydantic.v1 注入 shim, 避免 langchain_anthropic 运行期导入失败。
        ensure_pydantic_v1_shim()
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(**self._params)
