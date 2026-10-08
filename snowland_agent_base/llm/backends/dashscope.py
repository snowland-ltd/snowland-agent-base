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
from typing import Dict, Optional

from langchain_core.language_models import BaseChatModel

from snowland_agent_base._compat import ensure_pydantic_v1_shim
from snowland_agent_base.llm.base import LLMBackend


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
